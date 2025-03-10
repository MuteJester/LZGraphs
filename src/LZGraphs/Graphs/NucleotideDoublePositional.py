import logging
import re
import time
from typing import List, Tuple, Union, Generator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Adjust the relative import paths as needed
from .LZGraphBase import LZGraphBase
from ..Utilities.decomposition import lempel_ziv_decomposition
from ..Utilities.misc import window, choice

# --------------------------------------------------------------------------
# Global Logger Configuration for NDPLZGraph
# --------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# If no handlers are attached, attach one so users see logs by default
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------

def derive_lz_reading_frame_position(cdr3: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Given a nucleotide sequence (cdr3), this function returns:
      (lz_subpatterns, reading_frame_positions, start_positions)

    - lz_subpatterns: LZ decomposition of the sequence (e.g., ["ATG", "C", "TAA"]).
    - reading_frame_positions: The reading frame (0,1,2) of each subpattern,
      based on the subpattern's start in the original sequence.
    - start_positions: The cumulative start positions of each subpattern.
    """
    lzc = lempel_ziv_decomposition(cdr3)
    cumulative_positions = []
    reading_frames = []
    total_len = 0

    for sp in lzc:
        # sp is a subpattern (e.g., "ATG")
        start_pos = total_len
        reading_frame = start_pos % 3
        reading_frames.append(reading_frame)
        total_len += len(sp)
        cumulative_positions.append(total_len)
    return lzc, reading_frames, cumulative_positions

# --------------------------------------------------------------------------
# NDPLZGraph Class
# --------------------------------------------------------------------------

class NDPLZGraph(LZGraphBase):
    """
    This class implements the "Nucleotide Double Positional" version of the LZGraph,
    suitable for analyzing nucleotide sequences. Each node has the format:
       {LZ_subpattern}{reading_frame_start}_{start_position_in_sequence},
    for example: "ATG0_3" might mean the subpattern "ATG", reading frame 0,
    starting at position 3 in the overall sequence.

    The class inherits from LZGraphBase and thus uses the same random-walk logic,
    gene annotation logic (if present), and so on.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
        calculate_trainset_pgen: bool = False
    ):
        """
        Constructor for NDPLZGraph.

        Args:
            data (pd.DataFrame): Must include at least a column 'cdr3_rearrangement'
                with the nucleotide sequences. If 'V' and 'J' columns exist, we embed
                gene information (self.genetic=True).
            verbose (bool): Whether to log progress info.
            calculate_trainset_pgen (bool): If True, compute the walk_probability for
                each sequence in the dataset, storing results in self.train_pgen.
        """
        super().__init__()

        # Detect presence of V/J columns
        if isinstance(data, pd.DataFrame) and ('V' in data.columns) and ('J' in data.columns):
            self.genetic = True
        else:
            self.genetic = False

        # If we have gene data, load and log
        if self.genetic:
            self._load_gene_data(data)
            self.verbose_driver(0, verbose)  # "Gene Information Loaded"

        # Build the graph by iterating over data
        self.__simultaneous_graph_construction(data)
        self.verbose_driver(1, verbose)  # "Graph Constructed"

        # Convert dictionaries to Series and normalize
        self.length_distribution = pd.Series(self.lengths)
        self.terminal_states = pd.Series(self.terminal_states)
        self.initial_states = pd.Series(self.initial_states)

        self.length_distribution_proba = self.terminal_states / self.terminal_states.sum()

        # Filter rarely observed initial states (example: >5)
        self.initial_states = self.initial_states[self.initial_states > 5]
        self.initial_states_probability = self.initial_states / self.initial_states.sum()

        self.verbose_driver(2, verbose)  # "Graph Metadata Derived"

        # Subpattern probabilities & edge weight normalization
        self._derive_subpattern_individual_probability()
        self.verbose_driver(8, verbose)

        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        # If gene data is present, batch-normalize gene weights
        if self.genetic:
            self._batch_gene_weight_normalization(n_process=3, verbose=verbose)
            self.verbose_driver(4, verbose)

        # Additional map derivations
        self.edges_list = None
        self._derive_terminal_state_map()
        self.verbose_driver(7, verbose)
        self._derive_stop_probability_data()
        self.verbose_driver(8, verbose)
        self.verbose_driver(5, verbose)

        # Mark constructor end time
        self.constructor_end_time = time.time()
        self.verbose_driver(6, verbose)

        # Optionally compute PGEN for each sequence
        if calculate_trainset_pgen:
            logger.info("Calculating trainset PGEN for NDPLZGraph. This may take time...")
            self.train_pgen = np.array([
                self.walk_probability(self.encode_sequence(seq), verbose=False)
                for seq in data["cdr3_rearrangement"]
            ])

        # Done
        self.verbose_driver(-2, verbose)

    # --------------------------------------------------------------------------
    # Node Format: {LZ_subpattern}{reading_frame}_{start_position}
    # --------------------------------------------------------------------------

    @staticmethod
    def encode_sequence(cdr3: str) -> List[str]:
        """
        Encode a nucleotide sequence (cdr3) into the NDPLZGraph format:
          {lz_subpattern}{reading_frame}_{start_position}

        Example: If cdr3="ATGCG", the function might yield subpatterns:
          - "ATG" (frame=0, pos=0)
          - "CG"  (frame=0, pos=3)
          resulting in nodes "ATG0_3", "CG0_5", etc.
        """
        subs, frames, positions = derive_lz_reading_frame_position(cdr3)
        return [
            f"{sub}{frame}_{pos}"
            for sub, frame, pos in zip(subs, frames, positions)
        ]

    @staticmethod
    def clean_node(base: str) -> str:
        """
        Given a sub-pattern that looks like "ATG0_3", extract only the nucleotides ("ATG").
        """
        match = re.search(r"[ATGC]+", base)
        return match.group(0) if match else ""

    # --------------------------------------------------------------------------
    # Graph-Building Methods
    # --------------------------------------------------------------------------

    def _decomposed_sequence_generator(self, data: Union[pd.DataFrame, pd.Series]) -> Generator:
        """
        Generates tuples for each row in the data:
          - If self.genetic=True, yields (steps, reading_frames, locations, v, j)
          - Else, yields (steps, reading_frames, locations)
        Each row is used to build edges in the graph.
        """
        if self.genetic:
            # We have columns: 'cdr3_rearrangement', 'V', 'J'
            for cdr3, v, j in tqdm(
                zip(data["cdr3_rearrangement"], data["V"], data["J"]),
                desc="Building NDPLZGraph",
                leave=False
            ):
                subs, frames, cumpos = derive_lz_reading_frame_position(cdr3)
                steps = window(subs, 2)
                reading_frames = window(frames, 2)
                locations = window(cumpos, 2)

                # Track length distribution
                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + 1

                # Update terminal & initial states
                last_node = f"{subs[-1]}{frames[-1]}_{cumpos[-1]}"
                first_node = f"{subs[0]}{frames[0]}_1"

                self._update_terminal_states(last_node)
                self._update_initial_states(first_node)

                yield (steps, reading_frames, locations, v, j)
        else:
            # We only have 'cdr3_rearrangement' column
            if isinstance(data, pd.DataFrame):
                seq_iterable = data["cdr3_rearrangement"]
            else:
                seq_iterable = data

            for cdr3 in tqdm(seq_iterable, desc="Building NDPLZGraph", leave=False):
                subs, frames, cumpos = derive_lz_reading_frame_position(cdr3)
                steps = window(subs, 2)
                reading_frames = window(frames, 2)
                locations = window(cumpos, 2)

                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + 1

                last_node = f"{subs[-1]}{frames[-1]}_{cumpos[-1]}"
                first_node = f"{subs[0]}{frames[0]}_1"

                self._update_terminal_states(last_node)
                self._update_initial_states(first_node)

                yield (steps, reading_frames, locations)

    def __simultaneous_graph_construction(self, data: pd.DataFrame) -> None:
        """
        Custom routine to build the NDPLZGraph from the data.
        """
        logger.debug("Starting __simultaneous_graph_construction for NDPLZGraph...")
        processing_stream = self._decomposed_sequence_generator(data)

        if self.genetic:
            for steps, reading_frames, locations, v, j in processing_stream:
                for (A, B), (pos_a, pos_b), (loc_a, loc_b) in zip(steps, reading_frames, locations):
                    A_ = f"{A}{pos_a}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1

                    B_ = f"{B}{pos_b}_{loc_b}"
                    self._insert_edge_and_information(A_, B_, v, j)

                # Ensure the final node is accounted for
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)
        else:
            for gen_tuple in processing_stream:
                steps, reading_frames, locations = gen_tuple
                for (A, B), (pos_a, pos_b), (loc_a, loc_b) in zip(steps, reading_frames, locations):
                    A_ = f"{A}{pos_a}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1

                    B_ = f"{B}{pos_b}_{loc_b}"
                    self._insert_edge_and_information_no_genes(A_, B_)
                # Ensure the final node is accounted for
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)

        logger.debug("Finished __simultaneous_graph_construction.")

    # --------------------------------------------------------------------------
    # Probability / Random Walk Methods
    # --------------------------------------------------------------------------

    def walk_probability(self, walk: Union[str, List[str]], verbose: bool = True) -> float:
        """
        Return the probability (PGEN) of generating the given walk on the NDPLZGraph.

        If 'walk' is a string, we convert it to sub-patterns. If an edge is missing,
        we log a warning (if verbose=True) and return 0.
        """
        if isinstance(walk, str):
            # Convert raw sequence
            subs, frames, cumpos = derive_lz_reading_frame_position(walk)
            walk_ = [f"{s}{f}" for s, f in zip(subs, cumpos)]
            # Actually, your code used s+pos but we have reading frames, so let's match your logic:
            # walk_ = [i + str(j) for i, j in zip(subs, cumpos)]
            # That said, the original code might be a slight mismatch. We'll keep your original approach for minimal changes:
            walk_ = [f"{sub}{pos}" for sub, pos in zip(subs, cumpos)]
        else:
            walk_ = walk

        if len(walk_) == 0:
            logger.warning("Empty walk passed to walk_probability. Returning 0.")
            return 0.0

        # Probability starts with the subpattern probability of the first node
        first_node = walk_[0]
        if first_node not in self.subpattern_individual_probability["proba"]:
            if verbose:
                logger.warning(f"First node {first_node} not recognized; returning 0.")
            return 0.0

        proba = self.subpattern_individual_probability["proba"][first_node]
        for step1, step2 in window(walk_, 2):
            if self.graph.has_edge(step1, step2):
                edge_weight = self.graph[step1][step2]["weight"]
                proba *= edge_weight
            else:
                if verbose:
                    logger.warning(f"No Edge Connecting {step1} -> {step2}. Returning 0.")
                return 0.0

        return proba

    def gene_random_walk(
        self,
        seq_len: Union[int, str],
        initial_state: Union[str, None] = None,
        vj_init: str = "marginal"
    ):
        """
        Generate a random walk constrained by a target length (or 'unsupervised') and
        selected V/J genes. If seq_len is 'unsupervised', we pick a random length from
        observed terminal states. Otherwise, we aim to match seq_len.

        The random walk ensures each edge has the specified (V, J).
        """
        selected_v, selected_j = self._select_random_vj_genes(vj_init)

        if seq_len == "unsupervised":
            terminal_states = self.terminal_states.index
            if initial_state is None:
                initial_state = self._random_initial_state()
        else:
            terminal_states = self._length_specific_terminal_state(seq_len)

        if initial_state is None:
            raise ValueError("No initial state provided for gene_random_walk.")

        current_state = initial_state
        walk = [initial_state]

        if self.genetic_walks_black_list is None:
            self.genetic_walks_black_list = {}

        while current_state not in terminal_states:
            if current_state not in self.graph:
                logger.warning(f"Current state {current_state} not found in graph; ending walk.")
                break

            edge_info = pd.DataFrame(dict(self.graph[current_state]))
            # Apply blacklisting
            if (current_state, selected_v, selected_j) in self.genetic_walks_black_list:
                blacklisted_cols = self.genetic_walks_black_list[(current_state, selected_v, selected_j)]
                edge_info = edge_info.drop(columns=blacklisted_cols, errors="ignore")

            # Check if edges have both selected_v and selected_j
            if not {selected_v, selected_j}.issubset(edge_info.index):
                # If we can't find both genes among edges
                if len(walk) > 2:
                    prev_state = walk[-2]
                    self.genetic_walks_black_list[(prev_state, selected_v, selected_j)] = \
                        self.genetic_walks_black_list.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                    current_state = prev_state
                    walk.pop()
                else:
                    walk = [walk[0]]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)
                continue

            # If we do have them, filter by edges that contain V/J
            sub_df = edge_info.T[[selected_v, selected_j]].dropna(how="any")
            if sub_df.empty:
                # No valid edges
                if len(walk) > 2:
                    prev_state = walk[-2]
                    self.genetic_walks_black_list[(prev_state, selected_v, selected_j)] = \
                        self.genetic_walks_black_list.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                    current_state = prev_state
                    walk.pop()
                else:
                    walk = [walk[0]]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)
                continue

            weights = edge_info.loc["weight", sub_df.index]
            if weights.sum() == 0:
                if len(walk) > 2:
                    prev_state = walk[-2]
                    self.genetic_walks_black_list[(prev_state, selected_v, selected_j)] = \
                        self.genetic_walks_black_list.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                    current_state = prev_state
                    walk.pop()
                else:
                    walk = [walk[0]]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)
                continue

            weights = weights / weights.sum()
            current_state = np.random.choice(weights.index, p=weights.values)
            walk.append(current_state)

        return walk, selected_v, selected_j

    def unsupervised_random_walk(self):
        """
        A random initial state is chosen, and the walk proceeds until
        we hit a recognized terminal state. Gene constraints are not applied.
        Returns (walk, sequence_str).
        """
        init_state = self._random_initial_state()
        walk = [init_state]
        sequence = self.clean_node(init_state)

        while walk[-1] not in self.terminal_states:
            next_state = self.random_step(walk[-1])
            walk.append(next_state)
            sequence += self.clean_node(next_state)

        return walk, sequence

    # --------------------------------------------------------------------------
    # Gene & Variation Analysis
    # --------------------------------------------------------------------------

    def walk_genes(self, walk: List[str], dropna: bool = True) -> pd.DataFrame:
        """
        Given a walk (list of node IDs), return a DataFrame representing the
        possible genes and their probabilities at each edge in the walk.

        If 'dropna=True', rows (genes) with all-NaN across columns are dropped.
        """
        columns = []
        edge_data_list = []

        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ed = self.graph[walk[i]][walk[i + 1]].copy()
                # Remove special keys
                for key in ("weight", "Vsum", "Jsum"):
                    ed.pop(key, None)
                edge_data_list.append(pd.Series(ed))
                columns.append(f"{walk[i]}->{walk[i+1]}")
            else:
                logger.warning(f"Edge missing for {walk[i]} -> {walk[i+1]}. Skipping...")

        if not edge_data_list:
            raise Exception("No valid edges found in walk for gene analysis.")

        df = pd.concat(edge_data_list, axis=1)
        df.columns = columns

        if dropna:
            df.dropna(how="all", inplace=True)

        if df.empty:
            raise Exception("No gene data found after dropping all-NaN rows.")

        # For clarity, mark gene type and sum
        df["type"] = df.index.to_series().apply(
            lambda x: "V" if "v" in x.lower() else ("J" if "j" in x.lower() else "Unknown")
        )
        df["sum"] = df.sum(axis=1, numeric_only=True)
        return df

    def sequence_variation_curve(self, cdr3_sample: str) -> Tuple[List[str], List[int]]:
        """
        Return two lists:
          - The encoded subpatterns for cdr3_sample.
          - The out-degree of each subpattern in the graph.
        """
        encoded = self.encode_sequence(cdr3_sample)
        curve = [self.graph.out_degree(node) for node in encoded]
        return encoded, curve

    def path_gene_table(
        self,
        cdr3_sample: str,
        threshold: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return two tables (V genes, J genes) that could generate the given cdr3_sample.
        Genes missing from more than 'threshold' edges in the walk are dropped.

        If threshold=None, we default to length/4 for V and length/2 for J.
        """
        encoded = self.encode_sequence(cdr3_sample)
        length = len(encoded)
        if threshold is None:
            threshold_v = length * 0.25
            threshold_j = length * 0.5
        else:
            threshold_v = threshold
            threshold_j = threshold

        # V Genes
        gene_table_v = self.walk_genes(encoded, dropna=False)
        mask_v = gene_table_v.isna().sum(axis=1) < threshold_v
        vgene_table = gene_table_v[mask_v & gene_table_v.index.str.contains("V")]

        # J Genes
        gene_table_j = self.walk_genes(encoded, dropna=False)
        mask_j = gene_table_j.isna().sum(axis=1) < threshold_j
        jgene_table = gene_table_j[mask_j & gene_table_j.index.str.contains("J")]

        # Sort them by ascending NaN count (just for convenience)
        jgene_table = jgene_table.loc[jgene_table.isna().sum(axis=1).sort_values().index]
        vgene_table = vgene_table.loc[vgene_table.isna().sum(axis=1).sort_values().index]

        return vgene_table, jgene_table

    def gene_variation(self, cdr3: str) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing how many V-genes and J-genes are possible
        at each subpattern of the given cdr3.

        Raises Exception if no gene data is available (self.genetic=False).
        """
        if not self.genetic:
            raise Exception("The LZGraph Has No Gene Data (genetic=False).")

        encoded_nodes = self.encode_sequence(cdr3)

        # The first subpattern can have all observed V/J genes in marginal usage
        n_v = [len(self.marginal_vgenes)]
        n_j = [len(self.marginal_jgenes)]

        for node in encoded_nodes[1:]:
            in_edges = self.graph.in_edges(node)
            v_candidates = set()
            j_candidates = set()

            for ea, eb in in_edges:
                ed = pd.Series(self.graph[ea][eb]).drop(["Vsum", "Jsum", "weight"], errors="ignore")
                v_candidates |= set(g for g in ed.index if g.startswith("V"))
                j_candidates |= set(g for g in ed.index if g.startswith("J"))

            n_v.append(len(v_candidates))
            n_j.append(len(j_candidates))

        from ..Utilities.decomposition import lempel_ziv_decomposition  # if needed here
        lz_subpatterns = lempel_ziv_decomposition(cdr3)

        df = pd.DataFrame({
            "genes": n_v + n_j,
            "type": (["V"] * len(n_v)) + (["J"] * len(n_j)),
            "sp": lz_subpatterns + lz_subpatterns
        })
        return df
