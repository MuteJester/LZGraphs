import logging
import time
from typing import List, Tuple, Union, Optional, Generator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .lz_graph_base import LZGraphBase
from ..utilities.decomposition import lempel_ziv_decomposition
from ..utilities.misc import window
from ..exceptions import NoValidPathError

# --------------------------------------------------------------------------
# Global Logger Configuration for NDPLZGraph
# --------------------------------------------------------------------------
logger = logging.getLogger(__name__)

__all__ = ["NDPLZGraph", "derive_lz_reading_frame_position"]

# --------------------------------------------------------------------------
# Helper Function
# --------------------------------------------------------------------------

def derive_lz_reading_frame_position(cdr3: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Given a nucleotide sequence (cdr3), this function returns:
      (lz_subpatterns, reading_frame_positions, cumulative_end_positions)

    - lz_subpatterns: LZ decomposition of the sequence (e.g., ["ATG", "C", "TAA"]).
    - reading_frame_positions: The reading frame (0,1,2) of each subpattern,
      based on the subpattern's start in the original sequence.
    - cumulative_end_positions: The cumulative end position of each subpattern
      (i.e., the total number of characters consumed up to and including that subpattern).
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
        data: Union[pd.DataFrame, List[str], pd.Series],
        *,
        abundances: Optional[List[int]] = None,
        v_genes: Optional[List[str]] = None,
        j_genes: Optional[List[str]] = None,
        verbose: bool = True,
        calculate_trainset_pgen: bool = False,
        smoothing_alpha: float = 0.0,
        min_initial_state_count: int = 5,
    ):
        """
        Create a nucleotide reading-frame-positional LZGraph.

        *data* can be a pandas DataFrame with a ``cdr3_rearrangement``
        column, a plain list of nucleotide sequences, or a pandas Series.

        When *data* is a list or Series the optional keyword arguments
        *abundances*, *v_genes* and *j_genes* may be used to supply
        additional per-sequence information.  When *data* is a DataFrame
        these must be ``None`` — use DataFrame columns instead.

        Args:
            data: Sequence data.  DataFrame (with ``cdr3_rearrangement``
                column), list of strings, or pandas Series.
            abundances: Per-sequence abundance counts (list input only).
            v_genes: Per-sequence V gene annotations (list input only).
            j_genes: Per-sequence J gene annotations (list input only).
            verbose: Whether to log progress info.
            calculate_trainset_pgen: If True, compute the walk_probability for
                each sequence in the dataset, storing results in self.train_pgen.
            smoothing_alpha: Laplace smoothing parameter for edge weights.
                0.0 means no smoothing (default).
            min_initial_state_count: Minimum observation count for initial states.
                States observed fewer times than this are excluded. Default is 5.
        """
        super().__init__()

        # Normalize flexible input → DataFrame
        data = self._normalize_input(
            data, "cdr3_rearrangement",
            abundances=abundances, v_genes=v_genes, j_genes=j_genes,
        )

        # PGEN configuration
        self.impute_missing_edges = True
        self.smoothing_alpha = smoothing_alpha
        self.min_initial_state_count = min_initial_state_count

        # Detect presence of V/J columns
        self.has_gene_data = ('V' in data.columns) and ('J' in data.columns)

        # If we have gene data, load and log
        if self.has_gene_data:
            self._load_gene_data(data)
            self.verbose_driver(0, verbose)  # "Gene Information Loaded"

        # Build the graph by iterating over data
        self.__simultaneous_graph_construction(data)
        self.verbose_driver(1, verbose)  # "Graph Constructed"

        # Normalize and derive probability dicts
        self.length_counts = dict(self.lengths)

        total_terminal = sum(self.terminal_state_counts.values())
        self.length_probabilities = (
            {k: v / total_terminal for k, v in self.terminal_state_counts.items()}
            if total_terminal > 0 else {}
        )

        # Filter rarely observed initial states
        self.initial_state_counts = {
            k: v for k, v in self.initial_state_counts.items()
            if v > self.min_initial_state_count
        }
        total_initial = sum(self.initial_state_counts.values())
        self.initial_state_probabilities = (
            {k: v / total_initial for k, v in self.initial_state_counts.items()}
            if total_initial > 0 else {}
        )

        self.verbose_driver(2, verbose)  # "Graph Metadata Derived"

        # Subpattern probabilities & edge weight normalization
        self._derive_node_probability()
        self.verbose_driver(8, verbose)

        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        # Additional map derivations
        self._edges_cache = None
        self._derive_stop_probability_data()
        self.verbose_driver(9, verbose)

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
    def extract_subpattern(base: str) -> str:
        """
        Given a sub-pattern that looks like "ATG0_3", extract only the nucleotides ("ATG").
        The format is {nucleotides}{frame_digit}_{position}, so split on '_' and
        drop the trailing frame digit.
        """
        prefix = base.split('_', 1)[0]
        return prefix[:-1] if len(prefix) > 1 else prefix

    # --------------------------------------------------------------------------
    # Graph-Building Methods
    # --------------------------------------------------------------------------

    def _decomposed_sequence_generator(self, data: Union[pd.DataFrame, pd.Series]) -> Generator:
        """
        Generates tuples for each row in the data.

        If an ``abundance`` column is present in the DataFrame, each sequence
        is weighted by its abundance count. Otherwise each sequence counts as 1.

        Yields:
            If genetic: (steps, reading_frames, locations, v, j, count)
            Otherwise:  (steps, reading_frames, locations, count)
        """
        has_abundance = isinstance(data, pd.DataFrame) and 'abundance' in data.columns

        if self.has_gene_data:
            iterables = [data["cdr3_rearrangement"], data["V"], data["J"]]
            if has_abundance:
                iterables.append(data["abundance"])
            for row in tqdm(zip(*iterables), desc="Building NDPLZGraph", leave=False):
                if has_abundance:
                    cdr3, v, j, abundance = row
                    count = int(abundance)
                else:
                    cdr3, v, j = row
                    count = 1

                subs, frames, cumpos = derive_lz_reading_frame_position(cdr3)
                steps = window(subs, 2)
                reading_frames = window(frames, 2)
                locations = window(cumpos, 2)

                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + count

                last_node = f"{subs[-1]}{frames[-1]}_{cumpos[-1]}"
                first_node = f"{subs[0]}{frames[0]}_{cumpos[0]}"

                self._update_terminal_states(last_node, count=count)
                self._update_initial_states(first_node, count=count)

                yield (steps, reading_frames, locations, v, j, count)
        else:
            if has_abundance:
                seq_iter = zip(data["cdr3_rearrangement"], data["abundance"])
            elif isinstance(data, pd.DataFrame):
                seq_iter = ((cdr3, 1) for cdr3 in data["cdr3_rearrangement"])
            else:
                seq_iter = ((cdr3, 1) for cdr3 in data)

            for cdr3, abundance in tqdm(seq_iter, desc="Building NDPLZGraph", leave=False):
                count = int(abundance)
                subs, frames, cumpos = derive_lz_reading_frame_position(cdr3)
                steps = window(subs, 2)
                reading_frames = window(frames, 2)
                locations = window(cumpos, 2)

                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + count

                last_node = f"{subs[-1]}{frames[-1]}_{cumpos[-1]}"
                first_node = f"{subs[0]}{frames[0]}_{cumpos[0]}"

                self._update_terminal_states(last_node, count=count)
                self._update_initial_states(first_node, count=count)

                yield (steps, reading_frames, locations, count)

    def __simultaneous_graph_construction(self, data: pd.DataFrame) -> None:
        """
        Custom routine to build the NDPLZGraph from the data.
        """
        logger.debug("Starting __simultaneous_graph_construction for NDPLZGraph...")
        processing_stream = self._decomposed_sequence_generator(data)
        freq = self.node_outgoing_counts  # local ref for speed

        if self.has_gene_data:
            insert = self._insert_edge_and_information
            for steps, reading_frames, locations, v, j, count in processing_stream:
                for (A, B), (pos_a, pos_b), (loc_a, loc_b) in zip(steps, reading_frames, locations):
                    A_ = f"{A}{pos_a}_{loc_a}"
                    freq[A_] = freq.get(A_, 0) + count
                    B_ = f"{B}{pos_b}_{loc_b}"
                    insert(A_, B_, v, j, count=count)
                freq[B_] = freq.get(B_, 0)
        else:
            insert = self._insert_edge_and_information_no_genes
            for gen_tuple in processing_stream:
                steps, reading_frames, locations, count = gen_tuple
                for (A, B), (pos_a, pos_b), (loc_a, loc_b) in zip(steps, reading_frames, locations):
                    A_ = f"{A}{pos_a}_{loc_a}"
                    freq[A_] = freq.get(A_, 0) + count
                    B_ = f"{B}{pos_b}_{loc_b}"
                    insert(A_, B_, count=count)
                freq[B_] = freq.get(B_, 0)

        logger.debug("Finished __simultaneous_graph_construction.")

    # --------------------------------------------------------------------------
    # Probability / Random Walk Methods
    # --------------------------------------------------------------------------

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
            terminal_states = list(self.terminal_state_counts.keys())
            if initial_state is None:
                initial_state = self._random_initial_state()
        else:
            terminal_states = self._length_specific_terminal_state(seq_len)

        if initial_state is None:
            raise NoValidPathError(message="No initial state provided for gene_random_walk.")

        current_state = initial_state
        walk = [initial_state]

        if self._walk_exclusions is None:
            self._walk_exclusions = {}

        while current_state not in terminal_states:
            if current_state not in self.graph:
                logger.warning(f"Current state {current_state} not found in graph; ending walk.")
                break

            # Get outgoing edges
            edges = {nb: self.graph[current_state][nb]['data'] for nb in self.graph[current_state]}
            # Apply blacklisting
            if (current_state, selected_v, selected_j) in self._walk_exclusions:
                blacklisted_cols = self._walk_exclusions[(current_state, selected_v, selected_j)]
                edges = {nb: ed for nb, ed in edges.items() if nb not in blacklisted_cols}

            # Filter to edges containing both V and J genes
            valid_edges = {nb: ed for nb, ed in edges.items()
                           if ed.has_gene(selected_v) and ed.has_gene(selected_j)}

            if not valid_edges:
                if len(walk) > 2:
                    prev_state = walk[-2]
                    self._walk_exclusions[(prev_state, selected_v, selected_j)] = \
                        self._walk_exclusions.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                    current_state = prev_state
                    walk.pop()
                else:
                    walk = [walk[0]]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)
                continue

            nbs = list(valid_edges.keys())
            weights = np.array([valid_edges[nb].weight for nb in nbs])
            w_sum = weights.sum()
            if w_sum == 0:
                if len(walk) > 2:
                    prev_state = walk[-2]
                    self._walk_exclusions[(prev_state, selected_v, selected_j)] = \
                        self._walk_exclusions.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                    current_state = prev_state
                    walk.pop()
                else:
                    walk = [walk[0]]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)
                continue

            weights /= w_sum
            current_state = np.random.choice(nbs, p=weights)
            walk.append(current_state)

        return walk, selected_v, selected_j

