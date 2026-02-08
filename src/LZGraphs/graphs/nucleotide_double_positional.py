import logging
import re
import time
from typing import List, Tuple, Union, Generator

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
        data: pd.DataFrame,
        verbose: bool = True,
        calculate_trainset_pgen: bool = False,
        smoothing_alpha: float = 0.0,
        initial_state_threshold: int = 5,
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
            smoothing_alpha (float): Laplace smoothing parameter for edge weights.
                0.0 means no smoothing (default).
            initial_state_threshold (int): Minimum observation count for initial states.
                States observed fewer times than this are excluded. Default is 5.
        """
        super().__init__()

        # PGEN configuration
        self.impute_missing_edges = True
        self.smoothing_alpha = smoothing_alpha
        self.initial_state_threshold = initial_state_threshold

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

        # Filter rarely observed initial states
        self.initial_states = self.initial_states[self.initial_states > self.initial_state_threshold]
        self.initial_states_probability = self.initial_states / self.initial_states.sum()

        self.verbose_driver(2, verbose)  # "Graph Metadata Derived"

        # Subpattern probabilities & edge weight normalization
        self._derive_subpattern_individual_probability()
        self.verbose_driver(8, verbose)

        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        # If gene data is present, batch-normalize gene weights
        if self.genetic:
            self._batch_gene_weight_normalization(verbose=verbose)
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
                first_node = f"{subs[0]}{frames[0]}_{cumpos[0]}"

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
                first_node = f"{subs[0]}{frames[0]}_{cumpos[0]}"

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
            raise NoValidPathError(message="No initial state provided for gene_random_walk.")

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

