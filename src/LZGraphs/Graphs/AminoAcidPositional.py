import logging
import re
import time
from collections import Counter
from typing import List, Tuple, Union, Optional, Generator

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Replace these imports with the correct paths in your package
from .LZGraphBase import LZGraphBase
from ..Utilities.decomposition import lempel_ziv_decomposition
from ..Utilities.misc import window, choice
from ..Exceptions import (
    EmptyDataError,
    MissingColumnError,
    InvalidSequenceError,
    NoGeneDataError,
    GeneAnnotationError,
)

# --------------------------------------------------------------------------
# Global Logger Configuration
# --------------------------------------------------------------------------
# Configure logging so that users see log messages without setting it up themselves
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def derive_lz_and_position(cdr3_sequence: str) -> Tuple[List[str], List[int]]:
    """
    Decompose a CDR3 amino acid sequence into its LZ subpatterns along with
    cumulative positions. For example, "ABCDE" might become (["AB", "CD", "E"], [2,4,5]).
    """
    lz_subpatterns = lempel_ziv_decomposition(cdr3_sequence)
    cumulative_lengths = []
    total_length = 0
    for subpattern in lz_subpatterns:
        total_length += len(subpattern)
        cumulative_lengths.append(total_length)
    return lz_subpatterns, cumulative_lengths


def path_to_sequence(lz_subpatterns: List[str]) -> str:
    """
    Given a list of LZ subpatterns with positions attached, clean them to remove the
    numeric part and return a single concatenated amino acid sequence.
    """
    cleaned_nodes = [AAPLZGraph.clean_node(sp) for sp in lz_subpatterns]
    return ''.join(cleaned_nodes)

# --------------------------------------------------------------------------
# The AAPLZGraph Class
# --------------------------------------------------------------------------

class AAPLZGraph(LZGraphBase):
    """
    Implements the "Amino Acid Positional" version of the LZGraph for analyzing
    amino-acid sequences, especially for immunological data.

    Each node is labeled as:
        {LZ_subpattern}_{start_position_in_sequence}
    """

    # Valid amino acid characters (standard 20 amino acids)
    VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')

    def __init__(
        self,
        data: pd.DataFrame,
        verbose: bool = True,
        calculate_trainset_pgen: bool = False,
        validate_sequences: bool = True
    ):
        """
        Create an amino-acid-positional LZGraph from a DataFrame.

        The DataFrame must contain at least a column "cdr3_amino_acid".
        Optionally, columns "V" and "J" may also be provided to embed
        gene information. If these columns are present, self.genetic is set to True.

        Args:
            data (pd.DataFrame): Input data for constructing the graph. Must contain
                a "cdr3_amino_acid" column; optionally "V" and "J" columns.
            verbose (bool): Whether to log progress information.
            calculate_trainset_pgen (bool): If True, compute PGEN for each sequence in `data`.
            validate_sequences (bool): If True, validate that sequences contain only
                standard amino acids. Set to False to skip validation for performance.

        Raises:
            TypeError: If data is not a pandas DataFrame.
            ValueError: If required columns are missing or sequences are invalid.
        """
        super().__init__()  # Initialize LZGraphBase

        # Input validation
        self._validate_input(data, validate_sequences)

        # Determine if we have gene data
        self.genetic = (
            isinstance(data, pd.DataFrame) and
            ("V" in data.columns) and
            ("J" in data.columns)
        )

        # Load gene data if present
        if self.genetic:
            self._load_gene_data(data)
            self.verbose_driver(0, verbose)  # "Gene Information Loaded"

        # Build the graph with a custom routine
        self.__simultaneous_graph_construction(data)
        self.verbose_driver(1, verbose)  # "Graph Constructed"

        # Convert dicts to Series and normalize
        self.length_distribution = pd.Series(self.lengths)
        self.terminal_states = pd.Series(self.terminal_states)
        self.initial_states = pd.Series(self.initial_states)

        self.length_distribution_proba = self.terminal_states / self.terminal_states.sum()

        # Filter out rarely observed initial states (for example, those <= 5)
        self.initial_states = self.initial_states[self.initial_states > 5]
        self.initial_states_probability = self.initial_states / self.initial_states.sum()

        self.verbose_driver(2, verbose)  # "Graph Metadata Derived"

        # Derive subpattern probabilities & normalize edges
        self._derive_subpattern_individual_probability()
        self.verbose_driver(8, verbose)

        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        # If gene data is available, normalize gene weights in parallel
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

        # Optionally compute the PGEN for each sequence
        if calculate_trainset_pgen:
            logger.info("Calculating PGEN for the training set. This may take some time...")
            self.train_pgen = np.array([
                self.walk_probability(seq, verbose=False)
                for seq in data["cdr3_amino_acid"]
            ])

        self.constructor_end_time = time.time()
        self.verbose_driver(6, verbose)
        self.verbose_driver(-2, verbose)

    # --------------------------------------------------------------------------
    # Input Validation
    # --------------------------------------------------------------------------

    def _validate_input(self, data: pd.DataFrame, validate_sequences: bool) -> None:
        """
        Validate input data before graph construction.

        Args:
            data: Input DataFrame
            validate_sequences: Whether to check sequence content

        Raises:
            TypeError: If data is not a pandas DataFrame
            ValueError: If required columns are missing or data is invalid
        """
        # Check type
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"Expected pandas DataFrame, got {type(data).__name__}. "
                "Please provide a DataFrame with a 'cdr3_amino_acid' column."
            )

        # Check for required column
        if 'cdr3_amino_acid' not in data.columns:
            raise MissingColumnError(
                column_name='cdr3_amino_acid',
                available_columns=list(data.columns)
            )

        # Check for empty data
        if len(data) == 0:
            raise EmptyDataError("DataFrame is empty. Cannot build LZGraph from zero sequences.")

        # Check for null values in CDR3 column
        null_count = data['cdr3_amino_acid'].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"Found {null_count} null values in 'cdr3_amino_acid' column. "
                "Please remove or fill null values before building the graph."
            )

        # Check for empty strings
        empty_count = (data['cdr3_amino_acid'].str.len() == 0).sum()
        if empty_count > 0:
            raise ValueError(
                f"Found {empty_count} empty strings in 'cdr3_amino_acid' column. "
                "Please remove empty sequences before building the graph."
            )

        # Validate sequence content if requested
        if validate_sequences:
            self._validate_sequence_content(data['cdr3_amino_acid'])

        # Validate gene columns if present
        if 'V' in data.columns and 'J' in data.columns:
            self._validate_gene_columns(data)

    def _validate_sequence_content(self, sequences: pd.Series) -> None:
        """
        Validate that sequences contain only valid amino acid characters.

        Args:
            sequences: Series of amino acid sequences

        Raises:
            ValueError: If invalid characters are found
        """
        # Sample up to 1000 sequences for validation (performance)
        sample_size = min(1000, len(sequences))
        sample = sequences.sample(n=sample_size, random_state=42) if len(sequences) > sample_size else sequences

        invalid_chars_found = set()
        invalid_sequences = []

        for seq in sample:
            if not isinstance(seq, str):
                raise InvalidSequenceError(
                    sequence=str(seq),
                    message=f"Sequence must be a string, got {type(seq).__name__}: {seq}"
                )
            invalid_in_seq = set(seq.upper()) - self.VALID_AMINO_ACIDS
            if invalid_in_seq:
                invalid_chars_found.update(invalid_in_seq)
                if len(invalid_sequences) < 3:
                    invalid_sequences.append(seq)

        if invalid_chars_found:
            examples = ", ".join(f"'{s}'" for s in invalid_sequences[:3])
            raise InvalidSequenceError(
                sequence=invalid_sequences[0] if invalid_sequences else None,
                invalid_chars=''.join(sorted(invalid_chars_found)),
                message=(
                    f"Found invalid amino acid characters: {sorted(invalid_chars_found)}. "
                    f"Valid amino acids are: {sorted(self.VALID_AMINO_ACIDS)}. "
                    f"Example invalid sequences: {examples}"
                )
            )

    def _validate_gene_columns(self, data: pd.DataFrame) -> None:
        """
        Validate V and J gene columns.

        Args:
            data: DataFrame with V and J columns

        Raises:
            ValueError: If gene columns contain invalid data
        """
        # Check for nulls in gene columns
        v_nulls = data['V'].isna().sum()
        j_nulls = data['J'].isna().sum()

        if v_nulls > 0 or j_nulls > 0:
            raise ValueError(
                f"Found null values in gene columns: V has {v_nulls} nulls, "
                f"J has {j_nulls} nulls. Please fill or remove rows with missing genes."
            )

    # --------------------------------------------------------------------------
    # Overridden / specialized methods
    # --------------------------------------------------------------------------

    @staticmethod
    def encode_sequence(amino_acid: str) -> List[str]:
        """
        Convert an amino acid string into LZ sub-patterns with positions.
        Each sub-pattern has the format: '{LZ_subpattern}_{position}'.
        """
        lz, locs = derive_lz_and_position(amino_acid)
        return [f"{subp}_{pos}" for subp, pos in zip(lz, locs)]

    @staticmethod
    def clean_node(base: str) -> str:
        """
        Given a sub-pattern that might look like "ABC_10", extract only the amino acids ("ABC").
        """
        match = re.search(r'[A-Z]+', base)
        return match.group(0) if match else ""

    def _decomposed_sequence_generator(
        self,
        data: Union[pd.DataFrame, pd.Series]
    ) -> Generator:
        """
        A generator that yields the information needed to build the graph:
        (steps, locations, v, j) if self.genetic == True, otherwise (steps, locations).
        """
        if self.genetic:
            # DataFrame with cdr3_amino_acid, V, J columns
            for cdr3, v, j in tqdm(
                zip(data["cdr3_amino_acid"], data["V"], data["J"]),
                desc="Building Graph",
                leave=False
            ):
                lz, locs = derive_lz_and_position(cdr3)
                steps = window(lz, 2)
                locations = window(locs, 2)

                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + 1
                self._update_terminal_states(f"{lz[-1]}_{locs[-1]}")
                self._update_initial_states(f"{lz[0]}_1")

                yield (steps, locations, v, j)
        else:
            # Possibly just a "cdr3_amino_acid" column
            seq_iter = data["cdr3_amino_acid"] if isinstance(data, pd.DataFrame) else data
            for cdr3 in tqdm(seq_iter, desc="Building Graph", leave=False):
                lz, locs = derive_lz_and_position(cdr3)
                steps = window(lz, 2)
                locations = window(locs, 2)

                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + 1
                self._update_terminal_states(f"{lz[-1]}_{locs[-1]}")
                self._update_initial_states(f"{lz[0]}_1")

                yield (steps, locations)

    def __simultaneous_graph_construction(self, data: pd.DataFrame) -> None:
        """
        Custom simultaneous construction of the graph, mirroring the parent's
        _simultaneous_graph_construction but applying our specialized decomposition.
        """
        logger.debug("Starting custom __simultaneous_graph_construction...")
        processing_stream = self._decomposed_sequence_generator(data)
        if self.genetic:
            for steps, locations, v, j in processing_stream:
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information(A_, B_, v, j)
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)
        else:
            for steps, locations in processing_stream:
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information_no_genes(A_, B_)
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)

        logger.debug("Finished custom __simultaneous_graph_construction.")

    # --------------------------------------------------------------------------
    # Probability / Gene-Related Methods
    # --------------------------------------------------------------------------

    def walk_probability(
        self,
        walk: Union[str, List[str]],
        verbose: bool = True,
        use_epsilon: bool = False,
        use_log: bool = False
    ) -> float:
        """
        Given a walk (a sequence or a pre-encoded LZ pattern list), return
        the probability (PGEN) of generating it under this graph.

        If edges are missing, we handle them by a geometric mean approach.
        If verbose=True, log warnings on missing edges.

        Args:
            walk: The walk as a string or list of sub-patterns.
            verbose: Whether to log missing-edge warnings.
            use_epsilon: Not used in the main logic here, but kept for consistency.
            use_log: If True, return log-probability instead of probability.
                     Recommended for long sequences (>30 amino acids) to prevent
                     numerical underflow. Default is False for backward compatibility.

        Returns:
            float: Probability of generating the walk, or log-probability if use_log=True.
                   For log-probability mode, returns log(P) where P is the probability.
        """
        # If the user passed a raw sequence, encode it
        if isinstance(walk, str):
            lz, locs = derive_lz_and_position(walk)
            walk_ = [f"{subp}_{pos}" for subp, pos in zip(lz, locs)]
        else:
            walk_ = walk

        if len(walk_) == 0:
            logger.warning("Empty walk provided to walk_probability. Returning eps.")
            return np.log(np.finfo(float).eps) if use_log else np.finfo(float).eps

        # If the first subpattern isn't observed, return near-zero
        first_node = walk_[0]
        if first_node not in self.subpattern_individual_probability['proba']:
            eps_val = np.finfo(float).eps ** 2
            return np.log(eps_val) if use_log else eps_val

        missing_count = 0
        total_steps = 0

        if use_log:
            # Log-space computation to prevent underflow
            log_proba = np.log(self.subpattern_individual_probability['proba'][first_node])

            for step1, step2 in window(walk_, 2):
                if self.graph.has_edge(step1, step2):
                    edge_weight = self.graph[step1][step2]["weight"]
                    log_proba += np.log(edge_weight)
                else:
                    if verbose:
                        logger.warning(f"No Edge Connecting: {step1} --> {step2}. Probability adjusted.")
                    missing_count += 1
                total_steps += 1

            if missing_count > 0 and total_steps > 0:
                # Geometric mean approach in log-space
                # gmean = proba^(1/total_steps) => log(gmean) = log_proba / total_steps
                log_gmean = log_proba / total_steps
                log_proba += log_gmean * missing_count

            return log_proba
        else:
            # Original probability-space computation
            proba = self.subpattern_individual_probability['proba'][first_node]

            for step1, step2 in window(walk_, 2):
                if self.graph.has_edge(step1, step2):
                    edge_weight = self.graph[step1][step2]["weight"]
                    proba *= edge_weight
                else:
                    if verbose:
                        logger.warning(f"No Edge Connecting: {step1} --> {step2}. Probability adjusted.")
                    missing_count += 1
                total_steps += 1

            if missing_count > 0 and total_steps > 0:
                # Geometric mean approach
                gmean = np.power(proba, 1.0 / total_steps)
                proba *= (gmean ** missing_count)

            return proba

    def walk_log_probability(
        self,
        walk: Union[str, List[str]],
        verbose: bool = True
    ) -> float:
        """
        Convenience method to compute log-probability of a walk.
        Equivalent to walk_probability(walk, use_log=True).

        Recommended for long sequences (>30 amino acids) to prevent
        numerical underflow.

        Args:
            walk: The walk as a string or list of sub-patterns.
            verbose: Whether to log missing-edge warnings.

        Returns:
            float: Log-probability of generating the walk.
        """
        return self.walk_probability(walk, verbose=verbose, use_log=True)

    def walk_gene_probability(
        self,
        walk: Union[str, List[str]],
        v: str,
        j: str,
        verbose: bool = True,
        use_epsilon: bool = False
    ) -> Tuple[float, float]:
        """
        Compute the probability of generating a walk under a specific (V, J) gene pair.
        We start with the marginal probabilities for v and j, then multiply by
        edge-level usage.

        Returns:
            (proba_v, proba_j) as a tuple of floats.
            If an edge is missing, we either return 0 or an epsilon if use_epsilon=True.
        """
        # Possibly re-encode the walk if the user passed a raw string
        if isinstance(walk, str):
            lz, locs = derive_lz_and_position(walk)
            walk_ = [f"{subp}_{pos}" for subp, pos in zip(lz, locs)]
        else:
            walk_ = walk

        try:
            proba_v = self.marginal_vgenes.loc[v]
            proba_j = self.marginal_jgenes.loc[j]
        except KeyError:
            logger.warning(f"Gene {v} or {j} not found in the marginal distributions.")
            val = np.finfo(float).eps if use_epsilon else 0.0
            return (val, val)

        for step1, step2 in window(walk_, 2):
            if not self.graph.has_edge(step1, step2):
                if verbose:
                    logger.warning(f"No edge for {step1}->{step2}.")
                val = np.finfo(float).eps if use_epsilon else 0.0
                return (val, val)

            e_data = self.graph[step1][step2]
            # If these genes aren't on the edge, it's effectively 0
            if v not in e_data or j not in e_data:
                if verbose:
                    logger.warning(f"Edge {step1}->{step2} missing {v} or {j}.")
                val = np.finfo(float).eps if use_epsilon else 0.0
                return (val, val)

            proba_v *= e_data[v]
            proba_j *= e_data[j]

        return proba_v, proba_j

    # --------------------------------------------------------------------------
    # Random Walk, Multi-gene Walk, and Variation Methods
    # --------------------------------------------------------------------------

    def multi_gene_random_walk(
        self,
        N: int,
        seq_len: Union[int, str],
        initial_state: Optional[str] = None,
        vj_init: str = "marginal"
    ):
        """
        Generate N random walks, each constrained to use a randomly selected (V, J) pair.
        If seq_len is an integer, we aim for a terminal state that matches that length.
        If seq_len == 'unsupervised', we consider all terminal states.

        Args:
            N (int): Number of random walks to generate.
            seq_len (int or 'unsupervised'): Desired sequence length or 'unsupervised'.
            initial_state (str): Optional initial node.
            vj_init (str): 'marginal' or 'combined' for random gene selection.

        Returns:
            A list of tuples: [(walk, selected_v, selected_j), ...].
        """
        selected_v, selected_j = self._select_random_vj_genes(vj_init)

        if seq_len == "unsupervised":
            final_states = list(self.terminal_states.index)
        else:
            final_states = self._length_specific_terminal_state(seq_len)

        if self.genetic_walks_black_list is None:
            self.genetic_walks_black_list = {}

        # We'll keep track of how many times each final state can still be used
        lengths = pd.Series(self.terminal_states).value_counts()
        max_length = lengths.idxmax() if not lengths.empty else None

        results = []
        for _ in tqdm(range(N), desc="Generating multi-gene walks"):
            if initial_state is None:
                current_state = self._random_initial_state()
                walk = [current_state]
            else:
                current_state = initial_state
                walk = [initial_state]

            # while the walk is not in a valid final state
            while current_state not in lengths.index:
                # Extract data from the current state's edges
                if current_state not in self.graph:
                    logger.warning(f"Current state {current_state} not in graph.")
                    break

                edge_info = pd.DataFrame(dict(self.graph[current_state]))
                # Apply blacklist if present
                if (current_state, selected_v, selected_j) in self.genetic_walks_black_list:
                    blacklisted = self.genetic_walks_black_list[(current_state, selected_v, selected_j)]
                    edge_info = edge_info.drop(columns=blacklisted, errors="ignore")

                # Check for presence of selected V/J genes
                # We'll consider edges that contain both selected_v and selected_j
                # in the attribute keys
                sub_df = edge_info.T[[selected_v, selected_j]].dropna(how="any") if \
                    {selected_v, selected_j}.issubset(edge_info.index) else pd.DataFrame()

                if sub_df.empty:
                    # No valid edges
                    if len(walk) > 2:
                        prev_state = walk[-2]
                        self.genetic_walks_black_list[(prev_state, selected_v, selected_j)] = \
                            self.genetic_walks_black_list.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                        current_state = prev_state
                        walk.pop()
                    else:
                        walk = walk[:1]
                        current_state = walk[0]
                        selected_v, selected_j = self._select_random_vj_genes(vj_init)
                    continue

                # Weighted choice among these edges
                w = edge_info.loc["weight", sub_df.index]
                w /= w.sum()
                if w.empty:
                    # Again, no valid edges
                    if len(walk) > 2:
                        prev_state = walk[-2]
                        self.genetic_walks_black_list[(prev_state, selected_v, selected_j)] = \
                            self.genetic_walks_black_list.get((prev_state, selected_v, selected_j), []) + [walk[-1]]
                        current_state = prev_state
                        walk.pop()
                    else:
                        walk = walk[:1]
                        current_state = walk[0]
                        selected_v, selected_j = self._select_random_vj_genes(vj_init)
                    continue

                current_state = np.random.choice(w.index, p=w.values)
                walk.append(current_state)

            results.append((walk, selected_v, selected_j))

            # If the walk ended in a length we track, decrement
            if (walk[-1] in lengths.index) and (walk[-1] != max_length):
                lengths[walk[-1]] -= 1
                if lengths[walk[-1]] < 0:
                    lengths.pop(walk[-1])

        return results

    def unsupervised_random_walk(self):
        """
        Conduct a random walk from a randomly selected initial state
        to a final state, ignoring gene constraints. The walk stops when
        `is_stop_condition` is True.

        Returns:
            (walk, sequence):
                - walk: list of node names
                - sequence: cleaned amino-acid sequence of the walk
        """
        random_init = self._random_initial_state()
        current_state = random_init
        walk = [random_init]
        sequence = self.clean_node(random_init)

        while not self.is_stop_condition(current_state):
            current_state = self.random_step(current_state)
            walk.append(current_state)
            sequence += self.clean_node(current_state)

        return walk, sequence

    def walk_genes(
        self,
        walk: List[str],
        dropna: bool = True,
        raise_error: bool = True
    ) -> pd.DataFrame:
        """
        Given a walk (list of nodes), return a DataFrame of gene usage at each edge.

        Args:
            walk: The node path.
            dropna: If True, drop edges with no gene data.
            raise_error: If True and result is empty, raise an Exception.

        Returns:
            A DataFrame where rows = gene names (V*, J*) and columns = edges in walk.
        """
        trans_genes = {}
        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i+1]):
                edge_attrs = self.graph[walk[i]][walk[i+1]].copy()
                # Remove these special keys
                for remove_key in ["weight", "Vsum", "Jsum"]:
                    edge_attrs.pop(remove_key, None)
                trans_genes[f"{walk[i]}->{walk[i+1]}"] = edge_attrs

        df = pd.DataFrame(trans_genes)
        if dropna:
            df.dropna(how="all", inplace=True)

        if df.empty and raise_error:
            raise GeneAnnotationError("No gene data found in the edges for the given walk.")

        # Example: add gene type and sum columns for clarity
        df["type"] = ["V" if "v" in idx.lower() else "J" for idx in df.index]
        df["sum"] = df.sum(axis=1, numeric_only=True)

        return df

    def random_walk_distribution_based(self, length_distribution: pd.Series):
        """
        Creates random walks in proportion to a given length distribution.
        We do a large number of unsupervised walks, then sample from them
        to match the specified distribution.

        Args:
            length_distribution: A Series whose index is lengths and values are
                how many sequences of that length we want.

        Returns:
            A 2D array (list of pairs) of shape [N, 2], where each row is (Seq, Walk).
        """
        N = length_distribution.sum() * 3  # multiply by some factor
        N = int(N)

        walks = []
        seqs = []
        logger.info(f"Generating ~{N} random walks to filter by length distribution...")
        for _ in tqdm(range(N), desc="Random Walk Distribution"):
            rw, rseq = self.unsupervised_random_walk()
            walks.append(rw)
            seqs.append(rseq)

        df = pd.DataFrame({"Seqs": seqs, "Walks": walks})
        df["L"] = df["Seqs"].str.len()

        samples = []
        for length_val in length_distribution.index:
            needed = length_distribution[length_val]
            subset = df[df["L"] == length_val]
            if len(subset) < needed:
                logger.warning(
                    f"Requested {needed} sequences of length {length_val}, but only found {len(subset)}."
                )
                needed = len(subset)
            if needed > 0:
                samples.append(subset.sample(n=needed, replace=False))

        if not samples:
            return np.array([])

        final = pd.concat(samples, ignore_index=True)
        return final[["Seqs", "Walks"]].values

    def get_gene_graph(self, v: str, j: str) -> nx.DiGraph:
        """
        Returns a subgraph containing only edges that contain both gene v and j.
        """
        if self.edges_list is None:
            self.edges_list = list(self.graph.edges(data=True))

        to_drop = []
        for src, dst, attrs in self.edges_list:
            if (v not in attrs) or (j not in attrs):
                to_drop.append((src, dst))

        G = self.graph.copy()
        G.remove_edges_from(to_drop)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def cac_random_gene_walk(self, initial_state=None, vj_init="combined"):
        """
        Conduct a random walk in a "combine-and-conquer" style,
        using a subgraph that only contains edges with the selected V/J.

        If the subgraph for (V, J) doesn't exist yet, create it. Then pick a random
        initial state from that subgraph and walk until a final node is reached.
        """
        selected_v, selected_j = self._select_random_vj_genes(vj_init)

        if (selected_v, selected_j) not in self.cac_graphs:
            G = self.get_gene_graph(selected_v, selected_j)
            self.cac_graphs[(selected_v, selected_j)] = G
        else:
            G = self.cac_graphs[(selected_v, selected_j)]

        final_states = list(set(self.terminal_states.index) & set(G.nodes))
        first_states = self.initial_states.loc[list(set(self.initial_states.index) & set(G.nodes))]
        first_states = first_states / first_states.sum()

        if initial_state is None:
            current_state = np.random.choice(first_states.index, p=first_states.values)
        else:
            current_state = initial_state

        walk = [current_state]
        if self.genetic_walks_black_list is None:
            self.genetic_walks_black_list = {}

        while current_state not in final_states:
            edge_info = pd.DataFrame(dict(G[current_state]))
            # Apply blacklist
            if (selected_v, selected_j, current_state) in self.genetic_walks_black_list:
                edge_info = edge_info.drop(
                    columns=self.genetic_walks_black_list[(selected_v, selected_j, current_state)],
                    errors="ignore"
                )

            if edge_info.shape[1] == 0:
                if len(walk) > 1:
                    prev_state = walk[-2]
                    blacklisted_cols = self.genetic_walks_black_list.get((selected_v, selected_j, prev_state), [])
                    blacklisted_cols.append(current_state)
                    self.genetic_walks_black_list[(selected_v, selected_j, prev_state)] = blacklisted_cols
                    walk.pop()
                    current_state = walk[-1]
                else:
                    # Stuck at the start
                    break

            sub_df = edge_info.T[[selected_v, selected_j]].dropna(how="any") if \
                {selected_v, selected_j}.issubset(edge_info.index) else pd.DataFrame()
            if sub_df.empty:
                # No valid edges
                if len(walk) > 1:
                    prev_state = walk[-2]
                    blacklisted_cols = self.genetic_walks_black_list.get((selected_v, selected_j, prev_state), [])
                    blacklisted_cols.append(current_state)
                    self.genetic_walks_black_list[(selected_v, selected_j, prev_state)] = blacklisted_cols
                    walk.pop()
                    current_state = walk[-1]
                else:
                    break
            else:
                w = edge_info.loc["weight", sub_df.index]
                w /= w.sum()
                next_state = np.random.choice(w.index, p=w.values)
                walk.append(next_state)
                current_state = next_state

        return walk, selected_v, selected_j

    def sequence_variation_curve(self, cdr3_sample: str):
        """
        Given a CDR3 sequence, return two lists:
            (encoded_subpatterns, out_degree_list)
        where out_degree_list[i] is the out-degree of the node in the graph
        corresponding to the i-th subpattern.
        """
        encoded = self.encode_sequence(cdr3_sample)
        curve = [self.graph.out_degree(node) for node in encoded]
        return encoded, curve

    def path_gene_table(
        self,
        cdr3_sample: str,
        threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return two tables (for V genes and J genes) representing all possible
        V/J usage that could generate the given cdr3_sample. Genes missing from
        more than 'threshold' fraction of edges are dropped.

        Args:
            cdr3_sample: The amino acid sequence to examine.
            threshold: If None, defaults to length/4 for V genes,
                       and length/2 for J genes.

        Returns:
            (vgene_table, jgene_table) as DataFrames.
        """
        encoded = self.encode_sequence(cdr3_sample)
        length = len(encoded)

        if threshold is None:
            threshold_v = length * 0.25
            threshold_j = length * 0.5
        else:
            threshold_v = threshold
            threshold_j = threshold

        # Get gene table once (avoid duplicate expensive call)
        gene_table = self.walk_genes(encoded, dropna=False, raise_error=False)
        na_counts = gene_table.isna().sum(axis=1)

        # For V genes
        mask_v = na_counts < threshold_v
        vgene_table = gene_table[mask_v & gene_table.index.str.contains("V", case=False)]

        # For J genes
        mask_j = na_counts < threshold_j
        jgene_table = gene_table[mask_j & gene_table.index.str.contains("J", case=False)]

        # Sort by ascending number of NaNs (optional clarity)
        jgene_table = jgene_table.loc[jgene_table.isna().sum(axis=1).sort_values().index]
        vgene_table = vgene_table.loc[vgene_table.isna().sum(axis=1).sort_values().index]

        return vgene_table, jgene_table

    def gene_variation(self, cdr3: str) -> pd.DataFrame:
        """
        Return a DataFrame that shows how many V and J genes are possible
        for each subpattern in the given cdr3 sequence.

        The DataFrame columns:
            - 'genes': number of possible V or J genes
            - 'type': 'V' or 'J'
            - 'sp': the LZ subpattern
        """
        if not self.genetic:
            raise NoGeneDataError(
                operation="gene_repertoire_per_subpattern",
                message="Cannot compute gene repertoire: this LZGraph has no gene data (genetic=False)."
            )

        encoded_a = self.encode_sequence(cdr3)
        n_v_genes = []
        n_j_genes = []

        # First subpattern: full marginal V, J size
        n_v_genes.append(len(self.marginal_vgenes))
        n_j_genes.append(len(self.marginal_jgenes))

        for node in encoded_a[1:]:
            in_edges = self.graph.in_edges(node)
            v_genes = set()
            j_genes = set()
            for e_a, e_b in in_edges:
                # Gather keys ignoring weight, Vsum, Jsum
                ed = pd.Series(self.graph[e_a][e_b]).drop(["weight", "Vsum", "Jsum"], errors="ignore")
                # Gene names are like "TRBV30-1*01" and "TRBJ1-2*01", so use 'in' not startswith
                v_genes |= set(g for g in ed.index if "V" in g)
                j_genes |= set(g for g in ed.index if "J" in g)

            n_v_genes.append(len(v_genes))
            n_j_genes.append(len(j_genes))

        # Combine into a DataFrame
        lz_subpatterns = lempel_ziv_decomposition(cdr3)
        j_df = pd.DataFrame({
            "genes": n_v_genes + n_j_genes,
            "type": (["V"] * len(n_v_genes)) + (["J"] * len(n_j_genes)),
            "sp": lz_subpatterns + lz_subpatterns
        })
        return j_df
