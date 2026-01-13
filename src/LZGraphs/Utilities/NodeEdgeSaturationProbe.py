import random
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# NumPy 2.0 compatibility: trapz was renamed to trapezoid
try:
    from numpy import trapezoid as np_trapz
except ImportError:
    from numpy import trapz as np_trapz

from ..Graphs.AminoAcidPositional import derive_lz_and_position
from ..Graphs.NucleotideDoublePositional import derive_lz_reading_frame_position
from ..Utilities.decomposition import lempel_ziv_decomposition
from ..Utilities.misc import window


class NodeEdgeSaturationProbe:
    """
    A utility class to measure graph saturation without full LZGraph construction.

    This class emulates the creation process of an LZGraph by accumulating
    node and edge counts as sequences are processed. It's useful for:
    - Measuring repertoire diversity (K-diversity indices)
    - Analyzing saturation curves
    - Understanding pattern discovery rates

    Args:
        node_function (str or callable): The node extraction method to use:
            - 'naive': Emulate Naive LZGraph extraction
            - 'ndp': Emulate Nucleotide Double Positional LZGraph
            - 'aap': Amino Acid Positional LZGraph
            - callable: Custom function that takes a sequence and returns list of nodes
        log_level (int): Logging verbosity (1 = log node/edge counts, 0 = no logging)
        verbose (bool): Whether to show progress bars

    Attributes:
        nodes (set): Set of unique nodes discovered
        edges (set): Set of unique edges discovered
        log_memory (dict): Dictionary containing results of a single test run

    Example:
        >>> probe = NodeEdgeSaturationProbe(node_function='aap')
        >>> curve = probe.saturation_curve(sequences, log_every=100)
        >>> print(f"Final nodes: {curve['nodes'].iloc[-1]}")
    """

    def __init__(self, node_function: Union[str, Callable] = 'naive',
                 log_level: int = 1, verbose: bool = False):
        self.nodes = set()
        self.edges = set()
        self.log_memory = dict()
        self.verbose = verbose
        self.log_level = log_level
        self.node_function = None

        if node_function == 'naive':
            self.node_function = self.naive_node_extractor
        elif node_function == 'ndp':
            self.node_function = self.ndp_node_extractor
        elif node_function == 'aap':
            self.node_function = self.aap_node_extractor
        else:
            self.node_function = node_function

    def log(self, n_sequences: int) -> None:
        """Log current node and edge counts."""
        if self.log_level == 1:
            self.log_memory[n_sequences] = {
                'nodes': len(self.nodes),
                'edges': len(self.edges)
            }

    @staticmethod
    def naive_node_extractor(seq: str) -> List[str]:
        """
        Node extraction procedure used by the Naive LZGraph.

        Args:
            seq (str): A sequence of nucleotides or amino acids

        Returns:
            list: A list of nodes extracted from the given sequence
        """
        return lempel_ziv_decomposition(seq)

    @staticmethod
    def ndp_node_extractor(seq: str) -> List[str]:
        """
        Node extraction procedure used by the Nucleotide Double Positional LZGraph.

        Args:
            seq (str): A sequence of nucleotides

        Returns:
            list: A list of nodes with reading frame and position information
        """
        LZ, POS, locations = derive_lz_reading_frame_position(seq)
        nodes_local = list(map(lambda x, y, z: x + str(y) + '_' + str(z), LZ, POS, locations))
        return nodes_local

    @staticmethod
    def aap_node_extractor(seq: str) -> List[str]:
        """
        Node extraction procedure used by the Amino Acid Positional LZGraph.

        Args:
            seq (str): A sequence of amino acids

        Returns:
            list: A list of nodes with position information
        """
        LZ, locations = derive_lz_and_position(seq)
        nodes_local = list(map(lambda x, y: x + '_' + str(y), LZ, locations))
        return nodes_local

    def test_sequences(self, sequence_list: List[str], log_every: int = 1000,
                       iteration_number: Optional[int] = None) -> None:
        """
        Process sequences and log node/edge counts periodically.

        Args:
            sequence_list: A list of nucleotide or amino acid sequences
            log_every: After how many sequences to log counts
            iteration_number: Optional iteration identifier (unused but kept for compatibility)
        """
        slen = len(sequence_list)

        if self.verbose:
            itr = tqdm(enumerate(sequence_list, start=1), leave=False, position=0, total=slen)
        else:
            itr = enumerate(sequence_list, start=1)

        for ax, seq in itr:
            nodes_local = self.node_function(seq)
            self.nodes.update(nodes_local)
            self.edges.update(window(nodes_local, 2))

            if ax % log_every == 0 or ax >= slen:
                self.log(ax)

    def _reset(self) -> None:
        """Reset internal state for a new test."""
        self.nodes = set()
        self.edges = set()
        self.log_memory = dict()

    def resampling_test(self, sequence_list: List[str], n_tests: int,
                        log_every: int = 1000, sample_size: int = 0) -> List[dict]:
        """
        Run multiple saturation tests with random resampling.

        Args:
            sequence_list: A list of nucleotide or amino acid sequences
            n_tests: The number of resampling iterations
            log_every: After how many sequences to log counts
            sample_size: Number of sequences to sample per test (0 = use all)

        Returns:
            list: A list of log dictionaries for each iteration
        """
        result = []
        sequence_list = list(sequence_list)  # Ensure it's a list for shuffling

        if sample_size == 0:
            for n in range(n_tests):
                np.random.shuffle(sequence_list)
                self.test_sequences(sequence_list, log_every, n)
                result.append(self.log_memory.copy())
                self._reset()
        else:
            for n in range(n_tests):
                np.random.shuffle(sequence_list)
                sample = random.sample(sequence_list, min(sample_size, len(sequence_list)))
                self.test_sequences(sample, log_every, n)
                result.append(self.log_memory.copy())
                self._reset()

        return result

    def saturation_curve(self, sequence_list: List[str],
                         log_every: int = 100) -> pd.DataFrame:
        """
        Compute the full saturation curve for a sequence list.

        The saturation curve shows how the number of unique nodes and edges
        grows as more sequences are processed. This is useful for understanding
        the diversity and complexity of a repertoire.

        Args:
            sequence_list: A list of nucleotide or amino acid sequences
            log_every: After how many sequences to log counts (smaller = more resolution)

        Returns:
            pd.DataFrame: DataFrame with columns ['n_sequences', 'nodes', 'edges']

        Example:
            >>> probe = NodeEdgeSaturationProbe(node_function='aap')
            >>> curve = probe.saturation_curve(sequences, log_every=50)
            >>> curve.plot(x='n_sequences', y=['nodes', 'edges'])
        """
        self._reset()
        self.test_sequences(sequence_list, log_every=log_every)

        # Convert log_memory to DataFrame
        data = []
        for n_seq, counts in sorted(self.log_memory.items()):
            data.append({
                'n_sequences': n_seq,
                'nodes': counts['nodes'],
                'edges': counts['edges']
            })

        self._reset()
        return pd.DataFrame(data)

    def half_saturation_point(self, sequence_list: List[str],
                              log_every: int = 50,
                              metric: str = 'nodes') -> int:
        """
        Find K50: the sample size at which 50% of final nodes/edges are reached.

        This metric indicates how quickly the repertoire's pattern space
        is discovered. Lower values suggest more redundancy (patterns are
        discovered quickly), while higher values suggest more diversity.

        Args:
            sequence_list: A list of nucleotide or amino acid sequences
            log_every: Logging frequency (smaller = more precision)
            metric: 'nodes' or 'edges'

        Returns:
            int: The number of sequences needed to reach 50% of final count

        Example:
            >>> k50 = probe.half_saturation_point(sequences)
            >>> print(f"50% of patterns found after {k50} sequences")
        """
        curve = self.saturation_curve(sequence_list, log_every=log_every)

        if curve.empty:
            return 0

        final_count = curve[metric].iloc[-1]
        half_target = final_count * 0.5

        # Find first point where we reach 50%
        above_half = curve[curve[metric] >= half_target]
        if above_half.empty:
            return curve['n_sequences'].iloc[-1]

        return int(above_half['n_sequences'].iloc[0])

    def saturation_rate(self, sequence_list: List[str],
                        log_every: int = 100,
                        metric: str = 'nodes') -> pd.DataFrame:
        """
        Compute the rate of node/edge discovery along the saturation curve.

        The rate (slope) indicates how many new patterns are discovered per
        sequence. High rates early on suggest rapid pattern discovery, while
        declining rates indicate saturation.

        Args:
            sequence_list: A list of sequences
            log_every: Logging frequency
            metric: 'nodes' or 'edges'

        Returns:
            pd.DataFrame: DataFrame with columns ['n_sequences', 'rate']
                where rate is the derivative of the saturation curve

        Example:
            >>> rates = probe.saturation_rate(sequences)
            >>> # Rate should decrease as repertoire saturates
            >>> print(f"Initial rate: {rates['rate'].iloc[0]:.2f}")
            >>> print(f"Final rate: {rates['rate'].iloc[-1]:.2f}")
        """
        curve = self.saturation_curve(sequence_list, log_every=log_every)

        if len(curve) < 2:
            return pd.DataFrame({'n_sequences': [], 'rate': []})

        # Compute numerical gradient
        rates = np.gradient(curve[metric].values, curve['n_sequences'].values)

        return pd.DataFrame({
            'n_sequences': curve['n_sequences'],
            'rate': rates
        })

    def area_under_saturation_curve(self, sequence_list: List[str],
                                    log_every: int = 100,
                                    normalize: bool = True,
                                    metric: str = 'nodes') -> float:
        """
        Compute the Area Under the Saturation Curve (AUSC).

        AUSC provides a more robust diversity measure than single-point metrics
        like K1000. It captures the entire pattern discovery trajectory and is
        less sensitive to outliers.

        Interpretation:
        - Higher AUSC = patterns are discovered gradually (more diverse)
        - Lower AUSC = patterns are discovered quickly (less diverse)

        Args:
            sequence_list: A list of sequences
            log_every: Logging frequency
            normalize: If True, normalize by maximum possible area (0-1 scale)
            metric: 'nodes' or 'edges'

        Returns:
            float: The area under the saturation curve. If normalized, value
                   is between 0 and 1, where 1 indicates maximum diversity
                   (linear growth throughout).

        Example:
            >>> ausc = probe.area_under_saturation_curve(sequences, normalize=True)
            >>> print(f"Normalized AUSC: {ausc:.3f}")
        """
        curve = self.saturation_curve(sequence_list, log_every=log_every)

        if len(curve) < 2:
            return 0.0

        # Compute area using trapezoidal rule
        ausc = float(np_trapz(curve[metric].values, curve['n_sequences'].values))

        if normalize:
            # Maximum possible area: final count * total sequences
            max_ausc = curve[metric].iloc[-1] * curve['n_sequences'].iloc[-1]
            if max_ausc > 0:
                return ausc / max_ausc
            return 0.0

        return ausc

    def diversity_profile(self, sequence_list: List[str],
                          log_every: int = 100) -> pd.DataFrame:
        """
        Compute a comprehensive diversity profile of the sequence repertoire.

        This method combines multiple saturation-based metrics into a single
        summary, useful for repertoire comparison.

        Args:
            sequence_list: A list of sequences
            log_every: Logging frequency for saturation curve

        Returns:
            pd.DataFrame: Single-row DataFrame with diversity metrics:
                - n_sequences: Total sequences processed
                - final_nodes: Total unique nodes
                - final_edges: Total unique edges
                - k50_nodes: Half-saturation point for nodes
                - k50_edges: Half-saturation point for edges
                - ausc_nodes: Normalized AUSC for nodes
                - ausc_edges: Normalized AUSC for edges

        Example:
            >>> profile = probe.diversity_profile(sequences)
            >>> print(profile.T)  # Print as column for readability
        """
        curve = self.saturation_curve(sequence_list, log_every=log_every)

        if curve.empty:
            return pd.DataFrame()

        # Recompute half-saturation using the already computed curve
        final_nodes = curve['nodes'].iloc[-1]
        final_edges = curve['edges'].iloc[-1]

        # K50 for nodes
        half_nodes = curve[curve['nodes'] >= final_nodes * 0.5]
        k50_nodes = int(half_nodes['n_sequences'].iloc[0]) if not half_nodes.empty else curve['n_sequences'].iloc[-1]

        # K50 for edges
        half_edges = curve[curve['edges'] >= final_edges * 0.5]
        k50_edges = int(half_edges['n_sequences'].iloc[0]) if not half_edges.empty else curve['n_sequences'].iloc[-1]

        # AUSC normalized
        n_seq = curve['n_sequences'].values
        ausc_nodes = float(np_trapz(curve['nodes'].values, n_seq)) / (final_nodes * n_seq[-1]) if final_nodes > 0 else 0
        ausc_edges = float(np_trapz(curve['edges'].values, n_seq)) / (final_edges * n_seq[-1]) if final_edges > 0 else 0

        return pd.DataFrame([{
            'n_sequences': len(sequence_list),
            'final_nodes': int(final_nodes),
            'final_edges': int(final_edges),
            'k50_nodes': k50_nodes,
            'k50_edges': k50_edges,
            'ausc_nodes': ausc_nodes,
            'ausc_edges': ausc_edges
        }])


def get_k1000_diversity(list_of_sequences: List[str],
                        lzgraph_encoding_function: Callable,
                        draws: int = 25) -> float:
    """
    Legacy function for computing K1000 diversity.

    Deprecated: Use K1000_Diversity from LZGraphs.Metrics instead for
    better functionality including confidence intervals.

    Args:
        list_of_sequences: A list of CDR3 sequences
        lzgraph_encoding_function: The LZGraph encoding function
        draws: Number of resampling iterations

    Returns:
        float: Mean K1000 diversity value
    """
    import warnings
    warnings.warn(
        "get_k1000_diversity is deprecated. Use K1000_Diversity from "
        "LZGraphs.Metrics instead for confidence intervals and adaptive sampling.",
        DeprecationWarning,
        stacklevel=2
    )
    NESP = NodeEdgeSaturationProbe(node_function=lzgraph_encoding_function)
    result = NESP.resampling_test(list(set(list_of_sequences)), n_tests=draws, sample_size=1000)
    K_tests = [list(i.values())[-1]['nodes'] for i in result]
    return float(np.mean(K_tests))
