"""
Tests for NDPLZGraph Class
==========================

Tests covering the NDPLZGraph (Nucleotide-position Dependent) implementation
which incorporates positional information into the LZ decomposition for
nucleotide sequences.

Test Categories:
- Graph construction and structure
- Node and edge properties
- Positional encoding
- Probability calculations
- Gene-aware random walks
- Terminal states and length distributions
"""

import pytest
import numpy as np
from LZGraphs import NDPLZGraph


class TestNDPLZGraphConstruction:
    """Tests for NDPLZGraph construction and basic properties."""

    def test_graph_has_expected_edge_count(self, ndp_lzgraph):
        """Verify the graph has the expected number of edges."""
        assert len(ndp_lzgraph.edges) == 20587

    def test_initial_states_populated(self, ndp_lzgraph):
        """Verify initial states are correctly counted with position."""
        # T at position 1 should be the most common initial state
        assert ndp_lzgraph.initial_states['T0_1'] == 4991

    def test_node_frequency_tracking(self, ndp_lzgraph):
        """Verify per-node frequency is correctly tracked."""
        assert ndp_lzgraph.per_node_observed_frequency['AC0_14'] == 69


class TestNDPLZGraphProbabilities:
    """Tests for probability calculations in NDPLZGraph."""

    def test_subpattern_probability_rare_pattern(self, ndp_lzgraph):
        """Verify probability for a rare positional pattern."""
        expected = 0.0005410062716652975
        actual = ndp_lzgraph.subpattern_individual_probability.loc['A0_19', 'proba']
        assert actual == expected

    def test_subpattern_probability_common_pattern(self, ndp_lzgraph):
        """Verify probability for a common positional pattern."""
        expected = 0.04535435910794077
        actual = ndp_lzgraph.subpattern_individual_probability.loc['CA2_7', 'proba']
        assert actual == expected

    def test_walk_probability_calculation(self, ndp_lzgraph, test_data_ndp):
        """Verify walk probability calculation produces expected results."""
        lzpgens = []
        for sequence in test_data_ndp['cdr3_rearrangement'].iloc[:15]:
            walk = NDPLZGraph.encode_sequence(sequence)
            lzpgen = ndp_lzgraph.walk_probability(walk, verbose=False)
            lzpgens.append(lzpgen)

        # Verify first sequence's log-probability
        assert np.round(np.log(lzpgens[0]), 2) == -41.71


class TestNDPLZGraphTerminalStates:
    """Tests for terminal state mapping in NDPLZGraph."""

    def test_terminal_state_map_mid_sequence(self, ndp_lzgraph):
        """Verify terminal state mapping for mid-sequence position."""
        expected_terminals = [
            'C2_54', 'T2_57', 'TTT0_60', 'TTC0_54', 'TC1_51', 'C2_57',
            'T2_54', 'T2_60', 'TT1_57', 'TT1_60', 'TTT0_57', 'TC1_60',
            'C2_60', 'TTT0_63', 'TTC0_57', 'TTC0_60', 'T2_63'
        ]
        actual = ndp_lzgraph.terminal_state_map['TC1_51']
        assert set(actual) == set(expected_terminals)

    def test_terminal_state_map_late_sequence(self, ndp_lzgraph):
        """Verify terminal state mapping for late sequence position."""
        expected_terminals = ['TC1_60', 'TTT0_63', 'T2_63']
        actual = ndp_lzgraph.terminal_state_map['TC1_60']
        assert set(actual) == set(expected_terminals)

    def test_terminal_state_map_short_list(self, ndp_lzgraph):
        """Verify terminal state mapping with few terminal options."""
        expected_terminals = ['TT1_45', 'TTT0_51', 'TTT0_48']
        actual = ndp_lzgraph.terminal_state_map['TT1_45']
        assert set(actual) == set(expected_terminals)

    def test_terminal_states_count(self, ndp_lzgraph):
        """Verify terminal state counts."""
        assert ndp_lzgraph.terminal_states['C2_42'] == 316


class TestNDPLZGraphRandomWalks:
    """Tests for random walk generation in NDPLZGraph."""

    def test_unsupervised_random_walk_returns_result(self, ndp_lzgraph):
        """Verify unsupervised random walk generates a sequence."""
        result = ndp_lzgraph.unsupervised_random_walk()
        assert result is not None

    def test_gene_random_walk_unsupervised(self, ndp_lzgraph):
        """Verify gene-aware random walk with unsupervised length."""
        result = ndp_lzgraph.gene_random_walk(seq_len='unsupervised')
        assert result is not None

    def test_random_walk_with_initial_state(self, ndp_lzgraph):
        """Verify random walk starts with specified initial state."""
        # RandomWalkMixin.random_walk returns just the walk list
        walk = ndp_lzgraph.random_walk(initial_state='T0_1')
        assert walk[0] == 'T0_1'


class TestNDPLZGraphFeatures:
    """Tests for feature extraction from NDPLZGraph."""

    def test_eigenvector_centrality(self, ndp_lzgraph):
        """Verify eigenvector centrality calculation."""
        feature_vector = ndp_lzgraph.eigenvector_centrality()
        assert np.round(np.log(feature_vector['AT0_26']), 5) == -38.79923


class TestNDPLZGraphGeneData:
    """Tests for V/J gene data handling in NDPLZGraph."""

    def test_marginal_vgene_probability(self, ndp_lzgraph):
        """Verify marginal V gene probability."""
        assert ndp_lzgraph.marginal_vgenes['TRBV2-1*01'] == 0.0502

    def test_length_distribution_common(self, ndp_lzgraph):
        """Verify length distribution for common length."""
        assert ndp_lzgraph.lengths[42] == 1143

    def test_length_distribution_rare(self, ndp_lzgraph):
        """Verify length distribution for rare length."""
        assert ndp_lzgraph.lengths[27] == 1


class TestNDPLZGraphEncoding:
    """Tests for sequence encoding in NDPLZGraph."""

    def test_encode_sequence_includes_position(self):
        """Verify encoded sequence includes positional information."""
        sequence = "TGTGCCAGCAGCCAAGA"
        walk = NDPLZGraph.encode_sequence(sequence)

        # All nodes should contain underscore and position number
        for node in walk:
            assert '_' in node, f"Node {node} missing position separator"
            parts = node.rsplit('_', 1)
            assert parts[1].isdigit(), f"Node {node} missing valid position"

    def test_encode_sequence_positions_monotonic(self):
        """Verify positions in encoded sequence are monotonically increasing."""
        sequence = "TGTGCCAGCAGCCAAGA"
        walk = NDPLZGraph.encode_sequence(sequence)

        positions = [int(node.rsplit('_', 1)[1]) for node in walk]
        for i in range(1, len(positions)):
            assert positions[i] > positions[i - 1], \
                f"Position {positions[i]} not greater than {positions[i-1]}"
