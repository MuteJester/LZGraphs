"""
Tests for NaiveLZGraph Class
============================

Tests covering the NaiveLZGraph implementation which uses
position-independent LZ decomposition for nucleotide sequences.

Test Categories:
- Graph construction and structure
- Node and edge properties
- Probability calculations
- Random walk generation
- Feature extraction (eigenvector centrality)
"""

import pytest
import numpy as np
from LZGraphs import NaiveLZGraph


class TestNaiveLZGraphConstruction:
    """Tests for NaiveLZGraph construction and basic properties."""

    def test_graph_has_expected_edge_count(self, naive_lzgraph):
        """Verify the graph has the expected number of edges."""
        assert len(naive_lzgraph.edges) == 5137

    def test_initial_states_populated(self, naive_lzgraph):
        """Verify initial states are correctly counted."""
        assert naive_lzgraph.initial_states['T'] == 4994

    def test_node_frequency_tracking(self, naive_lzgraph):
        """Verify per-node frequency is correctly tracked."""
        assert naive_lzgraph.per_node_observed_frequency['AGT'] == 406


class TestNaiveLZGraphProbabilities:
    """Tests for probability calculations in NaiveLZGraph."""

    def test_subpattern_individual_probability_single_char(self, naive_lzgraph):
        """Verify single-character subpattern probability."""
        expected = 0.050210381498478625
        actual = naive_lzgraph.subpattern_individual_probability.loc['C', 'proba']
        assert actual == expected

    def test_subpattern_individual_probability_two_char(self, naive_lzgraph):
        """Verify two-character subpattern probability."""
        expected = 0.048503228527530355
        actual = naive_lzgraph.subpattern_individual_probability.loc['CA', 'proba']
        assert actual == expected

    def test_walk_probability_calculation(
        self, naive_lzgraph, test_data_nucleotide
    ):
        """Verify walk probability calculation produces expected results."""
        lzpgens = []
        for sequence in test_data_nucleotide['cdr3_rearrangement'].iloc[:15]:
            walk = NaiveLZGraph.encode_sequence(sequence)
            lzpgen = naive_lzgraph.walk_probability(walk, verbose=False)
            lzpgens.append(lzpgen)

        # Verify first sequence's log-probability
        assert np.round(np.log(lzpgens[0]), 2) == -84.92


class TestNaiveLZGraphTerminalStates:
    """Tests for terminal state mapping in NaiveLZGraph."""

    def test_terminal_state_map_single_char(self, naive_lzgraph):
        """Verify terminal state mapping for single character."""
        expected_terminals = [
            'C', 'TTT', 'T', 'TT', 'TTC', 'TC', 'CTTT', 'CTTC', 'TTTT'
        ]
        actual = naive_lzgraph.terminal_state_map['C']
        assert set(actual) == set(expected_terminals)

    def test_terminal_state_map_four_char(self, naive_lzgraph):
        """Verify terminal state mapping for four-character pattern."""
        expected_terminals = [
            'C', 'TTT', 'T', 'TT', 'TTC', 'TC', 'CTTT', 'CTTC', 'TTTT'
        ]
        actual = naive_lzgraph.terminal_state_map['TTTT']
        assert set(actual) == set(expected_terminals)


class TestNaiveLZGraphRandomWalks:
    """Tests for random walk generation in NaiveLZGraph."""

    def test_unsupervised_random_walk_returns_result(self, naive_lzgraph):
        """Verify unsupervised random walk generates a sequence."""
        result = naive_lzgraph.unsupervised_random_walk()
        assert result is not None

    def test_fixed_length_random_walk(self, naive_lzgraph):
        """Verify fixed-length random walk generates correct length."""
        walk, _ = naive_lzgraph.random_walk(25)
        assert len(walk) == 25


class TestNaiveLZGraphFeatures:
    """Tests for feature extraction from NaiveLZGraph."""

    def test_eigenvector_centrality(self, naive_lzgraph):
        """Verify eigenvector centrality calculation."""
        feature_vector = naive_lzgraph.eigenvector_centrality()
        assert np.round(feature_vector['AA'], 5) == 0.12581


class TestNaiveLZGraphEncoding:
    """Tests for sequence encoding in NaiveLZGraph."""

    def test_encode_sequence_returns_list(self):
        """Verify encode_sequence returns a list of subpatterns."""
        sequence = "TGTGCCAGCAGCCAAGA"
        walk = NaiveLZGraph.encode_sequence(sequence)
        assert isinstance(walk, list)
        assert len(walk) > 0

    def test_encode_sequence_covers_full_sequence(self):
        """Verify encoded walk covers the entire input sequence."""
        sequence = "TGTGCCAGCAGCCAAGA"
        walk = NaiveLZGraph.encode_sequence(sequence)
        reconstructed = ''.join(walk)
        assert reconstructed == sequence
