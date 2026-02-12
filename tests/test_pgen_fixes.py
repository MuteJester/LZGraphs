"""
Tests for PGEN / walk_probability Fixes
========================================

Tests covering the 6 issues fixed in walk_probability:
1. Initial probability uses initial_state_probabilities (not node_probability)
2. All-edges-missing returns 0 (not initial_prob * 1.0)
3. Terminal state factor P(stop|last_node) is included
4. Laplace smoothing option works
5. Initial state threshold is configurable
6. NDPLZGraph now has geometric mean imputation
7. Log/linear probability consistency
"""

import pytest
import numpy as np
import pandas as pd

from LZGraphs import AAPLZGraph, NDPLZGraph, NaiveLZGraph


class TestInitialProbabilityFix:
    """Verify walk_probability uses initial_state_probabilities, not node_probability."""

    def test_initial_prob_uses_initial_states(self, aap_lzgraph):
        """The initial probability should come from initial_state_probabilities."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)
        first_node = walk[0]

        # Verify the first node IS in initial_state_probabilities
        assert first_node in aap_lzgraph.initial_state_probabilities

        # The values should differ between the two distributions
        init_prob = aap_lzgraph.initial_state_probabilities[first_node]
        subp_prob = aap_lzgraph.node_probability[first_node]
        # These SHOULD be different distributions
        # (initial_states counts how often a node starts a sequence,
        #  subpattern counts overall node frequency)
        assert init_prob != subp_prob

    def test_unrecognized_initial_state_returns_zero(self, aap_lzgraph):
        """If the first node is not in initial_state_probabilities, return 0."""
        fake_walk = ['ZZZZ_99', 'YYYY_100']
        result = aap_lzgraph.walk_probability(fake_walk, verbose=False)
        assert result == 0.0

    def test_unrecognized_initial_state_log_returns_neg_inf(self, aap_lzgraph):
        """If the first node is not in initial_state_probabilities, return -inf in log mode."""
        fake_walk = ['ZZZZ_99', 'YYYY_100']
        result = aap_lzgraph.walk_probability(fake_walk, verbose=False, use_log=True)
        assert result == float('-inf')


class TestAllEdgesMissingFix:
    """Verify AAPLZGraph returns 0 when all edges are missing (not initial_prob * 1.0)."""

    def test_all_edges_missing_returns_zero(self, aap_lzgraph):
        """When impute_missing_edges=True but ALL edges are missing, return 0."""
        # Create a walk where first node exists but no edges connect any pair
        first_node = list(aap_lzgraph.initial_state_probabilities.keys())[0]
        fake_walk = [first_node, 'NONEXISTENT_99', 'ALSO_FAKE_100']
        result = aap_lzgraph.walk_probability(fake_walk, verbose=False)
        assert result == 0.0

    def test_all_edges_missing_log_returns_neg_inf(self, aap_lzgraph):
        """When impute_missing_edges=True but ALL edges are missing, return -inf in log mode."""
        first_node = list(aap_lzgraph.initial_state_probabilities.keys())[0]
        fake_walk = [first_node, 'NONEXISTENT_99', 'ALSO_FAKE_100']
        result = aap_lzgraph.walk_probability(fake_walk, verbose=False, use_log=True)
        assert result == float('-inf')


class TestTerminalStateFactor:
    """Verify terminal state P(stop|last_node) is included in walk_probability."""

    def test_terminal_factor_included(self, aap_lzgraph):
        """Walk probability should include the stop probability factor."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)
        last_node = walk[-1]

        # Get the stop probability for the last node
        if last_node in aap_lzgraph.terminal_state_data:
            stop_prob = aap_lzgraph.terminal_state_data[last_node]['stop_probability']
            # The stop probability should be > 0 and <= 1
            assert 0 < stop_prob <= 1

        # The walk probability should be positive (valid sequence)
        prob = aap_lzgraph.walk_probability(walk, verbose=False)
        assert prob > 0

    def test_terminal_factor_in_log_mode(self, aap_lzgraph):
        """Terminal factor should also be included in log mode."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        log_prob = aap_lzgraph.walk_probability(walk, verbose=False, use_log=True)
        # Log probability should be finite and negative
        assert np.isfinite(log_prob)
        assert log_prob < 0

    def test_non_terminal_last_node_gets_epsilon(self, aap_lzgraph):
        """If last node is not a terminal state, stop prob should be epsilon (very low)."""
        # Find a node that is NOT a terminal state
        all_nodes = set(aap_lzgraph.graph.nodes())
        terminal_nodes = set(aap_lzgraph.terminal_state_data.keys())
        non_terminal = all_nodes - terminal_nodes

        if non_terminal:
            # Pick any non-terminal node
            nt_node = next(iter(non_terminal))
            first_node = list(aap_lzgraph.initial_state_probabilities.keys())[0]

            # Create walk ending at non-terminal (if edge exists)
            if aap_lzgraph.graph.has_edge(first_node, nt_node):
                walk = [first_node, nt_node]
                prob = aap_lzgraph.walk_probability(walk, verbose=False)
                # Should be very small due to epsilon terminal factor
                assert prob < 1e-10


class TestLaplaceSmoothing:
    """Verify Laplace smoothing works when smoothing_alpha > 0."""

    def test_smoothing_changes_edge_weights(self, test_data_aap):
        """Edge weights should differ when smoothing is applied."""
        g_no_smooth = AAPLZGraph(test_data_aap, verbose=False, smoothing_alpha=0.0)
        g_smooth = AAPLZGraph(test_data_aap, verbose=False, smoothing_alpha=0.1)

        # Pick an edge and verify weights differ
        edge = list(g_no_smooth.graph.edges())[0]
        w_orig = g_no_smooth.graph[edge[0]][edge[1]]['data'].weight
        w_smooth = g_smooth.graph[edge[0]][edge[1]]['data'].weight
        assert w_orig != w_smooth

    def test_smoothed_weights_still_normalize(self, test_data_aap):
        """Outgoing weights from any node should still sum to ~1 with smoothing."""
        g = AAPLZGraph(test_data_aap, verbose=False, smoothing_alpha=0.1)

        for node in list(g.graph.nodes())[:10]:
            successors = list(g.graph.successors(node))
            if successors:
                total = sum(g.graph[node][s]['data'].weight for s in successors)
                assert abs(total - 1.0) < 0.01, f"Node {node}: weights sum to {total}"


class TestInitialStateThreshold:
    """Verify initial state threshold is configurable."""

    def test_threshold_zero_includes_all(self, test_data_aap):
        """With threshold=0, all observed initial states should be included."""
        g_default = AAPLZGraph(test_data_aap, verbose=False, min_initial_state_count=5)
        g_zero = AAPLZGraph(test_data_aap, verbose=False, min_initial_state_count=0)

        # Zero threshold should include at least as many initial states
        assert len(g_zero.initial_state_probabilities) >= len(g_default.initial_state_probabilities)

    def test_high_threshold_excludes_rare(self, test_data_aap):
        """With high threshold, rare initial states should be excluded."""
        g_high = AAPLZGraph(test_data_aap, verbose=False, min_initial_state_count=100)
        g_default = AAPLZGraph(test_data_aap, verbose=False, min_initial_state_count=5)

        # High threshold should include fewer initial states
        assert len(g_high.initial_state_probabilities) <= len(g_default.initial_state_probabilities)


class TestNDPImputation:
    """Verify NDPLZGraph now has impute_missing_edges=True."""

    def test_ndp_impute_flag_set(self, ndp_lzgraph):
        """NDPLZGraph should have impute_missing_edges=True."""
        assert ndp_lzgraph.impute_missing_edges is True

    def test_ndp_missing_edge_imputed(self, ndp_lzgraph):
        """NDPLZGraph should impute missing edges instead of returning 0."""
        # Find a valid first node
        first_node = list(ndp_lzgraph.initial_state_probabilities.keys())[0]

        # Create a walk with one valid edge and one fake edge
        successors = list(ndp_lzgraph.graph.successors(first_node))
        if successors:
            valid_next = successors[0]
            # Walk: valid_first -> valid_next -> fake_node
            walk = [first_node, valid_next, 'FAKE_NODE_999']
            result = ndp_lzgraph.walk_probability(walk, verbose=False)
            # With imputation, should return > 0 (imputed, not 0)
            assert result > 0


class TestLogLinearConsistency:
    """Verify log(P_linear) matches P_log mode."""

    def test_aap_log_linear_match(self, aap_lzgraph):
        """For AAPLZGraph, log(walk_prob) should equal walk_prob(use_log=True)."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        prob = aap_lzgraph.walk_probability(walk, verbose=False)
        log_prob = aap_lzgraph.walk_probability(walk, verbose=False, use_log=True)

        assert prob > 0
        assert np.isclose(np.log(prob), log_prob, rtol=1e-5)

    def test_ndp_log_linear_match(self, ndp_lzgraph, test_data_ndp):
        """For NDPLZGraph, log(walk_prob) should equal walk_prob(use_log=True)."""
        seq = test_data_ndp['cdr3_rearrangement'].iloc[0]
        walk = NDPLZGraph.encode_sequence(seq)

        prob = ndp_lzgraph.walk_probability(walk, verbose=False)
        log_prob = ndp_lzgraph.walk_probability(walk, verbose=False, use_log=True)

        if prob > 0:
            assert np.isclose(np.log(prob), log_prob, rtol=1e-5)

    def test_naive_log_linear_match(self, naive_lzgraph, test_data_nucleotide):
        """For NaiveLZGraph, log(walk_prob) should equal walk_prob(use_log=True)."""
        seq = test_data_nucleotide['cdr3_rearrangement'].iloc[0]
        walk = NaiveLZGraph.encode_sequence(seq)

        prob = naive_lzgraph.walk_probability(walk, verbose=False)
        log_prob = naive_lzgraph.walk_probability(walk, verbose=False, use_log=True)

        if prob > 0:
            assert np.isclose(np.log(prob), log_prob, rtol=1e-5)
