"""
Tests for simulate() and MLE stop probability
===============================================

Tests covering:
1. simulate() returns correct types
2. Deterministic output with seed
3. Generated sequence characteristics
4. Cache invalidation
5. Save/load cycle
6. All graph types
7. MLE stop probability correctness
8. clean_node() optimization equivalence
"""

import os
import tempfile
from copy import deepcopy

import numpy as np
import pytest

from LZGraphs import AAPLZGraph, NDPLZGraph, NaiveLZGraph


# =========================================================================
# simulate() tests
# =========================================================================


class TestSimulateBasic:
    """Basic simulate() functionality."""

    def test_returns_list_of_strings(self, aap_lzgraph):
        """simulate(n) returns a list of n strings."""
        results = aap_lzgraph.simulate(10, seed=42)
        assert isinstance(results, list)
        assert len(results) == 10
        assert all(isinstance(s, str) for s in results)
        assert all(len(s) > 0 for s in results)

    def test_deterministic_with_seed(self, aap_lzgraph):
        """Same seed produces same output."""
        results1 = aap_lzgraph.simulate(50, seed=123)
        results2 = aap_lzgraph.simulate(50, seed=123)
        assert results1 == results2

    def test_different_seeds_differ(self, aap_lzgraph):
        """Different seeds produce different output."""
        results1 = aap_lzgraph.simulate(50, seed=1)
        results2 = aap_lzgraph.simulate(50, seed=2)
        assert results1 != results2

    def test_return_walks(self, aap_lzgraph):
        """return_walks=True returns (walk, sequence) tuples."""
        results = aap_lzgraph.simulate(5, seed=42, return_walks=True)
        assert isinstance(results, list)
        assert len(results) == 5
        for walk, seq in results:
            assert isinstance(walk, list)
            assert isinstance(seq, str)
            assert len(walk) >= 1
            assert len(seq) > 0

    def test_walk_sequence_consistency(self, aap_lzgraph):
        """Walk and sequence should be consistent."""
        results = aap_lzgraph.simulate(10, seed=42, return_walks=True)
        for walk, seq in results:
            # Reconstruct sequence from walk
            reconstructed = ''.join(aap_lzgraph.clean_node(node) for node in walk)
            assert seq == reconstructed

    def test_simulate_zero(self, aap_lzgraph):
        """simulate(0) returns empty list."""
        results = aap_lzgraph.simulate(0, seed=42)
        assert results == []


class TestSimulateAllGraphTypes:
    """simulate() works on all graph types."""

    def test_aap_simulate(self, aap_lzgraph):
        results = aap_lzgraph.simulate(20, seed=42)
        assert len(results) == 20
        # AAP sequences should only contain amino acid letters
        aa_chars = set('ACDEFGHIKLMNPQRSTVWY')
        for seq in results:
            assert set(seq).issubset(aa_chars), f"Invalid chars in: {seq}"

    def test_ndp_simulate(self, ndp_lzgraph):
        results = ndp_lzgraph.simulate(20, seed=42)
        assert len(results) == 20
        # NDP sequences should only contain nucleotides
        nuc_chars = set('ATGC')
        for seq in results:
            assert set(seq).issubset(nuc_chars), f"Invalid chars in: {seq}"

    def test_naive_simulate(self, naive_lzgraph):
        results = naive_lzgraph.simulate(20, seed=42)
        assert len(results) == 20
        # Naive sequences should be nucleotide-like
        nuc_chars = set('ATGC')
        for seq in results:
            assert set(seq).issubset(nuc_chars), f"Invalid chars in: {seq}"


class TestSimulateLengthDistribution:
    """Generated sequences should have plausible length distributions."""

    def test_aap_length_range(self, aap_lzgraph):
        """AAP sequences should have CDR3-like lengths (roughly 5-30 AAs)."""
        results = aap_lzgraph.simulate(200, seed=42)
        lengths = [len(s) for s in results]
        assert min(lengths) >= 2
        assert max(lengths) <= 50
        mean_len = np.mean(lengths)
        assert 8 <= mean_len <= 25, f"Mean length {mean_len} outside expected range"


class TestSimulateCacheInvalidation:
    """Cache is properly invalidated on graph modification."""

    def test_cache_invalidated_on_remove_sequence(self, aap_lzgraph_copy):
        """simulate() works after remove_sequence()."""
        graph = aap_lzgraph_copy
        # Build cache
        results1 = graph.simulate(5, seed=42)
        assert len(results1) == 5

        # Remove a sequence (triggers recalculate which invalidates cache)
        seq = "CASSLGQAYEQYF"
        graph.remove_sequence(seq)

        # Cache should be rebuilt
        results2 = graph.simulate(5, seed=42)
        assert len(results2) == 5


class TestSimulateSaveLoad:
    """simulate() works after save/load cycle."""

    def test_simulate_after_pickle_load(self, aap_lzgraph):
        """Save, load, then simulate should work."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            aap_lzgraph.save(path)
            loaded = AAPLZGraph.load(path)
            results = loaded.simulate(10, seed=42)
            assert len(results) == 10
            assert all(isinstance(s, str) for s in results)
        finally:
            os.unlink(path)


# =========================================================================
# MLE stop probability tests
# =========================================================================


class TestMLEStopProbability:
    """Verify MLE stop probability computation."""

    def test_mle_formula(self, aap_lzgraph):
        """P(stop|t) = T(t) / (T(t) + f(t)) for all terminal states."""
        for state in aap_lzgraph.terminal_state_data.index:
            t_count = aap_lzgraph.terminal_states[state]
            f_count = aap_lzgraph.per_node_observed_frequency.get(state, 0)
            expected = t_count / (t_count + f_count) if (t_count + f_count) > 0 else 1.0
            actual = aap_lzgraph.terminal_state_data.loc[state, 'wsif/sep']
            assert abs(actual - expected) < 1e-10

    def test_range_zero_one(self, aap_lzgraph):
        """All stop probabilities in [0, 1] by construction."""
        probs = aap_lzgraph.terminal_state_data['wsif/sep'].values
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_effective_probs_sum_to_one(self, aap_lzgraph):
        """At each terminal: edge_weights * (1 - p_stop) + p_stop = 1."""
        for state in aap_lzgraph.terminal_state_data.index:
            p_stop = aap_lzgraph.terminal_state_data.loc[state, 'wsif/sep']
            succs = list(aap_lzgraph.graph.successors(state))
            if succs:
                edge_sum = sum(
                    aap_lzgraph.graph[state][s]['data'].weight for s in succs
                )
                effective_sum = edge_sum * (1 - p_stop) + p_stop
                assert abs(effective_sum - 1.0) < 0.01, \
                    f"State {state}: effective sum = {effective_sum}"

    def test_terminal_stop_dict_matches(self, aap_lzgraph):
        """_terminal_stop_dict should match terminal_state_data['wsif/sep']."""
        for state in aap_lzgraph.terminal_state_data.index:
            assert abs(
                aap_lzgraph._terminal_stop_dict[state]
                - aap_lzgraph.terminal_state_data.loc[state, 'wsif/sep']
            ) < 1e-15


# =========================================================================
# clean_node optimization tests
# =========================================================================


class TestCleanNodeOptimization:
    """Verify optimized clean_node produces same results as regex version."""

    def test_aap_clean_node(self, aap_lzgraph):
        """AAPLZGraph.clean_node matches regex version for all nodes."""
        import re
        for node in aap_lzgraph.graph.nodes():
            optimized = AAPLZGraph.clean_node(node)
            match = re.search(r'[A-Z]+', node)
            regex_result = match.group(0) if match else ""
            assert optimized == regex_result, f"Mismatch on '{node}': '{optimized}' vs '{regex_result}'"

    def test_ndp_clean_node(self, ndp_lzgraph):
        """NDPLZGraph.clean_node matches regex version for all nodes."""
        import re
        for node in ndp_lzgraph.graph.nodes():
            optimized = NDPLZGraph.clean_node(node)
            match = re.search(r'[ATGC]+', node)
            regex_result = match.group(0) if match else ""
            assert optimized == regex_result, f"Mismatch on '{node}': '{optimized}' vs '{regex_result}'"
