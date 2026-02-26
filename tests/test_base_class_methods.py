"""
Tests for LZGraphBase Methods and Quality Improvements
=======================================================

Tests covering previously untested public methods on the base class
and graph classes, plus regression tests for quality improvements.

Test Categories:
- Graph analysis methods (voterank, drop_isolates, eigenvector_centrality)
- Graph metadata and summary
- NaiveLZGraph.__eq__ with incompatible types
- NodeEdgeSaturationProbe __repr__
- utilities module export consistency
"""

import pytest
import numpy as np
import pandas as pd

from LZGraphs import (
    AAPLZGraph,
    NDPLZGraph,
    NaiveLZGraph,
    NodeEdgeSaturationProbe,
)


# =========================================================================
# Graph Analysis Methods
# =========================================================================

class TestVoteRank:
    """Tests for LZGraphBase.voterank method."""

    def test_voterank_returns_list(self, aap_lzgraph):
        """Verify voterank returns a list of nodes."""
        result = aap_lzgraph.voterank(n_nodes=10)
        assert isinstance(result, list)
        assert len(result) <= 10

    def test_voterank_nodes_exist_in_graph(self, aap_lzgraph):
        """Verify all returned nodes exist in the graph."""
        result = aap_lzgraph.voterank(n_nodes=5)
        for node in result:
            assert node in aap_lzgraph.graph.nodes()

    def test_voterank_ndp(self, ndp_lzgraph):
        """Verify voterank works on NDPLZGraph."""
        result = ndp_lzgraph.voterank(n_nodes=5)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_voterank_naive(self, naive_lzgraph):
        """Verify voterank works on NaiveLZGraph."""
        result = naive_lzgraph.voterank(n_nodes=5)
        assert isinstance(result, list)


class TestDropIsolates:
    """Tests for LZGraphBase.drop_isolates method."""

    def test_isolates_property(self, aap_lzgraph):
        """Verify isolates property returns a list."""
        isolates = aap_lzgraph.isolates
        assert isinstance(isolates, list)

    def test_drop_isolates_reduces_or_maintains_count(self, aap_lzgraph_copy):
        """Verify drop_isolates doesn't add nodes."""
        n_before = aap_lzgraph_copy.graph.number_of_nodes()
        aap_lzgraph_copy.drop_isolates()
        n_after = aap_lzgraph_copy.graph.number_of_nodes()
        assert n_after <= n_before

    def test_no_isolates_after_drop(self, aap_lzgraph_copy):
        """Verify no isolates remain after drop_isolates."""
        aap_lzgraph_copy.drop_isolates()
        assert len(aap_lzgraph_copy.isolates) == 0


class TestEigenvectorCentrality:
    """Tests for LZGraphBase.eigenvector_centrality method."""

    def test_eigenvector_centrality_returns_dict(self, aap_lzgraph):
        """Verify eigenvector_centrality returns a dictionary."""
        result = aap_lzgraph.eigenvector_centrality()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_eigenvector_centrality_values_positive(self, aap_lzgraph):
        """Verify centrality values are non-negative."""
        result = aap_lzgraph.eigenvector_centrality()
        for value in result.values():
            assert value >= 0

    def test_eigenvector_centrality_ndp(self, ndp_lzgraph):
        """Verify eigenvector_centrality works on NDPLZGraph."""
        result = ndp_lzgraph.eigenvector_centrality()
        assert isinstance(result, dict)
        assert len(result) > 0


class TestIsDag:
    """Tests for LZGraphBase.is_dag property."""

    def test_is_dag_returns_bool(self, aap_lzgraph):
        """Verify is_dag returns a boolean."""
        result = aap_lzgraph.is_dag
        assert isinstance(result, bool)


class TestGraphSummary:
    """Tests for LZGraphBase.graph_summary method."""

    def test_graph_summary_returns_series(self, aap_lzgraph):
        """Verify graph_summary returns a dict."""
        result = aap_lzgraph.graph_summary()
        assert isinstance(result, dict)

    def test_graph_summary_contains_expected_keys(self, aap_lzgraph):
        """Verify graph_summary contains all expected keys."""
        result = aap_lzgraph.graph_summary()
        expected_keys = [
            'Chromatic Number', 'Number of Isolates',
            'Max In Deg', 'Max Out Deg', 'Number of Edges'
        ]
        for key in expected_keys:
            assert key in result.keys()

    def test_graph_summary_positive_edges(self, aap_lzgraph):
        """Verify graph summary reports positive edge count."""
        result = aap_lzgraph.graph_summary()
        assert result['Number of Edges'] > 0


# =========================================================================
# Graph Metadata
# =========================================================================

class TestGetGraphMetadata:
    """Tests for LZGraphBase.get_graph_metadata method."""

    def test_metadata_returns_dict(self, aap_lzgraph):
        """Verify get_graph_metadata returns a dictionary."""
        result = aap_lzgraph.get_graph_metadata()
        assert isinstance(result, dict)

    def test_metadata_contains_expected_keys(self, aap_lzgraph):
        """Verify metadata contains all expected keys."""
        result = aap_lzgraph.get_graph_metadata()
        expected_keys = [
            'class', 'n_nodes', 'n_edges', 'genetic',
            'n_subpatterns', 'n_transitions',
            'n_initial_states', 'n_terminal_states'
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_metadata_class_name_correct(self, aap_lzgraph, ndp_lzgraph, naive_lzgraph):
        """Verify class name is correct for each graph type."""
        assert aap_lzgraph.get_graph_metadata()['class'] == 'AAPLZGraph'
        assert ndp_lzgraph.get_graph_metadata()['class'] == 'NDPLZGraph'
        assert naive_lzgraph.get_graph_metadata()['class'] == 'NaiveLZGraph'

    def test_metadata_genetic_flag(self, aap_lzgraph):
        """Verify genetic flag is correctly reported."""
        result = aap_lzgraph.get_graph_metadata()
        assert result['genetic'] is True

    def test_metadata_positive_counts(self, aap_lzgraph):
        """Verify node and edge counts are positive."""
        result = aap_lzgraph.get_graph_metadata()
        assert result['n_nodes'] > 0
        assert result['n_edges'] > 0
        assert result['n_initial_states'] > 0
        assert result['n_terminal_states'] > 0


# =========================================================================
# NaiveLZGraph.__eq__ Regression
# =========================================================================

class TestNaiveLZGraphEquality:
    """Tests for NaiveLZGraph.__eq__ with incompatible types."""

    def test_eq_with_string_returns_not_implemented(self, naive_lzgraph):
        """Verify __eq__ with non-NaiveLZGraph returns NotImplemented."""
        result = naive_lzgraph.__eq__("not a graph")
        assert result is NotImplemented

    def test_eq_with_none(self, naive_lzgraph):
        """Verify __eq__ with None returns NotImplemented."""
        result = naive_lzgraph.__eq__(None)
        assert result is NotImplemented

    def test_eq_with_int(self, naive_lzgraph):
        """Verify __eq__ with int returns NotImplemented."""
        result = naive_lzgraph.__eq__(42)
        assert result is NotImplemented

    def test_eq_same_graph_is_true(self, naive_lzgraph):
        """Verify a graph equals itself."""
        assert naive_lzgraph == naive_lzgraph

    def test_ne_with_different_type_no_error(self, naive_lzgraph):
        """Verify != with incompatible type doesn't raise."""
        result = (naive_lzgraph != "not a graph")
        assert result is True


# =========================================================================
# NodeEdgeSaturationProbe __repr__
# =========================================================================

class TestSaturationProbeRepr:
    """Tests for NodeEdgeSaturationProbe.__repr__ method."""

    def test_repr_fresh_probe(self):
        """Verify repr on fresh probe shows zeros."""
        probe = NodeEdgeSaturationProbe(node_function='aap')
        result = repr(probe)
        assert 'NodeEdgeSaturationProbe' in result
        assert 'nodes=0' in result
        assert 'edges=0' in result
        assert 'aap' in result

    def test_repr_after_processing(self, test_data_aap):
        """Verify repr updates after processing sequences."""
        probe = NodeEdgeSaturationProbe(node_function='aap')
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:10].tolist()
        probe.test_sequences(sequences, log_every=10)

        result = repr(probe)
        assert 'nodes=' in result
        # After processing, should have non-zero counts
        assert probe.nodes  # non-empty set

    def test_repr_with_naive(self):
        """Verify repr works with naive node function."""
        probe = NodeEdgeSaturationProbe(node_function='naive')
        result = repr(probe)
        assert 'naive' in result

    def test_repr_with_ndp(self):
        """Verify repr works with ndp node function."""
        probe = NodeEdgeSaturationProbe(node_function='ndp')
        result = repr(probe)
        assert 'ndp' in result

    def test_repr_with_custom_function(self):
        """Verify repr works with custom callable."""
        probe = NodeEdgeSaturationProbe(node_function=lambda x: list(x))
        result = repr(probe)
        assert 'NodeEdgeSaturationProbe' in result


# =========================================================================
# Saturation Probe Methods
# =========================================================================

class TestSaturationRate:
    """Tests for NodeEdgeSaturationProbe.saturation_rate method."""

    def test_saturation_rate_returns_dataframe(self, test_data_aap):
        """Verify saturation_rate returns a list of dicts."""
        probe = NodeEdgeSaturationProbe(node_function='aap')
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        result = probe.saturation_rate(sequences, log_every=20)
        assert isinstance(result, list)
        assert all('n_sequences' in row for row in result)
        assert all('rate' in row for row in result)

    def test_saturation_rate_positive_initially(self, test_data_aap):
        """Verify discovery rate is positive at the start."""
        probe = NodeEdgeSaturationProbe(node_function='aap')
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        result = probe.saturation_rate(sequences, log_every=20)
        if len(result) > 0:
            assert result[0]['rate'] > 0


# =========================================================================
# Utilities Import Consistency
# =========================================================================

class TestUtilitiesExport:
    """Tests for utilities module export consistency."""

    def test_lempel_ziv_from_utilities(self):
        """Verify lempel_ziv_decomposition is importable from utilities."""
        from LZGraphs.utilities import lempel_ziv_decomposition
        result = lempel_ziv_decomposition("ATGC")
        assert result == ['A', 'T', 'G', 'C']

    def test_lempel_ziv_from_top_level(self):
        """Verify lempel_ziv_decomposition is importable from top-level."""
        from LZGraphs import lempel_ziv_decomposition
        result = lempel_ziv_decomposition("ATGC")
        assert result == ['A', 'T', 'G', 'C']


# =========================================================================
# Graph Repr
# =========================================================================

class TestGraphRepr:
    """Tests for __repr__ across all graph types."""

    def test_ndp_repr(self, ndp_lzgraph):
        """Verify NDPLZGraph repr works (inherited from base)."""
        result = repr(ndp_lzgraph)
        assert 'NDPLZGraph' in result or 'LZGraph' in result

    def test_naive_repr(self, naive_lzgraph):
        """Verify NaiveLZGraph repr contains expected info."""
        result = repr(naive_lzgraph)
        assert 'NaiveLZGraph' in result
        assert 'nodes=' in result
        assert 'edges=' in result
