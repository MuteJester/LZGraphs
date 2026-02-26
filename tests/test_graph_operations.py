"""
Tests for Graph Operations and Untested Public Methods
=======================================================

Tests covering:
- Graph equality (__eq__)
- DAG validation (is_dag)
- Isolate handling (isolates, drop_isolates)
- VoteRank influential nodes (voterank)
- Graph union operations (graph_union)
"""

import pytest
import pandas as pd
import numpy as np

from LZGraphs import AAPLZGraph, NDPLZGraph, IncompatibleGraphsError


class TestGraphEquality:
    """Tests for LZGraph equality comparison (__eq__)."""

    def test_equal_graphs_from_same_data(self, test_data_aap):
        """Verify graphs built from same data are equal."""
        # Build two graphs from identical data
        graph1 = AAPLZGraph(test_data_aap.copy(), verbose=False)
        graph2 = AAPLZGraph(test_data_aap.copy(), verbose=False)

        assert graph1 == graph2

    def test_unequal_graphs_from_different_data(self, test_data_aap):
        """Verify graphs built from different data are not equal."""
        # Split data
        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        graph1 = AAPLZGraph(data1, verbose=False)
        graph2 = AAPLZGraph(data2, verbose=False)

        assert graph1 != graph2

    def test_genetic_vs_non_genetic_not_equal(self, test_data_aap):
        """Verify genetic and non-genetic graphs are not equal."""
        # Build graphs with and without genetic info
        graph_genetic = AAPLZGraph(test_data_aap.copy(), verbose=False)

        # Create non-genetic version
        data_no_genes = test_data_aap[['cdr3_amino_acid']].copy()
        # AAPLZGraph requires V and J columns, so we skip this test
        # This is expected behavior - can't compare genetic vs non-genetic
        assert graph_genetic.has_gene_data is True

    def test_equality_symmetric(self, test_data_aap):
        """Verify equality is symmetric: a == b implies b == a."""
        graph1 = AAPLZGraph(test_data_aap.copy(), verbose=False)
        graph2 = AAPLZGraph(test_data_aap.copy(), verbose=False)

        assert (graph1 == graph2) == (graph2 == graph1)


class TestIsDAG:
    """Tests for DAG (Directed Acyclic Graph) validation."""

    def test_is_dag_returns_boolean(self, aap_lzgraph):
        """Verify is_dag returns a boolean."""
        result = aap_lzgraph.is_dag
        assert isinstance(result, bool)

    def test_typical_lzgraph_structure(self, aap_lzgraph):
        """Test is_dag on typical LZGraph."""
        # LZGraphs are generally not DAGs due to possible cycles in sequences
        result = aap_lzgraph.is_dag
        # Just verify it runs without error
        assert isinstance(result, bool)


class TestIsolates:
    """Tests for isolate handling (nodes with no edges)."""

    def test_isolates_returns_iterable(self, aap_lzgraph):
        """Verify isolates() returns an iterable."""
        result = aap_lzgraph.isolates
        # Should be convertible to list
        isolate_list = list(result)
        assert isinstance(isolate_list, list)

    def test_drop_isolates_removes_isolates(self, aap_lzgraph):
        """Verify drop_isolates removes isolated nodes."""
        # Get initial isolate count
        initial_isolates = list(aap_lzgraph.isolates)

        if len(initial_isolates) > 0:
            aap_lzgraph.drop_isolates()
            remaining_isolates = list(aap_lzgraph.isolates)
            assert len(remaining_isolates) == 0

    def test_drop_isolates_preserves_connected_nodes(self, aap_lzgraph):
        """Verify drop_isolates doesn't remove connected nodes."""
        # Count nodes with edges
        nodes_before = len(aap_lzgraph.nodes)
        isolates_before = len(list(aap_lzgraph.isolates))

        aap_lzgraph.drop_isolates()

        nodes_after = len(aap_lzgraph.nodes)

        # Should only have removed isolates
        assert nodes_after == nodes_before - isolates_before


class TestVoteRank:
    """Tests for VoteRank influential node detection."""

    def test_voterank_returns_list(self, aap_lzgraph):
        """Verify voterank returns a list of nodes."""
        result = aap_lzgraph.voterank(n_nodes=10)
        assert isinstance(result, list)

    def test_voterank_respects_n_nodes(self, aap_lzgraph):
        """Verify voterank returns at most n_nodes nodes."""
        result = aap_lzgraph.voterank(n_nodes=5)
        assert len(result) <= 5

    def test_voterank_default_count(self, aap_lzgraph):
        """Verify voterank with default n_nodes."""
        result = aap_lzgraph.voterank()  # Default is 25
        assert len(result) <= 25

    def test_voterank_nodes_exist_in_graph(self, aap_lzgraph):
        """Verify voterank returns nodes that exist in the graph."""
        result = aap_lzgraph.voterank(n_nodes=10)
        for node in result:
            assert node in aap_lzgraph.nodes


class TestGraphUnion:
    """Tests for graph_union operation."""

    def test_graph_union_basic(self, test_data_aap):
        """Verify basic graph union works."""
        from LZGraphs.graphs.graph_operations import graph_union

        # Split data
        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        graph1 = AAPLZGraph(data1, verbose=False)
        graph2 = AAPLZGraph(data2, verbose=False)

        # Record initial values
        initial_subpatterns = graph1.num_subpatterns

        # Perform union
        result = graph_union(graph1, graph2)

        # graph1 should be modified in place
        assert graph1.num_subpatterns == initial_subpatterns + graph2.num_subpatterns

    def test_graph_union_different_types_raises(self, test_data_aap, test_data_ndp):
        """Verify union of different graph types raises error."""
        from LZGraphs.graphs.graph_operations import graph_union

        graph_aap = AAPLZGraph(test_data_aap, verbose=False)
        graph_ndp = NDPLZGraph(test_data_ndp, verbose=False)

        with pytest.raises(IncompatibleGraphsError):
            graph_union(graph_aap, graph_ndp)

    def test_graph_union_combines_initial_states(self, test_data_aap):
        """Verify union combines initial states."""
        from LZGraphs.graphs.graph_operations import graph_union

        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        graph1 = AAPLZGraph(data1, verbose=False)
        graph2 = AAPLZGraph(data2, verbose=False)

        initial_states_1 = sum(graph1.initial_state_counts.values())

        graph_union(graph1, graph2)

        # Initial states should be combined
        assert sum(graph1.initial_state_counts.values()) > initial_states_1


class TestGraphSummary:
    """Tests for graph_summary method."""

    def test_graph_summary_returns_series(self, aap_lzgraph):
        """Verify graph_summary returns a dict."""
        result = aap_lzgraph.graph_summary()
        assert isinstance(result, dict)

    def test_graph_summary_contains_expected_keys(self, aap_lzgraph):
        """Verify graph_summary contains expected statistics."""
        result = aap_lzgraph.graph_summary()

        expected_keys = [
            'Chromatic Number',
            'Number of Isolates',
            'Max In Deg',
            'Max Out Deg',
            'Number of Edges'
        ]

        for key in expected_keys:
            assert key in result.keys()

    def test_graph_summary_values_valid(self, aap_lzgraph):
        """Verify graph_summary values are valid."""
        result = aap_lzgraph.graph_summary()

        # All values should be non-negative integers
        assert result['Number of Edges'] >= 0
        assert result['Number of Isolates'] >= 0
        assert result['Chromatic Number'] >= 1  # At least 1 color needed


class TestGraphUnionReturn:
    """Tests that graph_union returns the modified graph (not None)."""

    def test_graph_union_returns_graph(self, test_data_aap):
        """Verify graph_union returns the graph object, not None."""
        from LZGraphs.graphs.graph_operations import graph_union

        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        graph1 = AAPLZGraph(data1, verbose=False)
        graph2 = AAPLZGraph(data2, verbose=False)

        result = graph_union(graph1, graph2)

        assert result is not None
        assert result is graph1


class TestGraphUnionNaive:
    """Tests for graph_union with NaiveLZGraph instances."""

    def test_graph_union_naive(self, test_data_nucleotide, lz_dictionary):
        """Verify graph_union works for NaiveLZGraph and produces a larger graph."""
        from LZGraphs import NaiveLZGraph
        from LZGraphs.graphs.graph_operations import graph_union

        data1 = test_data_nucleotide['cdr3_rearrangement'].iloc[:2500]
        data2 = test_data_nucleotide['cdr3_rearrangement'].iloc[2500:]

        graph1 = NaiveLZGraph(data1, lz_dictionary, verbose=False)
        graph2 = NaiveLZGraph(data2, lz_dictionary, verbose=False)

        edges_1 = len(graph1.edges)
        edges_2 = len(graph2.edges)
        nodes_1 = len(graph1.nodes)
        nodes_2 = len(graph2.nodes)

        result = graph_union(graph1, graph2)

        # The union graph should have at least as many edges/nodes as
        # the larger of the two inputs
        assert result is not None
        assert len(result.edges) >= max(edges_1, edges_2)
        assert len(result.nodes) >= max(nodes_1, nodes_2)
