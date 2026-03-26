"""Tests for LZGraph construction, properties, and dunders."""

import math
import numpy as np
import pytest
from LZGraphs import LZGraph, LZGraphError, NoGeneDataError


class TestConstruction:
    def test_basic_build(self, aap_graph):
        assert aap_graph.n_nodes > 0
        assert aap_graph.n_edges > 0
        assert aap_graph.variant == 'aap'

    def test_all_variants(self, aap_sequences):
        for v in ('aap', 'ndp', 'naive'):
            # ndp/naive expect nucleotides but won't crash on AA — just different tokenization
            g = LZGraph(aap_sequences, variant=v)
            assert g.n_nodes > 0
            assert g.variant == v

    def test_with_abundances(self, aap_sequences):
        abundances = [1, 2, 3, 1, 2, 1]
        g = LZGraph(aap_sequences, variant='aap', abundances=abundances)
        assert g.n_nodes > 0

    def test_with_genes(self, aap_gene_graph):
        assert aap_gene_graph.has_gene_data is True

    def test_without_genes(self, aap_graph):
        assert aap_graph.has_gene_data is False

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            LZGraph([], variant='aap')

    def test_string_raises(self):
        with pytest.raises(TypeError):
            LZGraph('CASSLGIRRT', variant='aap')

    def test_bad_variant(self, aap_sequences):
        with pytest.raises(ValueError):
            LZGraph(aap_sequences, variant='invalid')


class TestProperties:
    def test_n_nodes(self, aap_graph):
        assert isinstance(aap_graph.n_nodes, int)
        assert aap_graph.n_nodes > 0

    def test_n_edges(self, aap_graph):
        assert isinstance(aap_graph.n_edges, int)
        assert aap_graph.n_edges > 0

    def test_path_count(self, aap_graph):
        assert aap_graph.path_count > 0

    def test_is_dag(self, aap_graph):
        assert aap_graph.is_dag is True


class TestStructuralProperties:
    def test_n_sequences(self, aap_graph, aap_sequences):
        assert aap_graph.n_sequences == len(aap_sequences)

    def test_length_distribution(self, aap_graph, aap_sequences):
        ld = aap_graph.length_distribution
        assert isinstance(ld, dict)
        assert all(isinstance(k, int) for k in ld)
        assert all(isinstance(v, int) for v in ld.values())
        assert sum(ld.values()) == len(aap_sequences)
        # Each sequence length should appear
        for seq in aap_sequences:
            assert len(seq) in ld

    def test_nodes(self, aap_graph):
        nodes = aap_graph.nodes
        assert isinstance(nodes, list)
        assert len(nodes) > 0
        # Should not contain sentinel nodes
        for n in nodes:
            assert not n.startswith('@'), f"sentinel @ in nodes: {n}"
            assert '$' not in n, f"sentinel $ in nodes: {n}"
        # All should be strings
        assert all(isinstance(n, str) for n in nodes)

    def test_all_nodes(self, aap_graph):
        all_nodes = aap_graph.all_nodes
        nodes = aap_graph.nodes
        # all_nodes should be a superset of nodes
        assert len(all_nodes) >= len(nodes)
        # Should contain the sentinels
        has_start = any(n.startswith('@') for n in all_nodes)
        has_end = any('$' in n for n in all_nodes)
        assert has_start
        assert has_end

    def test_edges(self, aap_graph):
        edges = aap_graph.edges
        assert isinstance(edges, list)
        assert len(edges) > 0
        # Each edge is (src, dst, weight, count)
        for src, dst, w, c in edges:
            assert isinstance(src, str)
            assert isinstance(dst, str)
            assert isinstance(w, float)
            assert isinstance(c, int)
            assert 0.0 <= w <= 1.0
            assert c > 0
            # No sentinels
            assert not src.startswith('@') and '$' not in src
            assert not dst.startswith('@') and '$' not in dst

    def test_all_edges(self, aap_graph):
        all_edges = aap_graph.all_edges
        edges = aap_graph.edges
        assert len(all_edges) >= len(edges)
        assert len(all_edges) == aap_graph.n_edges

    def test_n_initial(self, aap_graph):
        assert isinstance(aap_graph.n_initial, int)
        assert aap_graph.n_initial >= 1

    def test_n_terminal(self, aap_graph):
        assert isinstance(aap_graph.n_terminal, int)
        assert aap_graph.n_terminal >= 1

    def test_max_out_degree(self, aap_graph):
        assert isinstance(aap_graph.max_out_degree, int)
        assert aap_graph.max_out_degree >= 1

    def test_max_in_degree(self, aap_graph):
        assert isinstance(aap_graph.max_in_degree, int)
        assert aap_graph.max_in_degree >= 1

    def test_density(self, aap_graph):
        d = aap_graph.density
        assert isinstance(d, float)
        assert 0.0 < d < 1.0

    def test_out_degrees(self, aap_graph):
        od = aap_graph.out_degrees
        assert isinstance(od, np.ndarray)
        assert od.dtype == np.uint32
        assert len(od) == aap_graph.n_nodes
        assert od.sum() == aap_graph.n_edges

    def test_in_degrees(self, aap_graph):
        ind = aap_graph.in_degrees
        assert isinstance(ind, np.ndarray)
        assert ind.dtype == np.uint32
        assert len(ind) == aap_graph.n_nodes
        assert ind.sum() == aap_graph.n_edges

    def test_degrees_consistent(self, aap_graph):
        # Sum of out-degrees == sum of in-degrees == n_edges
        assert aap_graph.out_degrees.sum() == aap_graph.in_degrees.sum()

    def test_successors(self, aap_graph):
        # C_2 is the first real node after @ — should have successors
        nodes = aap_graph.all_nodes
        # Find a node with known successors
        first_real = [n for n in nodes if not n.startswith('@') and '$' not in n][0]
        succs = aap_graph.successors(first_real)
        assert isinstance(succs, list)
        # Each successor is (label, weight, count)
        for label, w, c in succs:
            assert isinstance(label, str)
            assert isinstance(w, float)
            assert isinstance(c, int)

    def test_successors_missing_node(self, aap_graph):
        with pytest.raises(KeyError):
            aap_graph.successors('NONEXISTENT_99')

    def test_adjacency_csr(self, aap_graph):
        csr = aap_graph.adjacency_csr()
        assert isinstance(csr, dict)
        assert csr['row_offsets'].shape == (aap_graph.n_nodes + 1,)
        assert csr['col_indices'].shape == (aap_graph.n_edges,)
        assert csr['weights'].shape == (aap_graph.n_edges,)
        assert csr['counts'].shape == (aap_graph.n_edges,)
        assert csr['row_offsets'].dtype == np.uint32
        assert csr['weights'].dtype == np.float64

    def test_adjacency_csr_scipy_interop(self, aap_graph):
        """CSR arrays work with scipy.sparse.csr_matrix."""
        pytest.importorskip('scipy')
        from scipy.sparse import csr_matrix
        csr = aap_graph.adjacency_csr()
        A = csr_matrix(
            (csr['weights'], csr['col_indices'], csr['row_offsets']),
            shape=(aap_graph.n_nodes, aap_graph.n_nodes),
        )
        assert A.nnz == aap_graph.n_edges

    def test_node_labels_aap_format(self, aap_sequences):
        g = LZGraph(aap_sequences, variant='aap')
        for node in g.nodes:
            # AAP nodes should be "letters_digits" format
            assert '_' in node, f"AAP node missing underscore: {node}"

    def test_node_labels_naive_format(self, aap_sequences):
        g = LZGraph(aap_sequences, variant='naive')
        for node in g.nodes:
            # Naive nodes should NOT have positional encoding
            assert '_' not in node, f"Naive node has underscore: {node}"


class TestDunders:
    def test_repr(self, aap_graph):
        r = repr(aap_graph)
        assert 'LZGraph' in r
        assert 'aap' in r
        assert 'nodes=' in r

    def test_len(self, aap_graph):
        assert len(aap_graph) == aap_graph.n_nodes

    def test_contains_known(self, aap_graph):
        assert 'CASSLGIRRT' in aap_graph

    def test_contains_unknown(self, aap_graph):
        assert 'XXXXXXXXXX' not in aap_graph

    def test_summary(self, aap_graph):
        s = aap_graph.summary()
        assert 'n_nodes' in s
        assert 'n_edges' in s
        assert 'is_dag' in s
        assert s['n_nodes'] == aap_graph.n_nodes
