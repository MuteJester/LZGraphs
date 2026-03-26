"""Tests for graph operations, IO, gene data, and free functions."""

import os
import numpy as np
import pytest
from LZGraphs import (LZGraph, CorruptFileError, NoGeneDataError,
                       jensen_shannon_divergence, k_diversity,
                       saturation_curve, lz76_decompose)


class TestSetOperations:
    def test_union(self, aap_sequences):
        a = LZGraph(aap_sequences[:3], variant='aap')
        b = LZGraph(aap_sequences[3:], variant='aap')
        u = a | b
        assert u.n_nodes >= max(a.n_nodes, b.n_nodes)

    def test_union_method(self, aap_sequences):
        a = LZGraph(aap_sequences[:3], variant='aap')
        b = LZGraph(aap_sequences[3:], variant='aap')
        u = a.union(b)
        assert u.n_nodes > 0

    def test_intersection(self, aap_sequences):
        a = LZGraph(aap_sequences, variant='aap')
        b = LZGraph(aap_sequences[:3], variant='aap')
        i = a & b
        assert i.n_edges <= min(a.n_edges, b.n_edges)

    def test_difference(self, aap_sequences):
        a = LZGraph(aap_sequences, variant='aap')
        b = LZGraph(aap_sequences[:2], variant='aap')
        d = a - b
        assert d.n_nodes > 0

    def test_weighted_merge(self, aap_sequences):
        a = LZGraph(aap_sequences[:3], variant='aap')
        b = LZGraph(aap_sequences[3:], variant='aap')
        m = a.weighted_merge(b, alpha=0.7, beta=0.3)
        assert m.n_nodes > 0

    def test_variant_mismatch(self, aap_sequences):
        a = LZGraph(aap_sequences, variant='aap')
        b = LZGraph(['ATGCATGC', 'GCTAGCTA'], variant='ndp')
        with pytest.raises(ValueError, match='variant'):
            a | b


class TestPosterior:
    def test_posterior(self, aap_graph, aap_sequences):
        post = aap_graph.posterior(aap_sequences[:3], kappa=10.0)
        assert isinstance(post, LZGraph)
        assert post.n_nodes == aap_graph.n_nodes


class TestIO:
    def test_save_load_roundtrip(self, aap_graph, tmp_lzg):
        aap_graph.save(tmp_lzg)
        loaded = LZGraph.load(tmp_lzg)
        assert loaded.n_nodes == aap_graph.n_nodes
        assert loaded.n_edges == aap_graph.n_edges
        assert loaded.variant == aap_graph.variant

    def test_lzpgen_preserved(self, aap_graph, tmp_lzg):
        aap_graph.save(tmp_lzg)
        loaded = LZGraph.load(tmp_lzg)
        lp1 = aap_graph.lzpgen('CASSLGIRRT')
        lp2 = loaded.lzpgen('CASSLGIRRT')
        assert abs(lp1 - lp2) < 1e-10

    def test_gene_roundtrip(self, aap_gene_graph, tmp_lzg):
        aap_gene_graph.save(tmp_lzg)
        loaded = LZGraph.load(tmp_lzg)
        assert loaded.has_gene_data is True
        assert set(loaded.v_genes) == set(aap_gene_graph.v_genes)
        assert set(loaded.j_genes) == set(aap_gene_graph.j_genes)

    def test_load_nonexistent(self):
        with pytest.raises(OSError):
            LZGraph.load('/no/such/file.lzg')

    def test_load_corrupt(self, tmp_path):
        bad = str(tmp_path / 'bad.lzg')
        with open(bad, 'wb') as f:
            f.write(b'GARBAGE' * 20)
        with pytest.raises(CorruptFileError):
            LZGraph.load(bad)


class TestGeneData:
    def test_v_genes(self, aap_gene_graph):
        assert 'TRBV5-1' in aap_gene_graph.v_genes
        assert 'TRBV12-3' in aap_gene_graph.v_genes

    def test_j_genes(self, aap_gene_graph):
        assert 'TRBJ1-1' in aap_gene_graph.j_genes
        assert 'TRBJ2-7' in aap_gene_graph.j_genes

    def test_v_marginals(self, aap_gene_graph):
        vm = aap_gene_graph.v_marginals
        assert abs(sum(vm.values()) - 1.0) < 0.01

    def test_j_marginals(self, aap_gene_graph):
        jm = aap_gene_graph.j_marginals
        assert abs(sum(jm.values()) - 1.0) < 0.01

    def test_vj_distribution(self, aap_gene_graph):
        vj = aap_gene_graph.vj_distribution
        assert len(vj) > 0
        total = sum(d['prob'] for d in vj)
        assert abs(total - 1.0) < 0.01

    def test_no_gene_data(self, aap_graph):
        with pytest.raises(NoGeneDataError):
            _ = aap_graph.v_genes


class TestFeatures:
    def test_feature_stats(self, aap_graph):
        stats = aap_graph.feature_stats()
        assert isinstance(stats, np.ndarray)
        assert stats.shape == (15,)
        assert stats[0] == aap_graph.n_nodes  # first element is node count

    def test_feature_mass_profile(self, aap_graph):
        profile = aap_graph.feature_mass_profile(max_pos=30)
        assert isinstance(profile, np.ndarray)
        assert profile.shape == (31,)
        assert abs(sum(profile) - 1.0) < 0.1

    def test_feature_aligned(self, aap_sequences):
        ref = LZGraph(aap_sequences, variant='aap')
        query = LZGraph(aap_sequences[:3], variant='aap')
        vec = ref.feature_aligned(query)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (ref.n_nodes,)
        assert any(v > 0 for v in vec)  # some overlap


class TestFreeFunctions:
    def test_jsd_self(self, aap_graph):
        jsd = jensen_shannon_divergence(aap_graph, aap_graph)
        assert abs(jsd) < 1e-6

    def test_jsd_different(self, aap_sequences):
        a = LZGraph(aap_sequences[:3], variant='aap')
        b = LZGraph(aap_sequences[3:], variant='aap')
        jsd = jensen_shannon_divergence(a, b)
        assert jsd > 0

    def test_lz76_decompose(self):
        tokens = lz76_decompose('CASSLGIRRT')
        assert tokens == ['C', 'A', 'S', 'SL', 'G', 'I', 'R', 'RT']

    def test_k_diversity(self, aap_sequences):
        result = k_diversity(aap_sequences, k=3, variant='aap', draws=50, seed=42)
        assert 'mean' in result
        assert 'std' in result
        assert result['mean'] > 0

    def test_saturation_curve(self, aap_sequences):
        points = saturation_curve(aap_sequences, variant='aap', log_every=2)
        assert len(points) > 0
        assert points[-1]['n_sequences'] <= len(aap_sequences)
        # Monotonically increasing nodes
        nodes = [p['n_nodes'] for p in points]
        assert nodes == sorted(nodes)
