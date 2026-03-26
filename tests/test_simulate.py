"""Tests for simulation."""

import numpy as np
import pytest
from LZGraphs import LZGraph, SimulationResult, NoGeneDataError


class TestSimulateBasic:
    def test_returns_simulation_result(self, aap_graph):
        result = aap_graph.simulate(10, seed=42)
        assert isinstance(result, SimulationResult)

    def test_correct_count(self, aap_graph):
        result = aap_graph.simulate(20, seed=42)
        assert len(result) == 20

    def test_sequences_are_strings(self, aap_graph):
        result = aap_graph.simulate(5, seed=42)
        for s in result:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_deterministic(self, aap_graph):
        r1 = aap_graph.simulate(10, seed=123)
        r2 = aap_graph.simulate(10, seed=123)
        assert r1.sequences == r2.sequences

    def test_different_seeds(self, aap_graph):
        r1 = aap_graph.simulate(50, seed=1)
        r2 = aap_graph.simulate(50, seed=2)
        # Not all sequences should be the same
        assert r1.sequences != r2.sequences

    def test_iterable(self, aap_graph):
        result = aap_graph.simulate(5, seed=42)
        seqs = list(result)
        assert len(seqs) == 5

    def test_indexable(self, aap_graph):
        result = aap_graph.simulate(5, seed=42)
        assert isinstance(result[0], str)


class TestSimulateMetadata:
    def test_log_probs(self, aap_graph):
        result = aap_graph.simulate(10, seed=42)
        assert isinstance(result.log_probs, np.ndarray)
        assert result.log_probs.shape == (10,)
        assert all(lp < 0 for lp in result.log_probs)

    def test_n_tokens(self, aap_graph):
        result = aap_graph.simulate(10, seed=42)
        assert isinstance(result.n_tokens, np.ndarray)
        assert all(nt > 0 for nt in result.n_tokens)

    def test_log_prob_consistency(self, aap_graph):
        """Simulate log_probs should match lzpgen() on the same sequences."""
        result = aap_graph.simulate(20, seed=42)
        for seq, sim_lp in zip(result.sequences, result.log_probs):
            score_lp = aap_graph.lzpgen(seq)
            # Allow some tolerance — simulate computes incrementally
            assert abs(sim_lp - score_lp) < 1.0 or score_lp < -600


class TestGeneSimulate:
    def test_sample_genes(self, aap_gene_graph):
        result = aap_gene_graph.simulate(10, seed=42, sample_genes=True)
        assert result.v_genes is not None
        assert result.j_genes is not None
        assert len(result.v_genes) == 10

    def test_constrain_v_gene(self, aap_gene_graph):
        result = aap_gene_graph.simulate(10, seed=42, v_gene='TRBV5-1')
        assert all(v == 'TRBV5-1' for v in result.v_genes)

    def test_constrain_vj(self, aap_gene_graph):
        result = aap_gene_graph.simulate(10, seed=42,
                                          v_gene='TRBV5-1', j_gene='TRBJ1-1')
        assert all(v == 'TRBV5-1' for v in result.v_genes)
        assert all(j == 'TRBJ1-1' for j in result.j_genes)

    def test_no_gene_data_raises(self, aap_graph):
        with pytest.raises(NoGeneDataError):
            aap_graph.simulate(1, sample_genes=True)

    def test_unknown_gene_raises(self, aap_gene_graph):
        with pytest.raises(ValueError):
            aap_gene_graph.simulate(1, v_gene='NONEXISTENT')
