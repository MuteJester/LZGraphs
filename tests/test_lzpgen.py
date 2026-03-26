"""Tests for LZPGEN computation."""

import math
import numpy as np
import pytest
from LZGraphs import LZGraph


class TestLzpgenSingle:
    def test_known_sequence(self, aap_graph):
        lp = aap_graph.lzpgen('CASSLGIRRT')
        assert isinstance(lp, float)
        assert lp < 0  # log-prob is negative
        assert lp > -100  # should be reasonable, not -690

    def test_unknown_sequence(self, aap_graph):
        lp = aap_graph.lzpgen('XXXXXXXXXX')
        assert lp < -600  # should be near LZG_LOG_EPS

    def test_prob_mode(self, aap_graph):
        lp = aap_graph.lzpgen('CASSLGIRRT', log=True)
        p = aap_graph.lzpgen('CASSLGIRRT', log=False)
        assert abs(math.exp(lp) - p) < 1e-10

    def test_log_default(self, aap_graph):
        lp = aap_graph.lzpgen('CASSLGIRRT')
        assert lp < 0  # default is log


class TestLzpgenBatch:
    def test_batch_returns_array(self, aap_graph):
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF']
        result = aap_graph.lzpgen(seqs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_batch_matches_single(self, aap_graph):
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF']
        batch = aap_graph.lzpgen(seqs)
        for i, s in enumerate(seqs):
            single = aap_graph.lzpgen(s)
            assert abs(batch[i] - single) < 1e-10

    def test_batch_prob_mode(self, aap_graph):
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF']
        lps = aap_graph.lzpgen(seqs, log=True)
        ps = aap_graph.lzpgen(seqs, log=False)
        np.testing.assert_allclose(np.exp(lps), ps, atol=1e-10)


class TestLzpgenProperties:
    def test_training_sequences_have_positive_prob(self, aap_sequences, aap_graph):
        """At least some training sequences should have non-trivial probability."""
        lps = aap_graph.lzpgen(aap_sequences)
        assert any(lp > -100 for lp in lps)
