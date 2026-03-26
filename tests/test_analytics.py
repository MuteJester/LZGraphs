"""Tests for analytics, diversity, occupancy, and PGEN distribution."""

import math
import numpy as np
import pytest
from LZGraphs import LZGraph, PgenDistribution


class TestDiversity:
    def test_effective_diversity(self, aap_graph):
        d = aap_graph.effective_diversity()
        assert isinstance(d, float)
        assert d > 1.0

    def test_diversity_profile(self, aap_graph):
        dp = aap_graph.diversity_profile()
        assert 'entropy_nats' in dp
        assert 'entropy_bits' in dp
        assert 'effective_diversity' in dp
        assert 'uniformity' in dp
        assert dp['effective_diversity'] == pytest.approx(
            math.exp(dp['entropy_nats']), rel=1e-6)

    def test_hill_number(self, aap_graph):
        d0 = aap_graph.hill_number(0)
        d1 = aap_graph.hill_number(1)
        d2 = aap_graph.hill_number(2)
        # Hill number monotonicity: D(0) >= D(1) >= D(2)
        assert d0 >= d1 - 0.01
        assert d1 >= d2 - 0.01

    def test_hill_numbers_batch(self, aap_graph):
        orders = [0, 1, 2, 5]
        result = aap_graph.hill_numbers(orders)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_hill_curve(self, aap_graph):
        hc = aap_graph.hill_curve()
        assert 'orders' in hc
        assert 'values' in hc
        assert isinstance(hc['orders'], np.ndarray)
        assert len(hc['orders']) > 5

    def test_hill_curve_custom(self, aap_graph):
        hc = aap_graph.hill_curve(orders=[0, 1, 2])
        assert len(hc['orders']) == 3

    def test_power_sum(self, aap_graph):
        m1 = aap_graph.power_sum(1.0)
        assert abs(m1 - 1.0) < 1e-6  # M(1) = sum(pi) = 1

    def test_d1_equals_exp_entropy(self, aap_graph):
        d1 = aap_graph.hill_number(1.0)
        dp = aap_graph.diversity_profile()
        # MC estimation: allow 15% relative tolerance
        ratio = max(d1, dp['effective_diversity']) / max(min(d1, dp['effective_diversity']), 0.1)
        assert ratio < 1.15


class TestPgenDiagnostics:
    def test_proper_distribution(self, aap_graph):
        diag = aap_graph.pgen_diagnostics()
        assert diag['is_proper'] is True
        assert abs(diag['total_absorbed'] - 1.0) < 1e-6

    def test_dynamic_range(self, aap_graph):
        dr = aap_graph.pgen_dynamic_range()
        assert isinstance(dr, float)
        assert dr >= 0

    def test_dynamic_range_detail(self, aap_graph):
        dd = aap_graph.pgen_dynamic_range_detail()
        assert dd['max_log_prob'] >= dd['min_log_prob']


class TestPgenDistribution:
    def test_moments(self, aap_graph):
        m = aap_graph.pgen_moments()
        assert 'mean' in m
        assert 'variance' in m
        assert 'std' in m
        assert m['mean'] < 0  # log-probs are negative
        assert m['variance'] > 0
        assert abs(m['std'] - math.sqrt(m['variance'])) < 1e-6

    def test_analytical(self, aap_graph):
        dist = aap_graph.pgen_distribution()
        assert isinstance(dist, PgenDistribution)
        assert dist.n_components > 0

    def test_pdf(self, aap_graph):
        dist = aap_graph.pgen_distribution()
        x = dist.mean
        assert dist.pdf(x) > 0

    def test_cdf(self, aap_graph):
        dist = aap_graph.pgen_distribution()
        assert 0 < dist.cdf(dist.mean) < 1

    def test_sample(self, aap_graph):
        dist = aap_graph.pgen_distribution()
        samples = dist.sample(1000, seed=42)
        assert samples.shape == (1000,)
        assert abs(np.mean(samples) - dist.mean) < 0.5


class TestOccupancy:
    def test_richness_monotone(self, aap_graph):
        r1 = aap_graph.predicted_richness(1.0)
        r10 = aap_graph.predicted_richness(10.0)
        r100 = aap_graph.predicted_richness(100.0)
        assert r1 <= r10 <= r100

    def test_richness_positive(self, aap_graph):
        r_big = aap_graph.predicted_richness(1e6)
        assert r_big > 0

    def test_overlap(self, aap_graph):
        ov = aap_graph.predicted_overlap(100.0, 100.0)
        assert ov > 0

    def test_richness_curve(self, aap_graph):
        depths = [1.0, 10.0, 100.0]
        curve = aap_graph.richness_curve(depths)
        assert isinstance(curve, np.ndarray)
        assert curve.shape == (3,)
        assert curve[0] <= curve[1] <= curve[2]


class TestPerplexity:
    def test_sequence_perplexity(self, aap_graph):
        pp = aap_graph.sequence_perplexity('CASSLGIRRT')
        assert isinstance(pp, float)
        assert pp > 0

    def test_repertoire_perplexity(self, aap_graph, aap_sequences):
        pp = aap_graph.repertoire_perplexity(aap_sequences)
        assert pp > 0

    def test_path_entropy_rate(self, aap_graph, aap_sequences):
        rate = aap_graph.path_entropy_rate(aap_sequences)
        assert isinstance(rate, float)
        assert rate >= 0
