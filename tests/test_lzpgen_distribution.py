"""
Tests for LZPgen distribution methods
======================================

Tests covering:
1. lzpgen_distribution() — Monte Carlo empirical distribution
2. lzpgen_moments()  — exact DAG forward propagation
3. compare_lzpgen_distributions() — distribution comparison metrics
4. Consistency between Monte Carlo and exact moments
5. All graph types
"""

import numpy as np
import pytest

from LZGraphs import AAPLZGraph, NDPLZGraph, NaiveLZGraph, compare_lzpgen_distributions


# =========================================================================
# lzpgen_distribution() tests
# =========================================================================


class TestLZPgenDistributionBasic:
    """Basic lzpgen_distribution() functionality."""

    def test_returns_array(self, aap_lzgraph):
        """Returns a numpy array of the correct length."""
        result = aap_lzgraph.lzpgen_distribution(n=100, seed=42)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert result.dtype == np.float64

    def test_all_finite(self, aap_lzgraph):
        """All returned log-probabilities should be finite."""
        result = aap_lzgraph.lzpgen_distribution(n=500, seed=42)
        assert np.all(np.isfinite(result))

    def test_all_negative(self, aap_lzgraph):
        """Log-probabilities should all be negative."""
        result = aap_lzgraph.lzpgen_distribution(n=500, seed=42)
        assert np.all(result < 0)

    def test_deterministic_with_seed(self, aap_lzgraph):
        """Same seed gives same results."""
        r1 = aap_lzgraph.lzpgen_distribution(n=200, seed=123)
        r2 = aap_lzgraph.lzpgen_distribution(n=200, seed=123)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self, aap_lzgraph):
        """Different seeds give different results."""
        r1 = aap_lzgraph.lzpgen_distribution(n=200, seed=1)
        r2 = aap_lzgraph.lzpgen_distribution(n=200, seed=2)
        assert not np.array_equal(r1, r2)

    def test_zero_n(self, aap_lzgraph):
        """n=0 returns empty array."""
        result = aap_lzgraph.lzpgen_distribution(n=0, seed=42)
        assert len(result) == 0

    def test_consistent_with_walk_log_probability(self, aap_lzgraph):
        """Values should match walk_log_probability on the same walks."""
        walks_and_seqs = aap_lzgraph.simulate(20, seed=42, return_walks=True)
        dist = aap_lzgraph.lzpgen_distribution(n=20, seed=42)

        for i, (walk, seq) in enumerate(walks_and_seqs):
            expected = aap_lzgraph.walk_log_probability(walk, verbose=False)
            assert abs(dist[i] - expected) < 1e-10, \
                f"Walk {i}: dist={dist[i]:.6f} vs walk_log_prob={expected:.6f}"


class TestLZPgenDistributionAllGraphTypes:
    """lzpgen_distribution() works on all graph types."""

    def test_aap(self, aap_lzgraph):
        result = aap_lzgraph.lzpgen_distribution(n=100, seed=42)
        assert len(result) == 100
        assert np.all(np.isfinite(result))
        assert np.all(result < 0)

    def test_ndp(self, ndp_lzgraph):
        result = ndp_lzgraph.lzpgen_distribution(n=100, seed=42)
        assert len(result) == 100
        assert np.all(np.isfinite(result))
        assert np.all(result < 0)

    def test_naive(self, naive_lzgraph):
        result = naive_lzgraph.lzpgen_distribution(n=100, seed=42)
        assert len(result) == 100
        assert np.all(np.isfinite(result))
        assert np.all(result < 0)


class TestLZPgenDistributionStatistics:
    """Distribution should have plausible statistical properties."""

    def test_mean_range(self, aap_lzgraph):
        """Mean log-probability should be in a reasonable range for CDR3 data."""
        result = aap_lzgraph.lzpgen_distribution(n=5000, seed=42)
        mean = result.mean()
        # Typical CDR3 AA LZPgen is roughly -10 to -30
        assert -50 < mean < -5, f"Mean {mean} outside expected range"

    def test_std_positive(self, aap_lzgraph):
        """Distribution should have positive spread."""
        result = aap_lzgraph.lzpgen_distribution(n=5000, seed=42)
        assert result.std() > 0


# =========================================================================
# lzpgen_moments() tests
# =========================================================================


class TestLZPgenMomentsBasic:
    """Basic lzpgen_moments() functionality."""

    def test_returns_dict(self, aap_lzgraph):
        """Returns dict with expected keys."""
        moments = aap_lzgraph.lzpgen_moments()
        assert isinstance(moments, dict)
        assert 'mean' in moments
        assert 'variance' in moments
        assert 'std' in moments
        assert 'total_mass' in moments

    def test_mean_negative(self, aap_lzgraph):
        """Mean log-probability should be negative."""
        moments = aap_lzgraph.lzpgen_moments()
        assert moments['mean'] < 0

    def test_variance_non_negative(self, aap_lzgraph):
        """Variance should be non-negative."""
        moments = aap_lzgraph.lzpgen_moments()
        assert moments['variance'] >= 0

    def test_std_consistent(self, aap_lzgraph):
        """std should equal sqrt(variance)."""
        moments = aap_lzgraph.lzpgen_moments()
        assert abs(moments['std'] - np.sqrt(moments['variance'])) < 1e-10

    def test_total_mass_near_one(self, aap_lzgraph):
        """Total mass should be close to 1.0 for a proper model."""
        moments = aap_lzgraph.lzpgen_moments()
        assert abs(moments['total_mass'] - 1.0) < 0.05, \
            f"Total mass {moments['total_mass']} not near 1.0"

    def test_deterministic(self, aap_lzgraph):
        """Exact moments should be deterministic (no randomness)."""
        m1 = aap_lzgraph.lzpgen_moments()
        m2 = aap_lzgraph.lzpgen_moments()
        assert m1['mean'] == m2['mean']
        assert m1['variance'] == m2['variance']


class TestLZPgenMomentsAllGraphTypes:
    """lzpgen_moments() works on DAG-structured graph types."""

    def test_aap(self, aap_lzgraph):
        moments = aap_lzgraph.lzpgen_moments()
        assert moments['mean'] < 0
        assert moments['total_mass'] > 0.9

    def test_ndp(self, ndp_lzgraph):
        moments = ndp_lzgraph.lzpgen_moments()
        assert moments['mean'] < 0
        assert moments['total_mass'] > 0.9

    def test_naive_raises_for_cycles(self, naive_lzgraph):
        """NaiveLZGraph has cycles; lzpgen_moments() should raise RuntimeError."""
        # NaiveLZGraph nodes lack positional encoding, so cycles are common.
        with pytest.raises(RuntimeError, match="DAG"):
            naive_lzgraph.lzpgen_moments()


class TestLZPgenMomentsMatchesMonteCarlo:
    """Exact moments should match Monte Carlo estimates (within tolerance)."""

    def test_mean_matches(self, aap_lzgraph):
        """Exact mean should be close to Monte Carlo mean."""
        moments = aap_lzgraph.lzpgen_moments()
        dist = aap_lzgraph.lzpgen_distribution(n=50_000, seed=42)
        mc_mean = dist.mean()
        # With 50K samples, MC mean should be within ~0.1 of exact
        assert abs(moments['mean'] - mc_mean) < 0.5, \
            f"Exact mean {moments['mean']:.4f} vs MC mean {mc_mean:.4f}"

    def test_std_matches(self, aap_lzgraph):
        """Exact std should be close to Monte Carlo std."""
        moments = aap_lzgraph.lzpgen_moments()
        dist = aap_lzgraph.lzpgen_distribution(n=50_000, seed=42)
        mc_std = dist.std()
        # Allow generous tolerance for std comparison
        assert abs(moments['std'] - mc_std) / mc_std < 0.2, \
            f"Exact std {moments['std']:.4f} vs MC std {mc_std:.4f}"


# =========================================================================
# compare_lzpgen_distributions() tests
# =========================================================================


class TestCompareLZPgenDistributions:
    """Tests for the comparison function."""

    def test_identical_distributions(self, aap_lzgraph):
        """Comparing a distribution with itself should show near-zero divergence."""
        dist = aap_lzgraph.lzpgen_distribution(n=5000, seed=42)
        metrics = compare_lzpgen_distributions(dist, dist)

        assert metrics['ks_statistic'] == 0.0
        assert metrics['wasserstein'] == 0.0
        assert abs(metrics['mean_diff']) < 1e-10
        assert abs(metrics['std_ratio'] - 1.0) < 1e-10

    def test_same_graph_different_seeds(self, aap_lzgraph):
        """Two samples from same graph should be similar."""
        d1 = aap_lzgraph.lzpgen_distribution(n=5000, seed=1)
        d2 = aap_lzgraph.lzpgen_distribution(n=5000, seed=2)
        metrics = compare_lzpgen_distributions(d1, d2)

        # Should have small KS statistic and high overlap
        assert metrics['ks_statistic'] < 0.1
        assert metrics['overlap_coefficient'] > 0.8
        assert metrics['jsd'] < 0.1

    def test_returns_correct_keys(self, aap_lzgraph):
        """Result dict should have all expected keys."""
        d1 = aap_lzgraph.lzpgen_distribution(n=100, seed=1)
        d2 = aap_lzgraph.lzpgen_distribution(n=100, seed=2)
        metrics = compare_lzpgen_distributions(d1, d2)

        expected_keys = {
            'ks_statistic', 'ks_pvalue', 'wasserstein', 'jsd',
            'mean_diff', 'std_ratio', 'overlap_coefficient',
        }
        assert set(metrics.keys()) == expected_keys

    def test_all_values_finite(self, aap_lzgraph):
        """All returned metrics should be finite numbers."""
        d1 = aap_lzgraph.lzpgen_distribution(n=500, seed=1)
        d2 = aap_lzgraph.lzpgen_distribution(n=500, seed=2)
        metrics = compare_lzpgen_distributions(d1, d2)

        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_different_graph_types(self, aap_lzgraph, ndp_lzgraph):
        """Comparing distributions from different graph types should work."""
        d1 = aap_lzgraph.lzpgen_distribution(n=500, seed=42)
        d2 = ndp_lzgraph.lzpgen_distribution(n=500, seed=42)
        metrics = compare_lzpgen_distributions(d1, d2)

        # Different graph types should produce quite different distributions
        assert abs(metrics['mean_diff']) > 1.0

    def test_jsd_range(self, aap_lzgraph, ndp_lzgraph):
        """JSD should be in [0, 1] (base-2)."""
        d1 = aap_lzgraph.lzpgen_distribution(n=500, seed=42)
        d2 = ndp_lzgraph.lzpgen_distribution(n=500, seed=42)
        metrics = compare_lzpgen_distributions(d1, d2)
        assert 0 <= metrics['jsd'] <= 1.0
