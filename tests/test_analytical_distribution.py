"""
Tests for LZPgen analytical distribution
=========================================

Tests covering:
1. lzpgen_analytical_distribution() — length-conditional Gaussian mixture + saddlepoint
2. LZPgenDistribution class — pdf, cdf, ppf, rvs, saddlepoint, moments
3. Extended lzpgen_moments() — skewness and kurtosis
4. Consistency between analytical and Monte Carlo approaches
5. All graph types
"""

import numpy as np
import pytest

from LZGraphs import AAPLZGraph, NDPLZGraph, NaiveLZGraph, LZPgenDistribution


# =========================================================================
# lzpgen_analytical_distribution() basic tests
# =========================================================================


class TestAnalyticalDistributionBasic:
    """Basic lzpgen_analytical_distribution() functionality."""

    def test_returns_lzpgen_distribution(self, aap_lzgraph):
        """Returns an LZPgenDistribution object."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert isinstance(dist, LZPgenDistribution)

    def test_has_components(self, aap_lzgraph):
        """Distribution should have at least one component."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert dist.n_components >= 1

    def test_weights_sum_to_one(self, aap_lzgraph):
        """Mixture weights should sum to ~1."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert abs(dist.weights.sum() - 1.0) < 0.05

    def test_weights_non_negative(self, aap_lzgraph):
        """All weights should be non-negative."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert np.all(dist.weights >= 0)

    def test_stds_non_negative(self, aap_lzgraph):
        """All component standard deviations should be non-negative."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert np.all(dist.stds >= 0)

    def test_means_negative(self, aap_lzgraph):
        """Component means should be negative (log-probabilities)."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert np.all(dist.means < 0)

    def test_walk_lengths_present(self, aap_lzgraph):
        """Walk lengths should be provided."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert dist.walk_lengths is not None
        assert len(dist.walk_lengths) == dist.n_components

    def test_walk_lengths_positive(self, aap_lzgraph):
        """Walk lengths should be non-negative integers."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert np.all(dist.walk_lengths >= 0)

    def test_cumulants_present(self, aap_lzgraph):
        """Cumulants dict should have all expected keys."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert 'kappa_1' in dist.cumulants
        assert 'kappa_2' in dist.cumulants
        assert 'kappa_3' in dist.cumulants
        assert 'kappa_4' in dist.cumulants
        assert 'total_mass' in dist.cumulants

    def test_total_mass_near_one(self, aap_lzgraph):
        """Total mass should be close to 1.0."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert abs(dist.cumulants['total_mass'] - 1.0) < 0.05

    def test_deterministic(self, aap_lzgraph):
        """Should produce identical results on repeated calls."""
        d1 = aap_lzgraph.lzpgen_analytical_distribution()
        d2 = aap_lzgraph.lzpgen_analytical_distribution()
        assert d1.mean() == d2.mean()
        assert d1.var() == d2.var()
        np.testing.assert_array_equal(d1.weights, d2.weights)


class TestAnalyticalDistributionAllGraphTypes:
    """lzpgen_analytical_distribution() works on DAG graph types."""

    def test_aap(self, aap_lzgraph):
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert isinstance(dist, LZPgenDistribution)
        assert dist.mean() < 0
        assert dist.n_components >= 1

    def test_ndp(self, ndp_lzgraph):
        dist = ndp_lzgraph.lzpgen_analytical_distribution()
        assert isinstance(dist, LZPgenDistribution)
        assert dist.mean() < 0
        assert dist.n_components >= 1

    def test_naive_raises_for_cycles(self, naive_lzgraph):
        """NaiveLZGraph has cycles; should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="DAG"):
            naive_lzgraph.lzpgen_analytical_distribution()


# =========================================================================
# LZPgenDistribution class tests
# =========================================================================


class TestLZPgenDistributionPDF:
    """PDF tests."""

    def test_pdf_positive(self, aap_lzgraph):
        """PDF should be positive near the mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        x = dist.mean()
        assert dist.pdf(x) > 0

    def test_pdf_array(self, aap_lzgraph):
        """PDF should accept array input."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        x = np.linspace(dist.mean() - 3 * dist.std(),
                        dist.mean() + 3 * dist.std(), 100)
        y = dist.pdf(x)
        assert y.shape == (100,)
        assert np.all(y >= 0)

    def test_pdf_scalar(self, aap_lzgraph):
        """PDF should return scalar for scalar input."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        val = dist.pdf(dist.mean())
        assert isinstance(val, float)

    def test_pdf_tails_decay(self, aap_lzgraph):
        """PDF should decay in the tails."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        center = dist.pdf(m)
        far_left = dist.pdf(m - 6 * s)
        far_right = dist.pdf(m + 6 * s)
        assert far_left < center
        assert far_right < center

    def test_pdf_integrates_near_one(self, aap_lzgraph):
        """PDF should integrate to approximately 1."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 8 * s, m + 8 * s, 5000)
        dx = x[1] - x[0]
        integral = np.sum(dist.pdf(x)) * dx
        assert abs(integral - 1.0) < 0.02, f"PDF integral = {integral}"


class TestLZPgenDistributionCDF:
    """CDF tests."""

    def test_cdf_monotone(self, aap_lzgraph):
        """CDF should be monotonically non-decreasing."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 5 * s, m + 5 * s, 200)
        y = dist.cdf(x)
        assert np.all(np.diff(y) >= -1e-12)

    def test_cdf_range(self, aap_lzgraph):
        """CDF should be in [0, 1]."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 8 * s, m + 8 * s, 200)
        y = dist.cdf(x)
        assert np.all(y >= -1e-12)
        assert np.all(y <= 1 + 1e-12)

    def test_cdf_at_mean_near_half(self, aap_lzgraph):
        """CDF at the mean should be roughly 0.5 (for symmetric-ish dist)."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        cdf_mean = dist.cdf(dist.mean())
        # Allow generous tolerance since the distribution may be skewed
        assert 0.2 < cdf_mean < 0.8

    def test_cdf_scalar(self, aap_lzgraph):
        """CDF should return scalar for scalar input."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        val = dist.cdf(dist.mean())
        assert isinstance(val, float)


class TestLZPgenDistributionPPF:
    """Quantile function tests."""

    def test_ppf_at_half(self, aap_lzgraph):
        """ppf(0.5) should be near the median."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        median = dist.ppf(0.5)
        # Median should be in a reasonable range around the mean
        assert abs(median - dist.mean()) < 3 * dist.std()

    def test_ppf_cdf_roundtrip(self, aap_lzgraph):
        """ppf(cdf(x)) should return approximately x."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        x = dist.mean()
        roundtrip = dist.ppf(dist.cdf(x))
        assert abs(roundtrip - x) < 1e-6

    def test_ppf_monotone(self, aap_lzgraph):
        """ppf should be monotonically increasing."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        xs = dist.ppf(qs)
        assert np.all(np.diff(xs) > 0)


class TestLZPgenDistributionRVS:
    """Random variate sampling tests."""

    def test_rvs_shape(self, aap_lzgraph):
        """rvs should return array of correct size."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        samples = dist.rvs(size=500, seed=42)
        assert samples.shape == (500,)

    def test_rvs_deterministic(self, aap_lzgraph):
        """Same seed gives same results."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        s1 = dist.rvs(size=100, seed=42)
        s2 = dist.rvs(size=100, seed=42)
        np.testing.assert_array_equal(s1, s2)

    def test_rvs_mean_close(self, aap_lzgraph):
        """Sample mean should be close to analytical mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        samples = dist.rvs(size=50000, seed=42)
        assert abs(samples.mean() - dist.mean()) < 0.5


class TestLZPgenDistributionMoments:
    """Moment/shape accessor tests."""

    def test_mean_negative(self, aap_lzgraph):
        """Mean should be negative."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert dist.mean() < 0

    def test_var_non_negative(self, aap_lzgraph):
        """Variance should be non-negative."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert dist.var() >= 0

    def test_std_consistent(self, aap_lzgraph):
        """std should equal sqrt(variance)."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert abs(dist.std() - np.sqrt(dist.var())) < 1e-10

    def test_skewness_finite(self, aap_lzgraph):
        """Skewness should be a finite number."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert np.isfinite(dist.skewness())

    def test_kurtosis_finite(self, aap_lzgraph):
        """Excess kurtosis should be a finite number."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert np.isfinite(dist.kurtosis())


class TestLZPgenDistributionConfidenceInterval:
    """Confidence interval tests."""

    def test_ci_contains_mean(self, aap_lzgraph):
        """95% CI should contain the mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        lo, hi = dist.confidence_interval(0.05)
        assert lo < dist.mean() < hi

    def test_ci_width_reasonable(self, aap_lzgraph):
        """CI should have positive width."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        lo, hi = dist.confidence_interval(0.05)
        assert hi > lo

    def test_ci_narrows_with_alpha(self, aap_lzgraph):
        """Narrower alpha should give wider CI."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        lo1, hi1 = dist.confidence_interval(0.01)
        lo2, hi2 = dist.confidence_interval(0.10)
        assert (hi1 - lo1) > (hi2 - lo2)


# =========================================================================
# Saddlepoint approximation tests
# =========================================================================


class TestSaddlepointApproximation:
    """Tests for saddlepoint PDF/CDF."""

    def test_saddlepoint_pdf_positive(self, aap_lzgraph):
        """Saddlepoint PDF should be positive near the mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        val = dist.saddlepoint_pdf(dist.mean())
        assert val > 0

    def test_saddlepoint_pdf_shape(self, aap_lzgraph):
        """Saddlepoint PDF should work on arrays."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 3 * s, m + 3 * s, 50)
        y = dist.saddlepoint_pdf(x)
        assert y.shape == (50,)
        assert np.all(y >= 0)

    def test_saddlepoint_pdf_matches_mixture(self, aap_lzgraph):
        """Saddlepoint PDF should roughly agree with mixture PDF near the mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 2 * s, m + 2 * s, 20)
        mix_pdf = dist.pdf(x)
        sad_pdf = dist.saddlepoint_pdf(x)
        # Both should have roughly the same peak location and magnitude
        mix_peak = x[np.argmax(mix_pdf)]
        sad_peak = x[np.argmax(sad_pdf)]
        assert abs(mix_peak - sad_peak) < 2 * s

    def test_saddlepoint_cdf_monotone(self, aap_lzgraph):
        """Saddlepoint CDF should be non-decreasing."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 4 * s, m + 4 * s, 100)
        y = dist.saddlepoint_cdf(x)
        # Allow small numerical noise
        assert np.all(np.diff(y) >= -0.01)

    def test_saddlepoint_cdf_range(self, aap_lzgraph):
        """Saddlepoint CDF should be in [0, 1]."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        m = dist.mean()
        s = dist.std()
        x = np.linspace(m - 5 * s, m + 5 * s, 100)
        y = dist.saddlepoint_cdf(x)
        assert np.all(y >= -0.01)
        assert np.all(y <= 1.01)


# =========================================================================
# Consistency: analytical vs Monte Carlo
# =========================================================================


class TestAnalyticalVsMonteCarlo:
    """Analytical distribution should match Monte Carlo estimates."""

    def test_mean_matches_mc(self, aap_lzgraph):
        """Analytical mean should be close to Monte Carlo mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        mc = aap_lzgraph.lzpgen_distribution(n=50_000, seed=42)
        assert abs(dist.mean() - mc.mean()) < 0.5, \
            f"Analytical {dist.mean():.4f} vs MC {mc.mean():.4f}"

    def test_std_matches_mc(self, aap_lzgraph):
        """Analytical std should be close to Monte Carlo std."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        mc = aap_lzgraph.lzpgen_distribution(n=50_000, seed=42)
        mc_std = mc.std()
        assert abs(dist.std() - mc_std) / mc_std < 0.3, \
            f"Analytical {dist.std():.4f} vs MC {mc_std:.4f}"

    def test_mean_matches_moments(self, aap_lzgraph):
        """Analytical mean should match lzpgen_moments() mean."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        moments = aap_lzgraph.lzpgen_moments()
        assert abs(dist.mean() - moments['mean']) < 0.01, \
            f"Analytical {dist.mean():.4f} vs moments {moments['mean']:.4f}"

    def test_std_matches_moments(self, aap_lzgraph):
        """Analytical std should match lzpgen_moments() std."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        moments = aap_lzgraph.lzpgen_moments()
        assert abs(dist.std() - moments['std']) < 0.01, \
            f"Analytical {dist.std():.4f} vs moments {moments['std']:.4f}"

    def test_ndp_mean_matches_moments(self, ndp_lzgraph):
        """NDPLZGraph: analytical mean should match lzpgen_moments()."""
        dist = ndp_lzgraph.lzpgen_analytical_distribution()
        moments = ndp_lzgraph.lzpgen_moments()
        assert abs(dist.mean() - moments['mean']) < 0.01


# =========================================================================
# Extended lzpgen_moments() tests
# =========================================================================


class TestExtendedLZPgenMoments:
    """Tests for the m3/m4 extension of lzpgen_moments()."""

    def test_has_skewness_key(self, aap_lzgraph):
        """Return dict should include skewness."""
        moments = aap_lzgraph.lzpgen_moments()
        assert 'skewness' in moments

    def test_has_kurtosis_key(self, aap_lzgraph):
        """Return dict should include kurtosis."""
        moments = aap_lzgraph.lzpgen_moments()
        assert 'kurtosis' in moments

    def test_skewness_finite(self, aap_lzgraph):
        """Skewness should be finite."""
        moments = aap_lzgraph.lzpgen_moments()
        assert np.isfinite(moments['skewness'])

    def test_kurtosis_finite(self, aap_lzgraph):
        """Kurtosis should be finite."""
        moments = aap_lzgraph.lzpgen_moments()
        assert np.isfinite(moments['kurtosis'])

    def test_skewness_matches_analytical(self, aap_lzgraph):
        """Skewness from moments should match analytical distribution."""
        moments = aap_lzgraph.lzpgen_moments()
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert abs(moments['skewness'] - dist.skewness()) < 0.01

    def test_kurtosis_matches_analytical(self, aap_lzgraph):
        """Kurtosis from moments should match analytical distribution."""
        moments = aap_lzgraph.lzpgen_moments()
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        assert abs(moments['kurtosis'] - dist.kurtosis()) < 0.01

    def test_backward_compat_keys(self, aap_lzgraph):
        """Original keys (mean, variance, std, total_mass) should still be present."""
        moments = aap_lzgraph.lzpgen_moments()
        assert 'mean' in moments
        assert 'variance' in moments
        assert 'std' in moments
        assert 'total_mass' in moments

    def test_ndp_has_higher_moments(self, ndp_lzgraph):
        """NDPLZGraph should also return skewness and kurtosis."""
        moments = ndp_lzgraph.lzpgen_moments()
        assert 'skewness' in moments
        assert 'kurtosis' in moments
        assert np.isfinite(moments['skewness'])
        assert np.isfinite(moments['kurtosis'])


# =========================================================================
# Display / repr tests
# =========================================================================


class TestLZPgenDistributionDisplay:
    """Tests for __repr__ and summary."""

    def test_repr(self, aap_lzgraph):
        """repr should contain class name and key stats."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        r = repr(dist)
        assert 'LZPgenDistribution' in r
        assert 'n_components' in r

    def test_summary(self, aap_lzgraph):
        """summary() should return a multi-line string."""
        dist = aap_lzgraph.lzpgen_analytical_distribution()
        s = dist.summary()
        assert isinstance(s, str)
        assert 'Mean' in s
        assert 'Std' in s
        assert 'Skewness' in s
        assert 'Components' in s
