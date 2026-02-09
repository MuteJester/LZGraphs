"""LZPgen distribution analysis, comparison, and analytical approximation.

Provides:

- :class:`LZPgenDistribution` — a scipy-like distribution object representing
  the LZPgen distribution as a finite Gaussian mixture (one component per walk
  length) with saddlepoint PDF/CDF from exact cumulants.
- :func:`compare_lzpgen_distributions` — compare two empirical distributions.
"""

import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import brentq

__all__ = [
    'LZPgenDistribution',
    'compare_lzpgen_distributions',
]


class LZPgenDistribution:
    """Analytical LZPgen distribution derived from graph structure.

    Represented as a finite Gaussian mixture (one component per walk length)::

        f(x) = sum_k  weights[k] * Normal(x; means[k], stds[k])

    Each component corresponds to walks of a specific edge-count *k*.
    Parameters are computed by length-stratified forward propagation
    through the DAG in O(|E| * K_max) time — no Monte Carlo sampling.

    Also stores exact cumulants (kappa_1 … kappa_4) enabling a
    saddlepoint approximation of the PDF and CDF.

    Attributes:
        weights (numpy.ndarray): Mixture weights, shape ``(n_components,)``.
        means (numpy.ndarray): Component means, shape ``(n_components,)``.
        stds (numpy.ndarray): Component standard deviations.
        walk_lengths (numpy.ndarray): Walk edge-counts for each component.
        n_components (int): Number of active mixture components.
        cumulants (dict): Exact cumulants ``kappa_1`` … ``kappa_4`` and
            derived ``skewness``, ``kurtosis``.

    Example::

        dist = graph.lzpgen_analytical_distribution()
        x = np.linspace(-35, -10, 500)
        plt.plot(x, dist.pdf(x))
        print(f"95% CI: {dist.confidence_interval(0.05)}")
    """

    def __init__(self, weights, means, stds, walk_lengths=None,
                 cumulants=None):
        self.weights = np.asarray(weights, dtype=np.float64)
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        self.walk_lengths = (np.asarray(walk_lengths, dtype=np.int32)
                             if walk_lengths is not None else None)
        self.n_components = len(self.weights)
        self.cumulants = cumulants or {}

    # -----------------------------------------------------------------
    # Core distribution methods (Gaussian mixture)
    # -----------------------------------------------------------------

    def pdf(self, x):
        """Probability density function (Gaussian mixture)."""
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self.weights, self.means, self.stds):
            if sigma > 1e-15:
                result += w * norm.pdf(x, mu, sigma)
            else:
                # Point-mass component — skip in continuous PDF
                pass
        return float(result[0]) if scalar else result

    def cdf(self, x):
        """Cumulative distribution function (Gaussian mixture)."""
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self.weights, self.means, self.stds):
            if sigma > 1e-15:
                result += w * norm.cdf(x, mu, sigma)
            else:
                result += w * np.where(x >= mu, 1.0, 0.0)
        return float(result[0]) if scalar else result

    def ppf(self, q):
        """Percent-point (quantile) function via numerical CDF inversion."""
        q = np.asarray(q, dtype=np.float64)
        scalar = q.ndim == 0
        q = np.atleast_1d(q)

        mu = self.mean()
        s = self.std()
        lo = mu - 10 * s
        hi = mu + 10 * s

        result = np.empty_like(q)
        for i, qi in enumerate(q):
            qi = float(np.clip(qi, 1e-12, 1 - 1e-12))
            result[i] = brentq(lambda x: self.cdf(x) - qi, lo, hi,
                                xtol=1e-10)

        return float(result[0]) if scalar else result

    def rvs(self, size=1, seed=None):
        """Draw random variates from the Gaussian mixture."""
        rng = np.random.default_rng(seed)
        ks = rng.choice(self.n_components, size=size, p=self.weights)
        return rng.normal(self.means[ks], self.stds[ks])

    # -----------------------------------------------------------------
    # Moment / shape accessors
    # -----------------------------------------------------------------

    def mean(self):
        """Overall mean."""
        return float(np.sum(self.weights * self.means))

    def var(self):
        """Overall variance."""
        m = self.mean()
        return float(np.sum(self.weights * (self.stds ** 2 + self.means ** 2))
                      - m ** 2)

    def std(self):
        """Overall standard deviation."""
        return float(np.sqrt(max(self.var(), 0.0)))

    def skewness(self):
        """Skewness from exact cumulants (kappa_3 / kappa_2^{3/2})."""
        k2 = self.cumulants.get('kappa_2', 0)
        k3 = self.cumulants.get('kappa_3', 0)
        return float(k3 / k2 ** 1.5) if k2 > 0 else 0.0

    def kurtosis(self):
        """Excess kurtosis from exact cumulants (kappa_4 / kappa_2^2)."""
        k2 = self.cumulants.get('kappa_2', 0)
        k4 = self.cumulants.get('kappa_4', 0)
        return float(k4 / k2 ** 2) if k2 > 0 else 0.0

    def confidence_interval(self, alpha=0.05):
        """Return ``(lower, upper)`` for a ``1 - alpha`` confidence interval."""
        return (self.ppf(alpha / 2), self.ppf(1 - alpha / 2))

    # -----------------------------------------------------------------
    # Saddlepoint approximation
    # -----------------------------------------------------------------

    def _cgf_derivs(self, theta):
        """Evaluate CGF K(θ) and its first two derivatives from cumulants."""
        k1 = self.cumulants.get('kappa_1', 0)
        k2 = self.cumulants.get('kappa_2', 1)
        k3 = self.cumulants.get('kappa_3', 0)
        k4 = self.cumulants.get('kappa_4', 0)
        K = k1 * theta + k2 * theta ** 2 / 2 + k3 * theta ** 3 / 6 + k4 * theta ** 4 / 24
        Kp = k1 + k2 * theta + k3 * theta ** 2 / 2 + k4 * theta ** 3 / 6
        Kpp = k2 + k3 * theta + k4 * theta ** 2 / 2
        return K, Kp, Kpp

    def _find_saddlepoint(self, x):
        """Find theta_hat such that K'(theta_hat) = x via Newton's method."""
        k2 = self.cumulants.get('kappa_2', 1)
        k1 = self.cumulants.get('kappa_1', 0)
        theta = (x - k1) / k2 if k2 > 0 else 0.0
        for _ in range(30):
            K, Kp, Kpp = self._cgf_derivs(theta)
            if abs(Kp - x) < 1e-12:
                break
            if abs(Kpp) < 1e-15:
                break
            theta -= (Kp - x) / Kpp
        return theta

    def saddlepoint_pdf(self, x):
        """Saddlepoint approximation to the PDF using exact cumulants.

        Uses the first four cumulants to construct a polynomial CGF
        and solves for the saddlepoint at each evaluation point.
        Always positive (unlike the Edgeworth expansion).

        Args:
            x: Evaluation point(s).

        Returns:
            float or numpy.ndarray: Approximate density value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.empty_like(x)
        for i, xi in enumerate(x):
            theta = self._find_saddlepoint(xi)
            K, Kp, Kpp = self._cgf_derivs(theta)
            if Kpp <= 0:
                result[i] = 0.0
            else:
                result[i] = (1.0 / np.sqrt(2 * np.pi * Kpp)
                             * np.exp(K - theta * xi))
        return float(result[0]) if scalar else result

    def saddlepoint_cdf(self, x):
        """Lugannani-Rice saddlepoint CDF approximation.

        Args:
            x: Evaluation point(s).

        Returns:
            float or numpy.ndarray: Approximate CDF value(s).
        """
        x = np.asarray(x, dtype=np.float64)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)
        result = np.empty_like(x)
        for i, xi in enumerate(x):
            theta = self._find_saddlepoint(xi)
            K, Kp, Kpp = self._cgf_derivs(theta)
            if abs(theta) < 1e-10:
                # At the mean — use continuity correction
                result[i] = 0.5
            else:
                arg = 2 * (theta * xi - K)
                if arg < 0:
                    result[i] = 0.5
                else:
                    r = np.sign(theta) * np.sqrt(arg)
                    s = theta * np.sqrt(max(Kpp, 1e-15))
                    result[i] = (norm.cdf(r)
                                 + norm.pdf(r) * (1.0 / r - 1.0 / s))
            result[i] = np.clip(result[i], 0.0, 1.0)
        return float(result[0]) if scalar else result

    # -----------------------------------------------------------------
    # Display
    # -----------------------------------------------------------------

    def __repr__(self):
        m = self.mean()
        s = self.std()
        return (f"LZPgenDistribution(n_components={self.n_components}, "
                f"mean={m:.4f}, std={s:.4f})")

    def summary(self):
        """Return a human-readable summary string."""
        lines = [
            f"LZPgenDistribution  ({self.n_components} components)",
            f"  Mean     = {self.mean():.4f}",
            f"  Std      = {self.std():.4f}",
            f"  Skewness = {self.skewness():.4f}",
            f"  Kurtosis = {self.kurtosis():.4f}",
            f"  95% CI   = ({self.ppf(0.025):.2f}, {self.ppf(0.975):.2f})",
            "",
            "  Components:",
        ]
        for i in range(self.n_components):
            wl = (f"  k={self.walk_lengths[i]}"
                  if self.walk_lengths is not None else f"  #{i}")
            lines.append(
                f"    {wl}  weight={self.weights[i]:.4f}  "
                f"mean={self.means[i]:.4f}  std={self.stds[i]:.4f}"
            )
        return "\n".join(lines)


def compare_lzpgen_distributions(dist1, dist2, n_bins=200):
    """Compare two empirical LZPgen distributions.

    Takes two arrays of log-probability values (as returned by
    :meth:`~LZGraphs.graphs.lz_graph_base.LZGraphBase.lzpgen_distribution`)
    and computes a suite of comparison metrics.

    Args:
        dist1 (array-like): Log-probability values from the first graph.
        dist2 (array-like): Log-probability values from the second graph.
        n_bins (int): Number of bins for histogram-based metrics (JSD,
            overlap coefficient).  Default 200.

    Returns:
        dict: Comparison metrics:

            - **ks_statistic** (*float*): Kolmogorov-Smirnov test statistic.
            - **ks_pvalue** (*float*): KS test p-value.
            - **wasserstein** (*float*): Wasserstein-1 (earth mover's) distance.
            - **jsd** (*float*): Jensen-Shannon divergence (base 2) of binned
              distributions.
            - **mean_diff** (*float*): ``mean(dist1) - mean(dist2)``.
            - **std_ratio** (*float*): ``std(dist1) / std(dist2)``.
            - **overlap_coefficient** (*float*): Area of overlap between the
              two density estimates (0 = disjoint, 1 = identical).

    Example::

        d1 = graph_a.lzpgen_distribution(n=50_000, seed=42)
        d2 = graph_b.lzpgen_distribution(n=50_000, seed=42)
        metrics = compare_lzpgen_distributions(d1, d2)
        print(f"JSD = {metrics['jsd']:.4f}")
        print(f"KS  = {metrics['ks_statistic']:.4f} (p={metrics['ks_pvalue']:.2e})")
    """
    d1 = np.asarray(dist1, dtype=np.float64)
    d2 = np.asarray(dist2, dtype=np.float64)

    # Remove non-finite values
    d1 = d1[np.isfinite(d1)]
    d2 = d2[np.isfinite(d2)]

    if len(d1) == 0 or len(d2) == 0:
        raise ValueError("Both distributions must contain finite values")

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(d1, d2)

    # Wasserstein distance
    wass = stats.wasserstein_distance(d1, d2)

    # Jensen-Shannon divergence via binned histograms
    lo = min(d1.min(), d2.min())
    hi = max(d1.max(), d2.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    h1, _ = np.histogram(d1, bins=bins, density=True)
    h2, _ = np.histogram(d2, bins=bins, density=True)

    # Normalize to proper probability vectors
    eps = 1e-12
    p1 = h1 + eps
    p2 = h2 + eps
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    m = 0.5 * (p1 + p2)
    jsd = 0.5 * (stats.entropy(p1, m, base=2) + stats.entropy(p2, m, base=2))

    # Overlap coefficient
    bin_width = (hi - lo) / n_bins
    overlap = np.sum(np.minimum(h1, h2)) * bin_width

    return {
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_pval),
        'wasserstein': float(wass),
        'jsd': float(jsd),
        'mean_diff': float(d1.mean() - d2.mean()),
        'std_ratio': float(d1.std() / d2.std()) if d2.std() > 0 else float('inf'),
        'overlap_coefficient': float(min(overlap, 1.0)),
    }
