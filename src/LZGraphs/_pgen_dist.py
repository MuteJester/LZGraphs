"""PgenDistribution — scipy-like Gaussian mixture for log-PGEN."""

import numpy as np
from . import _clzgraph as _c


class PgenDistribution:
    """Gaussian mixture model of the log-PGEN distribution.

    Created by LZGraph.pgen_distribution(). Provides scipy-like interface.
    """

    def __init__(self, raw_dict):
        w = np.array(raw_dict['weights'], dtype=np.float64)
        w /= w.sum()  # normalize to exactly 1.0 (float precision fix)
        self._weights = w
        self._means = np.array(raw_dict['means'], dtype=np.float64)
        self._stds = np.array(raw_dict['stds'], dtype=np.float64)
        self._global_mean = raw_dict['global_mean']

    @property
    def n_components(self):
        return len(self._weights)

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def means(self):
        return self._means.copy()

    @property
    def stds(self):
        return self._stds.copy()

    @property
    def mean(self):
        """Global mean of the mixture."""
        return self._global_mean

    def pdf(self, x):
        """Probability density function."""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self._weights, self._means, self._stds):
            if sigma > 0:
                z = (x - mu) / sigma
                result += w * np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))
        return float(result) if result.ndim == 0 else result

    def cdf(self, x):
        """Cumulative distribution function."""
        from scipy.special import erfc
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self._weights, self._means, self._stds):
            if sigma > 0:
                z = (x - mu) / (sigma * np.sqrt(2))
                result += w * 0.5 * erfc(-z)
        return float(result) if result.ndim == 0 else result

    def sample(self, n, *, seed=None):
        """Draw n random samples."""
        rng = np.random.default_rng(seed)
        components = rng.choice(self.n_components, size=n, p=self._weights)
        samples = np.empty(n, dtype=np.float64)
        for i in range(n):
            c = components[i]
            samples[i] = rng.normal(self._means[c], self._stds[c])
        return samples

    def __repr__(self):
        return (f"PgenDistribution(components={self.n_components}, "
                f"mean={self.mean:.4f})")
