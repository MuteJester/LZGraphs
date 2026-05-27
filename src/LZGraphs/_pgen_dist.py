"""PgenDistribution — scipy-like Gaussian mixture for the forward-DP log-PGEN approximation."""
from __future__ import annotations

from typing import Any

import numpy as np


class PgenDistribution:
    """Gaussian mixture model of the forward-DP log-PGEN approximation.

    Created by LZGraph.pgen_distribution(). Provides scipy-like interface.
    """

    def __init__(self, raw_dict: dict[str, Any]) -> None:
        w = np.array(raw_dict['weights'], dtype=np.float64)
        w /= w.sum()  # normalize to exactly 1.0 (float precision fix)
        self._weights = w
        self._means = np.array(raw_dict['means'], dtype=np.float64)
        self._stds = np.array(raw_dict['stds'], dtype=np.float64)
        self._global_mean = raw_dict['global_mean']

    @property
    def n_components(self) -> int:
        return len(self._weights)

    @property
    def weights(self) -> np.ndarray:
        return self._weights.copy()

    @property
    def means(self) -> np.ndarray:
        return self._means.copy()

    @property
    def stds(self) -> np.ndarray:
        return self._stds.copy()

    @property
    def mean(self) -> float:
        """Global mean of the mixture."""
        return self._global_mean

    def pdf(self, x) -> float | np.ndarray:
        """Probability density function."""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self._weights, self._means, self._stds):
            if sigma > 0:
                z = (x - mu) / sigma
                result += w * np.exp(-0.5 * z * z) / (sigma * np.sqrt(2 * np.pi))
        return float(result) if result.ndim == 0 else result

    def cdf(self, x) -> float | np.ndarray:
        """Cumulative distribution function."""
        from scipy.special import erfc
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)
        for w, mu, sigma in zip(self._weights, self._means, self._stds):
            if sigma > 0:
                z = (x - mu) / (sigma * np.sqrt(2))
                result += w * 0.5 * erfc(-z)
        return float(result) if result.ndim == 0 else result

    def sample(self, n: int, *, seed: int | None = None) -> np.ndarray:
        """Draw n random samples from the mixture (vectorized)."""
        rng = np.random.default_rng(seed)
        components = rng.choice(self.n_components, size=n, p=self._weights)
        # Reparameterization: X ~ N(μ_c, σ_c)  ≡  μ_c + σ_c * Z, Z ~ N(0, 1).
        # One vectorized standard-normal draw + fancy indexing — O(n) numpy
        # rather than O(n) Python iterations.
        return self._means[components] + self._stds[components] * rng.standard_normal(n)

    def __repr__(self) -> str:
        return (f"PgenDistribution(components={self.n_components}, "
                f"mean={self.mean:.4f})")
