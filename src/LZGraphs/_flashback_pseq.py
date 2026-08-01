"""Sampling-free p-sequence analytics for :class:`FlashBackGraph`.

The exact core is the Mellin/power-sum transform

    M(q) = sum_s P(s)**q.

For a Markovian FlashBack DAG this transform and its derivatives are forward
dynamic programs over graph edges.  Explicit probability atoms are available
for small supports; large-support distributions are reconstructed on a
deterministic surprisal grid with reported resolution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import comb, erf, exp, floor, lgamma, log, pi, sqrt
from typing import TYPE_CHECKING, Any

import numpy as np

from ._constants import LOG_EPS

if TYPE_CHECKING:
    from ._flashback_graph import FlashBackGraph


def _as_scalar_or_array(original: Any, value: np.ndarray):
    return float(value) if np.ndim(original) == 0 else value


@dataclass(frozen=True)
class PseqAtoms:
    """Exact root-to-sink probability atoms for a small FlashBack graph.

    One array element represents one supported path/sequence. Equal
    probabilities are deliberately not merged.
    """

    surprisal: np.ndarray
    probabilities: np.ndarray
    lengths: np.ndarray

    @property
    def n_sequences(self) -> int:
        return int(self.probabilities.size)

    @property
    def probability_mass(self) -> float:
        return float(np.sum(self.probabilities, dtype=np.longdouble))

    def cdf(self, x, *, measure: str = "generated"):
        """CDF in surprisal space under the generated or counting measure."""
        values = np.asarray(x, dtype=np.float64)
        flat = values.reshape(-1)
        order = np.argsort(self.surprisal)
        support = self.surprisal[order]
        if measure == "generated":
            weights = self.probabilities[order]
        elif measure == "counting":
            weights = np.ones(self.n_sequences, dtype=np.float64)
        else:
            raise ValueError("measure must be 'generated' or 'counting'")
        cumulative = np.cumsum(weights, dtype=np.float64)
        cumulative /= cumulative[-1]
        indices = np.searchsorted(support, flat, side="right") - 1
        out = np.zeros_like(flat)
        mask = indices >= 0
        out[mask] = cumulative[indices[mask]]
        return _as_scalar_or_array(x, out.reshape(values.shape))

    def expected_richness(self, n: int) -> float:
        """Exact expected distinct sequences after ``n`` independent draws."""
        if n < 0:
            raise ValueError("n must be non-negative")
        p = self.probabilities.astype(np.longdouble)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = -np.expm1(np.longdouble(n) * np.log1p(-p))
        terms[p >= 1] = 1.0 if n > 0 else 0.0
        return float(np.sum(terms, dtype=np.longdouble))

    def expected_frequency(self, n: int, r: int) -> float:
        """Exact expected number of sequences observed exactly ``r`` times."""
        if n < 0 or r < 0 or r > n:
            raise ValueError("require n >= 0 and 0 <= r <= n")
        p = self.probabilities.astype(np.longdouble)
        out = np.zeros_like(p)
        interior = (p > 0) & (p < 1)
        log_coefficient = lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)
        out[interior] = np.exp(
            np.longdouble(log_coefficient)
            + np.longdouble(r) * np.log(p[interior])
            + np.longdouble(n - r) * np.log1p(-p[interior])
        )
        if r == n:
            out[p >= 1] = 1
        if r == 0:
            out[p <= 0] = 1
        return float(np.sum(out, dtype=np.longdouble))


@dataclass(frozen=True)
class PseqHistogram:
    """Deterministic surprisal-grid reconstruction of a p-sequence measure.

    ``weights`` are raw masses: they sum to one for ``measure='generated'``
    and to ``D0`` for ``measure='counting'``. The reconstruction uses linear
    grid transport, conserves total mass, and has no Monte Carlo variance.
    """

    surprisal: np.ndarray
    weights: np.ndarray
    measure: str
    q: float
    grid_spacing: float
    max_rounding_error: float
    true_max_surprisal: float
    length: int | None = None
    exact: bool = False

    @property
    def total_mass(self) -> float:
        return float(np.sum(self.weights, dtype=np.longdouble))

    @property
    def normalized_weights(self) -> np.ndarray:
        total = self.total_mass
        return self.weights / total if total > 0 else np.zeros_like(self.weights)

    @property
    def mean(self) -> float:
        w = self.normalized_weights
        return float(np.dot(w, self.surprisal))

    @property
    def variance(self) -> float:
        w = self.normalized_weights
        delta = self.surprisal - self.mean
        return float(np.dot(w, delta * delta))

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.variance, 0.0)))

    def cdf(self, x):
        """Normalized CDF in surprisal space."""
        values = np.asarray(x, dtype=np.float64)
        if self.total_mass <= 0:
            out = np.zeros_like(values)
        else:
            cumulative = np.cumsum(self.weights, dtype=np.float64) / self.total_mass
            out = np.interp(
                values,
                self.surprisal,
                cumulative,
                left=0.0,
                right=1.0,
            )
        return _as_scalar_or_array(x, out)

    def pdf(self, x):
        """Piecewise-linear normalized density of the grid reconstruction."""
        values = np.asarray(x, dtype=np.float64)
        if self.grid_spacing <= 0 or self.total_mass <= 0:
            out = np.zeros_like(values)
        else:
            density = self.weights / (self.total_mass * self.grid_spacing)
            out = np.interp(values, self.surprisal, density, left=0.0, right=0.0)
        return _as_scalar_or_array(x, out)

    def quantile(self, probability):
        """Approximate surprisal quantile of the reconstructed measure."""
        p = np.asarray(probability, dtype=np.float64)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("probability must be between 0 and 1")
        cumulative = np.cumsum(self.normalized_weights)
        out = np.interp(p, cumulative, self.surprisal)
        return _as_scalar_or_array(probability, out)


@dataclass
class PseqSaddlepoint:
    """Smooth saddlepoint approximation to generated-sequence surprisal.

    The cumulant-generating function and all derivatives used here come from
    exact transform DPs. Only the inversion from those exact quantities to a
    smooth PDF/CDF is approximate.
    """

    analysis: FlashBackPseqAnalysis
    exact: bool = False
    method: str = "lugannani_rice_with_discrete_grid_fallback"
    discrete_fallback_paths: int = 64
    fallback_bins: int = 4096
    _fallback: PseqHistogram | None = field(default=None, init=False, repr=False)

    def _fallback_distribution(self) -> PseqHistogram | None:
        if self.analysis.log_mellin(0.0) > log(self.discrete_fallback_paths + 0.5):
            return None
        if self._fallback is None:
            self._fallback = self.analysis.histogram(
                self.fallback_bins, measure="generated"
            )
        return self._fallback

    @staticmethod
    def _normal_pdf(z: float) -> float:
        return exp(-0.5 * z * z) / sqrt(2.0 * pi)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def _saddle(self, x: float) -> tuple[float, tuple[float, ...]]:
        lower = self.analysis.true_min_surprisal
        upper = self.analysis.true_max_surprisal
        if x <= lower:
            return float("-inf"), ()
        if x >= upper:
            return float("inf"), ()

        mean0 = self.analysis._cgf_derivatives(0.0, 2)[1]
        if abs(x - mean0) <= 1e-12 * max(1.0, abs(mean0)):
            return 0.0, self.analysis._cgf_derivatives(0.0, 4)

        if x < mean0:
            hi = 0.0
            lo = -1.0
            while self.analysis._cgf_derivatives(lo, 1)[1] > x and lo > -64:
                lo *= 2.0
        else:
            lo = 0.0
            hi = 1.0
            while self.analysis._cgf_derivatives(hi, 1)[1] < x and hi < 64:
                hi *= 2.0

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            mean = self.analysis._cgf_derivatives(mid, 1)[1]
            if mean < x:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        return t, self.analysis._cgf_derivatives(t, 4)

    def pdf(self, x):
        """Saddlepoint PDF approximation in surprisal space."""
        fallback = self._fallback_distribution()
        if fallback is not None:
            return fallback.pdf(x)
        values = np.asarray(x, dtype=np.float64)
        out = np.empty_like(values)
        for index in np.ndindex(values.shape):
            point = float(values[index])
            t, derivatives = self._saddle(point)
            if not np.isfinite(t) or not derivatives:
                out[index] = 0.0
                continue
            k, _, k2 = derivatives[:3]
            if k2 <= 0:
                out[index] = 0.0
            else:
                out[index] = exp(k - t * point) / sqrt(2.0 * pi * k2)
        return _as_scalar_or_array(x, out)

    def cdf(self, x):
        """Lugannani-Rice CDF approximation in surprisal space."""
        fallback = self._fallback_distribution()
        if fallback is not None:
            return fallback.cdf(x)
        values = np.asarray(x, dtype=np.float64)
        out = np.empty_like(values)
        mean = self.analysis._cgf_derivatives(0.0, 4)[1]
        variance = self.analysis._cgf_derivatives(0.0, 4)[2]
        for index in np.ndindex(values.shape):
            point = float(values[index])
            if point <= self.analysis.true_min_surprisal:
                out[index] = 0.0
                continue
            if point >= self.analysis.true_max_surprisal:
                out[index] = 1.0
                continue
            t, derivatives = self._saddle(point)
            if abs(t) < 1e-8:
                z = (point - mean) / sqrt(max(variance, 1e-300))
                out[index] = self._normal_cdf(z)
                continue
            k, _, k2 = derivatives[:3]
            argument = max(2.0 * (t * point - k), 0.0)
            w = (1.0 if t > 0 else -1.0) * sqrt(argument)
            u = t * sqrt(max(k2, 1e-300))
            if abs(w) < 1e-10 or abs(u) < 1e-10:
                z = (point - mean) / sqrt(max(variance, 1e-300))
                approximation = self._normal_cdf(z)
            else:
                approximation = (
                    self._normal_cdf(w)
                    + self._normal_pdf(w) * (1.0 / w - 1.0 / u)
                )
            out[index] = min(max(approximation, 0.0), 1.0)
        return _as_scalar_or_array(x, out)


class FlashBackPseqAnalysis:
    """Analytical p-sequence spectrum attached to one FlashBackGraph."""

    def __init__(self, graph: FlashBackGraph) -> None:
        if not graph.is_dag:
            raise ValueError("p-sequence analysis requires a FlashBack DAG")
        self.graph = graph
        csr = graph.adjacency_csr()
        self._row = np.asarray(csr["row_offsets"], dtype=np.int64)
        self._col = np.asarray(csr["col_indices"], dtype=np.int64)
        self._weights = np.asarray(csr["weights"], dtype=np.float64)
        self._log_weights = np.log(self._weights)
        self._labels = graph.all_nodes
        self._n = len(self._labels)
        roots = [i for i, label in enumerate(self._labels) if label.startswith("@")]
        if len(roots) != 1:
            raise ValueError(f"expected one FlashBack root, found {len(roots)}")
        self._root = roots[0]
        self._sinks = np.flatnonzero(self._row[1:] == self._row[:-1])
        self._topological_order = self._make_topological_order()
        self._symbol_lengths = np.array(
            [len(label.rsplit("_", 1)[0]) for label in self._labels],
            dtype=np.int64,
        )
        # The root contributes the @ and $ sentinels, which reversal removes.
        self._symbol_lengths[self._root] = max(
            int(self._symbol_lengths[self._root]) - 2, 0
        )
        (
            self._true_min_surprisal,
            self._true_max_surprisal,
            self._max_edges,
        ) = self._path_bounds()

    def _make_topological_order(self) -> np.ndarray:
        indegree = np.zeros(self._n, dtype=np.int64)
        for v in self._col:
            indegree[v] += 1
        queue = [int(i) for i in np.flatnonzero(indegree == 0)]
        order: list[int] = []
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            order.append(u)
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                indegree[v] -= 1
                if indegree[v] == 0:
                    queue.append(v)
        if len(order) != self._n:
            raise ValueError("FlashBack graph contains a cycle")
        return np.asarray(order, dtype=np.int64)

    def _path_bounds(self) -> tuple[float, float, int]:
        min_x = np.full(self._n, np.inf, dtype=np.float64)
        max_x = np.full(self._n, -np.inf, dtype=np.float64)
        max_edges = np.full(self._n, -1, dtype=np.int64)
        min_x[self._root] = 0.0
        max_x[self._root] = 0.0
        max_edges[self._root] = 0
        for u_raw in self._topological_order:
            u = int(u_raw)
            if not np.isfinite(max_x[u]):
                continue
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                min_x[v] = min(min_x[v], min_x[u] - self._log_weights[e])
                max_x[v] = max(max_x[v], max_x[u] - self._log_weights[e])
                max_edges[v] = max(max_edges[v], max_edges[u] + 1)
        reachable_sinks = self._sinks[np.isfinite(max_x[self._sinks])]
        if reachable_sinks.size == 0:
            raise ValueError("FlashBack graph has no reachable sink")
        return (
            float(np.min(min_x[reachable_sinks])),
            float(np.max(max_x[reachable_sinks])),
            int(np.max(max_edges[reachable_sinks])),
        )

    @property
    def true_min_surprisal(self) -> float:
        """Exact minimum root-to-sink surprisal."""
        return self._true_min_surprisal

    @property
    def true_max_surprisal(self) -> float:
        """Exact maximum root-to-sink surprisal."""
        return self._true_max_surprisal

    @staticmethod
    def _transport_jet(
        source: np.ndarray,
        log_probability: np.longdouble,
        q: float,
    ) -> np.ndarray:
        order = source.size - 1
        factor = np.exp(np.longdouble(q) * log_probability)
        out = np.zeros_like(source)
        powers = np.ones(order + 1, dtype=np.longdouble)
        for k in range(1, order + 1):
            powers[k] = powers[k - 1] * log_probability
        for r in range(order + 1):
            value = np.longdouble(0)
            for j in range(r + 1):
                value += comb(r, j) * source[j] * powers[r - j]
            out[r] = factor * value
        return out

    def log_mellin(self, q: float) -> float:
        """Exact-DP ``log(sum_s P(s)**q)`` using log-sum-exp propagation."""
        q = float(q)
        acc = np.full(self._n, -np.inf, dtype=np.float64)
        acc[self._root] = 0.0
        for u_raw in self._topological_order:
            u = int(u_raw)
            if not np.isfinite(acc[u]):
                continue
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                candidate = acc[u] + q * self._log_weights[e]
                acc[v] = np.logaddexp(acc[v], candidate)
        total = -np.inf
        for sink in self._sinks:
            total = np.logaddexp(total, acc[sink])
        return float(total)

    def mellin(self, q: float) -> float:
        """Exact-DP power sum ``M(q)=sum_s P(s)**q``."""
        log_value = self.log_mellin(q)
        if log_value > log(np.finfo(np.float64).max):
            return float("inf")
        return exp(log_value)

    def derivatives(self, q: float, order: int = 4) -> np.ndarray:
        """Return ``[M(q), M'(q), ..., M**(order)(q)]`` by exact DP."""
        if order < 0 or order > 8:
            raise ValueError("order must be between 0 and 8")
        states: list[np.ndarray | None] = [None] * self._n
        root = np.zeros(order + 1, dtype=np.longdouble)
        root[0] = 1
        states[self._root] = root
        total = np.zeros(order + 1, dtype=np.longdouble)
        sink_set = {int(x) for x in self._sinks}
        for u_raw in self._topological_order:
            u = int(u_raw)
            source = states[u]
            if source is None:
                continue
            if u in sink_set:
                total += source
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                transported = self._transport_jet(
                    source, np.longdouble(self._log_weights[e]), q
                )
                if states[v] is None:
                    states[v] = transported
                else:
                    states[v] += transported
            states[u] = None
        return np.asarray(total, dtype=np.float64)

    def cgf(self, t: float) -> float:
        """Generated-surprisal cumulant-generating function ``log M(1-t)``."""
        return self.log_mellin(1.0 - float(t))

    @staticmethod
    def _merge_raw_moments(
        log_mass_a: float,
        moments_a: np.ndarray,
        log_mass_b: float,
        moments_b: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        if not np.isfinite(log_mass_a):
            return log_mass_b, moments_b.copy()
        if not np.isfinite(log_mass_b):
            return log_mass_a, moments_a.copy()
        merged = float(np.logaddexp(log_mass_a, log_mass_b))
        weight_a = exp(log_mass_a - merged)
        weight_b = exp(log_mass_b - merged)
        return merged, weight_a * moments_a + weight_b * moments_b

    def _tilted_log_moments(
        self, q: float, order: int = 4
    ) -> tuple[float, np.ndarray]:
        """Stable normalized raw moments of log-P under path weights P**q."""
        log_mass = np.full(self._n, -np.inf, dtype=np.float64)
        moments = np.zeros((self._n, order + 1), dtype=np.float64)
        log_mass[self._root] = 0.0
        moments[self._root, 0] = 1.0
        sink_log_mass = -np.inf
        sink_moments = np.zeros(order + 1, dtype=np.float64)
        sink_set = {int(x) for x in self._sinks}
        for u_raw in self._topological_order:
            u = int(u_raw)
            if not np.isfinite(log_mass[u]):
                continue
            if u in sink_set:
                sink_log_mass, sink_moments = self._merge_raw_moments(
                    sink_log_mass, sink_moments, log_mass[u], moments[u]
                )
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                lp = float(self._log_weights[e])
                shifted = np.zeros(order + 1, dtype=np.float64)
                shifted[0] = 1.0
                powers = np.ones(order + 1, dtype=np.float64)
                for k in range(1, order + 1):
                    powers[k] = powers[k - 1] * lp
                for r in range(1, order + 1):
                    shifted[r] = sum(
                        comb(r, j) * moments[u, j] * powers[r - j]
                        for j in range(r + 1)
                    )
                candidate_mass = log_mass[u] + q * lp
                log_mass[v], moments[v] = self._merge_raw_moments(
                    log_mass[v], moments[v], candidate_mass, shifted
                )
        return sink_log_mass, sink_moments

    def _cgf_derivatives(self, t: float, order: int = 4) -> tuple[float, ...]:
        """Return K(t) and up to four derivatives for surprisal."""
        log_z, raw_logp = self._tilted_log_moments(1.0 - t, max(order, 1))
        raw_x = np.array(
            [((-1) ** r) * raw_logp[r] for r in range(raw_logp.size)]
        )
        output = [log_z]
        if order >= 1:
            output.append(float(raw_x[1]))
        if order >= 2:
            variance = raw_x[2] - raw_x[1] ** 2
            output.append(float(max(variance, 0.0)))
        if order >= 3:
            third = raw_x[3] - 3 * raw_x[1] * raw_x[2] + 2 * raw_x[1] ** 3
            output.append(float(third))
        if order >= 4:
            central4 = (
                raw_x[4]
                - 4 * raw_x[1] * raw_x[3]
                + 6 * raw_x[1] ** 2 * raw_x[2]
                - 3 * raw_x[1] ** 4
            )
            fourth_cumulant = central4 - 3 * output[2] ** 2
            output.append(float(fourth_cumulant))
        return tuple(output)

    def cumulants(self, order: int = 4) -> dict[str, float]:
        """Exact generated-surprisal cumulants from the transform DP."""
        if order < 1 or order > 4:
            raise ValueError("order must be between 1 and 4")
        values = self._cgf_derivatives(0.0, order)
        names = ["log_normalizer", "kappa1", "kappa2", "kappa3", "kappa4"]
        return {names[i]: values[i] for i in range(order + 1)}

    def saddlepoint(self) -> PseqSaddlepoint:
        """Return a smooth, sampling-free saddlepoint PDF/CDF approximation."""
        return PseqSaddlepoint(self)

    @staticmethod
    def _moments_from_derivatives(derivatives: np.ndarray) -> dict[str, float]:
        z = np.longdouble(derivatives[0])
        if z <= 0:
            return {
                "mass": 0.0,
                "mean": 0.0,
                "variance": 0.0,
                "std": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }
        raw = [
            ((-1) ** r) * np.longdouble(derivatives[r]) / z
            for r in range(min(5, len(derivatives)))
        ]
        mean = raw[1] if len(raw) > 1 else np.longdouble(0)
        variance = raw[2] - mean * mean if len(raw) > 2 else np.longdouble(0)
        variance = max(variance, np.longdouble(0))
        if variance <= np.longdouble(1e-14) * max(
            np.longdouble(1), mean * mean
        ):
            variance = np.longdouble(0)
        std = np.sqrt(variance)
        skewness = np.longdouble(0)
        kurtosis = np.longdouble(0)
        if std > 0 and len(raw) > 3:
            central3 = raw[3] - 3 * mean * raw[2] + 2 * mean**3
            skewness = central3 / std**3
        if variance > 0 and len(raw) > 4:
            central4 = (
                raw[4]
                - 4 * mean * raw[3]
                + 6 * mean * mean * raw[2]
                - 3 * mean**4
            )
            kurtosis = central4 / variance**2 - 3
        return {
            "mass": float(z),
            "mean": float(mean),
            "variance": float(variance),
            "std": float(std),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
        }

    def moments(self, order: int = 4) -> dict[str, float]:
        """Exact generated-surprisal moments (up to fourth order)."""
        if order < 2 or order > 4:
            raise ValueError("order must be between 2 and 4")
        return self._moments_from_derivatives(self.derivatives(1.0, order))

    def length_derivatives(self, q: float, order: int = 4) -> dict[int, np.ndarray]:
        """Exact transform derivatives grouped by reconstructed sequence length."""
        if order < 0 or order > 8:
            raise ValueError("order must be between 0 and 8")
        states: list[dict[int, np.ndarray] | None] = [None] * self._n
        root_jet = np.zeros(order + 1, dtype=np.longdouble)
        root_jet[0] = 1
        states[self._root] = {int(self._symbol_lengths[self._root]): root_jet}
        totals: dict[int, np.ndarray] = {}
        sink_set = {int(x) for x in self._sinks}
        for u_raw in self._topological_order:
            u = int(u_raw)
            source_by_length = states[u]
            if source_by_length is None:
                continue
            if u in sink_set:
                for length, jet in source_by_length.items():
                    totals.setdefault(length, np.zeros(order + 1, dtype=np.longdouble))
                    totals[length] += jet
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                if states[v] is None:
                    states[v] = {}
                destination = states[v]
                assert destination is not None
                increment = int(self._symbol_lengths[v])
                for length, jet in source_by_length.items():
                    new_length = length + increment
                    transported = self._transport_jet(
                        jet, np.longdouble(self._log_weights[e]), q
                    )
                    if new_length in destination:
                        destination[new_length] += transported
                    else:
                        destination[new_length] = transported
            states[u] = None
        return {
            length: np.asarray(totals[length], dtype=np.float64)
            for length in sorted(totals)
        }

    def length_profile(self) -> dict[int, dict[str, float]]:
        """Exact generated mass and surprisal moments for every AA length."""
        derivatives = self.length_derivatives(1.0, 4)
        return {
            length: self._moments_from_derivatives(jet)
            for length, jet in derivatives.items()
        }

    def exact_atoms(self, max_paths: int = 100_000) -> PseqAtoms:
        """Enumerate exact atoms when support size does not exceed ``max_paths``."""
        if max_paths < 1:
            raise ValueError("max_paths must be positive")
        log_count = self.log_mellin(0.0)
        if log_count > log(max_paths + 0.5):
            raise ValueError(
                f"graph has about exp({log_count:.3f}) paths, exceeding "
                f"max_paths={max_paths}"
            )
        surprisals: list[float] = []
        lengths: list[int] = []
        stack = [
            (
                self._root,
                0.0,
                int(self._symbol_lengths[self._root]),
            )
        ]
        sink_set = {int(x) for x in self._sinks}
        while stack:
            u, x, length = stack.pop()
            if u in sink_set:
                surprisals.append(x)
                lengths.append(length)
                if len(surprisals) > max_paths:
                    raise ValueError("path count exceeded max_paths during enumeration")
                continue
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                stack.append(
                    (
                        v,
                        x - float(self._log_weights[e]),
                        length + int(self._symbol_lengths[v]),
                    )
                )
        x_array = np.asarray(surprisals, dtype=np.float64)
        return PseqAtoms(
            surprisal=x_array,
            probabilities=np.exp(-x_array),
            lengths=np.asarray(lengths, dtype=np.int64),
        )

    @staticmethod
    def _shift_grid(source: np.ndarray, shift: float) -> np.ndarray:
        """Linearly shift the final axis right by a non-negative grid offset."""
        width = source.shape[-1]
        lower = int(floor(shift))
        fraction = shift - lower
        out = np.zeros_like(source)
        if lower < width:
            source_width = width - lower
            out[..., lower:] += source[..., :source_width] * (1.0 - fraction)
        upper = lower + 1
        if fraction > 0 and upper < width:
            source_width = width - upper
            out[..., upper:] += source[..., :source_width] * fraction
        return out

    def histogram(
        self,
        bins: int = 2048,
        *,
        measure: str = "generated",
        length: int | None = None,
    ) -> PseqHistogram:
        """Deterministically reconstruct the surprisal spectrum on a grid.

        Linear transport splits mass between adjacent grid points. ``bins``
        must leave enough support for the worst-case accumulation of one-grid
        interpolation error per traversed edge.
        """
        if measure == "generated":
            q = 1.0
        elif measure == "counting":
            q = 0.0
        else:
            raise ValueError("measure must be 'generated' or 'counting'")
        if length is not None and length < 0:
            raise ValueError("length must be non-negative")
        if self._true_max_surprisal == 0:
            mass = self.mellin(q)
            if length is not None:
                length_mass = self.length_derivatives(q, 0)
                mass = float(length_mass.get(length, np.zeros(1))[0])
            weights = np.array([mass], dtype=np.float64)
            return PseqHistogram(
                surprisal=np.array([0.0]),
                weights=weights,
                measure=measure,
                q=q,
                grid_spacing=0.0,
                max_rounding_error=0.0,
                true_max_surprisal=0.0,
                length=length,
            )
        if bins <= self._max_edges + 1:
            raise ValueError(
                f"bins must exceed max path edges + 1 ({self._max_edges + 1})"
            )
        spacing = self._true_max_surprisal / (bins - 1 - self._max_edges)
        grid = np.arange(bins, dtype=np.float64) * spacing
        if length is None:
            states: list[np.ndarray | None] = [None] * self._n
            root = np.zeros(bins, dtype=np.longdouble)
            root[0] = 1
            states[self._root] = root
            total = np.zeros(bins, dtype=np.longdouble)
        else:
            states = [None] * self._n
            root = np.zeros((length + 1, bins), dtype=np.longdouble)
            root_length = int(self._symbol_lengths[self._root])
            if root_length <= length:
                root[root_length, 0] = 1
            states[self._root] = root
            total = np.zeros(bins, dtype=np.longdouble)
        sink_set = {int(x) for x in self._sinks}
        for u_raw in self._topological_order:
            u = int(u_raw)
            source = states[u]
            if source is None:
                continue
            if u in sink_set:
                if length is None:
                    total += source
                else:
                    total += source[length]
            for e in range(int(self._row[u]), int(self._row[u + 1])):
                v = int(self._col[e])
                shifted = self._shift_grid(
                    source, -float(self._log_weights[e]) / spacing
                )
                if length is not None:
                    increment = int(self._symbol_lengths[v])
                    length_shifted = np.zeros_like(shifted)
                    if increment <= length:
                        length_shifted[increment:] = shifted[: length + 1 - increment]
                    shifted = length_shifted
                shifted *= np.exp(np.longdouble(q) * self._log_weights[e])
                if states[v] is None:
                    states[v] = shifted
                else:
                    states[v] += shifted
            states[u] = None
        return PseqHistogram(
            surprisal=grid,
            weights=np.asarray(total, dtype=np.float64),
            measure=measure,
            q=q,
            grid_spacing=float(spacing),
            max_rounding_error=float(self._max_edges * spacing),
            true_max_surprisal=self._true_max_surprisal,
            length=length,
        )

    def position(
        self,
        sequence: str,
        *,
        bins: int = 2048,
        max_exact_paths: int = 100_000,
    ) -> dict[str, float | str]:
        """Position one sequence in both generated and counting spectra."""
        log_p = float(self.graph.pgen(sequence))
        if abs(log_p - LOG_EPS) <= 1e-12:
            raise ValueError("sequence is outside the positive-probability support")
        x = -log_p
        if self.log_mellin(0.0) <= log(max_exact_paths + 0.5):
            atoms = self.exact_atoms(max_exact_paths)
            generated_cdf = float(atoms.cdf(x, measure="generated"))
            counting_cdf = float(atoms.cdf(x, measure="counting"))
            method = "exact_atoms"
            number_more_probable = float(np.count_nonzero(atoms.surprisal <= x))
        else:
            generated = self.histogram(bins, measure="generated")
            counting = self.histogram(bins, measure="counting")
            generated_cdf = float(generated.cdf(x))
            counting_cdf = float(counting.cdf(x))
            method = "deterministic_grid"
            number_more_probable = counting_cdf * self.mellin(0.0)
        d1 = float(self.graph.hill_number(1.0))
        d2 = float(self.graph.hill_number(2.0))
        return {
            "pseq": float(exp(log_p)),
            "log_pseq": log_p,
            "surprisal": x,
            "generated_mass_at_least_as_probable": generated_cdf,
            "fraction_of_sequences_at_least_as_probable": counting_cdf,
            "number_of_sequences_at_least_as_probable": number_more_probable,
            "relative_to_D1": float(exp(log_p) * d1),
            "relative_to_D2": float(exp(log_p) * d2),
            "method": method,
        }

    def _counting_probabilities(
        self, bins: int, max_exact_paths: int
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if self.log_mellin(0.0) <= log(max_exact_paths + 0.5):
            atoms = self.exact_atoms(max_exact_paths)
            return (
                atoms.probabilities.astype(np.longdouble),
                np.ones(atoms.n_sequences, dtype=np.longdouble),
                "exact_atoms",
            )
        histogram = self.histogram(bins, measure="counting")
        return (
            np.exp(-histogram.surprisal.astype(np.longdouble)),
            histogram.weights.astype(np.longdouble),
            "deterministic_grid",
        )

    def expected_richness(
        self,
        n: int,
        *,
        bins: int = 4096,
        max_exact_paths: int = 100_000,
    ) -> dict[str, float | str]:
        """Expected distinct sequences after ``n`` draws."""
        if n < 0:
            raise ValueError("n must be non-negative")
        p, counts, method = self._counting_probabilities(bins, max_exact_paths)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = -np.expm1(np.longdouble(n) * np.log1p(-p))
        terms[p >= 1] = 1.0 if n > 0 else 0.0
        value = np.sum(counts * terms, dtype=np.longdouble)
        return {"expected_richness": float(value), "n": int(n), "method": method}

    def expected_frequency_spectrum(
        self,
        n: int,
        max_count: int = 10,
        *,
        bins: int = 4096,
        max_exact_paths: int = 100_000,
    ) -> dict[str, Any]:
        """Expected number of species observed 0..``max_count`` times."""
        if n < 0 or max_count < 0 or max_count > n:
            raise ValueError("require n >= 0 and 0 <= max_count <= n")
        p, counts, method = self._counting_probabilities(bins, max_exact_paths)
        spectrum = np.zeros(max_count + 1, dtype=np.float64)
        interior = (p > 0) & (p < 1)
        for r in range(max_count + 1):
            values = np.zeros_like(p)
            log_coefficient = lgamma(n + 1) - lgamma(r + 1) - lgamma(n - r + 1)
            values[interior] = np.exp(
                np.longdouble(log_coefficient)
                + np.longdouble(r) * np.log(p[interior])
                + np.longdouble(n - r) * np.log1p(-p[interior])
            )
            if r == n:
                values[p >= 1] = 1
            if r == 0:
                values[p <= 0] = 1
            spectrum[r] = float(np.sum(counts * values, dtype=np.longdouble))
        return {
            "expected_counts": spectrum,
            "n": int(n),
            "max_count": int(max_count),
            "method": method,
        }

    def __repr__(self) -> str:
        return (
            f"FlashBackPseqAnalysis(nodes={self._n}, "
            f"log_D0={self.log_mellin(0.0):.4f})"
        )
