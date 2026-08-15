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
from ._publicness import MASS_TOL as _MASS_TOL
from ._publicness import TAIL_FLOOR as _TAIL_FLOOR
from ._publicness import TAIL_SIGMA as _TAIL_SIGMA
from ._publicness import PublicnessModel

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

    ``weights`` are raw masses. Without a length restriction, they sum to one
    for ``measure='generated'`` and to ``D0`` for ``measure='counting'``;
    with a restriction, they sum to the selected length's mass. The
    reconstruction uses linear grid transport, conserves total mass, and has
    no Monte Carlo variance.
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


@dataclass(frozen=True)
class PseqAttribution:
    """Tilted node and edge usage for a FlashBack p-sequence distribution.

    ``node_probability[u]`` and ``edge_probability[e]`` are the probabilities
    that a path drawn with normalized weight proportional to ``P(s)**q``
    visits that node or edge. Arrays are read-only, zero-copy views of native
    storage owned by this result.
    """

    analysis: FlashBackPseqAnalysis
    q: float
    log_mass: float
    node_probability: np.ndarray
    edge_probability: np.ndarray
    _owner: Any = field(repr=False, compare=False)

    @property
    def edge_sensitivity(self) -> np.ndarray:
        """Gradient ``d log(M(q)) / d log(edge_weight)`` by edge."""
        return self.q * self.edge_probability

    @property
    def edge_surprisal_contribution(self) -> np.ndarray:
        """Expected surprisal contributed by each edge under the tilt."""
        return -self.edge_probability * self.analysis._log_weights

    @property
    def expected_path_edges(self) -> float:
        """Expected number of transitions in a tilted path."""
        return float(np.sum(self.edge_probability, dtype=np.longdouble))

    @property
    def mean_surprisal(self) -> float:
        """Expected full-path surprisal under the tilted distribution."""
        return float(
            np.sum(
                -self.edge_probability.astype(np.longdouble)
                * self.analysis._log_weights.astype(np.longdouble),
                dtype=np.longdouble,
            )
        )

    def top_edges(
        self, k: int = 20, *, by: str = "occupancy"
    ) -> list[dict[str, float | int | str]]:
        """Return the highest-attribution edges without copying full arrays.

        Args:
            k: Maximum number of edges to return.
            by: Ranking score: ``'occupancy'``, ``'sensitivity'`` (absolute
                gradient), or ``'surprisal'`` (expected contribution).
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        if by == "occupancy":
            score = self.edge_probability
        elif by == "sensitivity":
            score = np.abs(self.q * self.edge_probability)
        elif by == "surprisal":
            score = self.edge_surprisal_contribution
        else:
            raise ValueError("by must be 'occupancy', 'sensitivity', or 'surprisal'")
        count = min(int(k), int(score.size))
        if count == 0:
            return []
        if count == score.size:
            indices = np.arange(score.size, dtype=np.int64)
        else:
            indices = np.argpartition(score, -count)[-count:]
        indices = indices[np.lexsort((indices, -score[indices]))]
        sources = np.searchsorted(self.analysis._row, indices, side="right") - 1
        targets = self.analysis._col[indices]
        labels = self.analysis.graph.all_nodes
        output = []
        for edge, source, target in zip(indices, sources, targets):
            edge = int(edge)
            source = int(source)
            target = int(target)
            output.append(
                {
                    "edge": edge,
                    "source": source,
                    "target": target,
                    "source_label": labels[source],
                    "target_label": labels[target],
                    "weight": float(self.analysis._weights[edge]),
                    "occupancy": float(self.edge_probability[edge]),
                    "sensitivity": float(self.q * self.edge_probability[edge]),
                    "surprisal_contribution": float(
                        -self.edge_probability[edge] * self.analysis._log_weights[edge]
                    ),
                }
            )
        return output


@dataclass
class PseqSaddlepoint:
    """Smooth saddlepoint approximation to generated-sequence surprisal.

    The cumulant-generating function and all derivatives used here come from
    exact transform DPs. Only the inversion from those exact quantities to a
    smooth PDF/CDF is approximate.
    """

    analysis: FlashBackPseqAnalysis
    exact: bool = False
    method: str = "native_safeguarded_newton_with_discrete_grid_fallback"
    discrete_fallback_paths: int = 64
    fallback_bins: int = 4096
    _fallback: PseqHistogram | None = field(default=None, init=False, repr=False)

    def _fallback_distribution(self) -> PseqHistogram | None:
        if self.analysis.graph.path_count > self.discrete_fallback_paths:
            return None
        if self._fallback is None:
            self._fallback = self.analysis.histogram(self.fallback_bins, measure="generated")
        return self._fallback

    @staticmethod
    def _normal_pdf(z: float) -> float:
        return exp(-0.5 * z * z) / sqrt(2.0 * pi)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + erf(z / sqrt(2.0)))

    def _saddle(self, x: float) -> tuple[float, tuple[float, ...]]:
        """Solve one saddlepoint, primarily for diagnostics and tests."""
        lower = self.analysis.true_min_surprisal
        upper = self.analysis.true_max_surprisal
        if x <= lower:
            return float("-inf"), ()
        if x >= upper:
            return float("inf"), ()
        result = self._native_evaluate(np.asarray([x], dtype=np.float64))
        t = float(result["saddle"][0])
        return t, self.analysis._cgf_derivatives(t, 4)

    def _native_evaluate(self, values: np.ndarray) -> dict[str, np.ndarray]:
        """Evaluate a flattened batch with the native safeguarded solver."""
        if np.any(~np.isfinite(values)):
            raise ValueError("saddlepoint evaluation points must be finite")
        from . import _clzgraph as _c

        result = _c.fb_pseq_saddlepoint_batch(self.analysis.graph._cap, values.reshape(-1).tolist())
        return {
            name: np.asarray(result[name], dtype=dtype)
            for name, dtype in (
                ("pdf", np.float64),
                ("cdf", np.float64),
                ("saddle", np.float64),
                ("iterations", np.uint32),
            )
        }

    def pdf_cdf(self, x):
        """Return batched saddlepoint PDF and CDF approximations.

        Array inputs are evaluated in one native call. Use this method when
        both quantities are needed so the saddlepoint roots are solved only
        once. Small discrete supports retain the deterministic-grid fallback.
        """
        fallback = self._fallback_distribution()
        if fallback is not None:
            return fallback.pdf(x), fallback.cdf(x)
        values = np.asarray(x, dtype=np.float64)
        result = self._native_evaluate(values)
        pdf = result["pdf"].reshape(values.shape)
        cdf = result["cdf"].reshape(values.shape)
        return _as_scalar_or_array(x, pdf), _as_scalar_or_array(x, cdf)

    def pdf(self, x):
        """Saddlepoint PDF approximation in surprisal space.

        Scalar and array inputs are accepted; arrays use one native batch.
        """
        return self.pdf_cdf(x)[0]

    def cdf(self, x):
        """Lugannani-Rice CDF approximation in surprisal space.

        Scalar and array inputs are accepted; arrays use one native batch.
        """
        return self.pdf_cdf(x)[1]


class FlashBackPseqAnalysis:
    """Analytical p-sequence spectrum attached to one FlashBackGraph."""

    def __init__(self, graph: FlashBackGraph) -> None:
        if not graph.is_dag:
            raise ValueError("p-sequence analysis requires a sentinel-bounded DAG")
        self.graph = graph
        from . import _clzgraph as _c

        structure = _c.fb_pseq_structure(graph._cap)
        self._row = np.frombuffer(structure["row_offsets"], dtype=np.uint32)
        self._col = np.frombuffer(structure["col_indices"], dtype=np.uint32)
        self._weights = np.frombuffer(structure["weights"], dtype=np.float64)
        self._log_weights = np.log(self._weights)
        self._n = self._row.size - 1
        self._root = int(structure["root"])
        self._sinks = np.flatnonzero(np.frombuffer(structure["sink_mask"], dtype=np.uint8))
        self._topological_order = np.frombuffer(structure["topological_order"], dtype=np.uint32)
        self._symbol_lengths = np.frombuffer(structure["symbol_lengths"], dtype=np.uint8)
        self._true_min_surprisal = float(structure["min_surprisal"])
        self._true_max_surprisal = float(structure["max_surprisal"])
        self._max_edges = int(structure["max_edges"])
        self._log_mellin_cache: dict[float, float] = {}

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
        """Return the stable log Mellin transform ``log(sum_s P(s)**q)``.

        The native DAG calculation maintains a log normalizer at every node,
        so it remains finite when the corresponding unnormalized power sum or
        derivative jet would overflow or underflow.
        """
        if not np.isfinite(q):
            raise ValueError("q must be finite")
        q = float(q)
        if q in self._log_mellin_cache:
            return self._log_mellin_cache[q]
        from . import _clzgraph as _c

        value = float(_c.fb_pseq_tilted_moments(self.graph._cap, q, 0)["log_mass"])
        self._log_mellin_cache[q] = value
        return value

    def _log_mellin_python(self, q: float) -> float:
        """Python log-sum-exp oracle used to validate the native DP."""
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
        """Return global Mellin derivatives through ``order``.

        Element ``r`` is ``sum_s P(s)**q * log(P(s))**r`` over every
        generated root-to-sink sequence. The calculation is a sampling-free
        native DAG dynamic program with long-double internal accumulation;
        the returned array is float64. Orders zero through eight are
        supported.
        """
        if order < 0 or order > 8:
            raise ValueError("order must be between 0 and 8")
        from . import _clzgraph as _c

        return np.asarray(
            _c.fb_pseq_derivatives(self.graph._cap, float(q), int(order)),
            dtype=np.float64,
        )

    def _derivatives_python(self, q: float, order: int = 4) -> np.ndarray:
        """Python reference implementation used to validate the native DP."""
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
                transported = self._transport_jet(source, np.longdouble(self._log_weights[e]), q)
                if states[v] is None:
                    states[v] = transported
                else:
                    states[v] += transported
            states[u] = None
        return np.asarray(total, dtype=np.float64)

    def cgf(self, t: float) -> float:
        """Generated-surprisal cumulant-generating function ``log M(1-t)``."""
        return self.log_mellin(1.0 - float(t))

    def tilted_moments(self, q: float, order: int = 4) -> dict[str, Any]:
        """Return normalized log-probability statistics under tilt ``P**q``.

        The result contains ``log_mass = log(sum_s P(s)**q)`` and arrays of
        normalized raw moments, central moments, and cumulants of
        ``log(P(s))`` through ``order``. Central moment zero is one and central
        moment one is zero. Cumulant zero is ``log_mass``. The native
        log-domain calculation remains stable when unnormalized derivative
        jets overflow or underflow. Orders zero through four are supported.
        """
        if order < 0 or order > 4:
            raise ValueError("order must be between 0 and 4")
        if not np.isfinite(q):
            raise ValueError("q must be finite")
        from . import _clzgraph as _c

        result = _c.fb_pseq_tilted_moments(self.graph._cap, float(q), int(order))
        raw = np.asarray(result["raw_moments"], dtype=np.float64)
        central = np.asarray(result["central_moments"], dtype=np.float64)
        cumulants = np.empty(order + 1, dtype=np.float64)
        cumulants[0] = float(result["log_mass"])
        if order >= 1:
            cumulants[1] = raw[1]
        if order >= 2:
            cumulants[2] = central[2]
        if order >= 3:
            cumulants[3] = central[3]
        if order >= 4:
            cumulants[4] = central[4] - 3.0 * central[2] ** 2
        return {
            "q": float(q),
            "log_mass": float(result["log_mass"]),
            "raw_log_probability_moments": raw,
            "central_log_probability_moments": central,
            "log_probability_cumulants": cumulants,
        }

    def attribution(self, q: float = 1.0) -> PseqAttribution:
        """Attribute the tilted p-sequence distribution to nodes and edges.

        A path receives normalized weight proportional to ``P(sequence)**q``.
        The returned node and edge arrays give the probability that a path
        drawn from that tilted distribution visits each object. For edge
        ``e``, ``q * edge_probability[e]`` is the independent-edge
        sensitivity ``d log(M(q)) / d log(weight[e])``.

        The calculation is a sampling-free native log-domain forward-backward
        dynamic program. Public arrays are read-only float64 views; native
        accumulation uses long double.
        """
        if not np.isfinite(q):
            raise ValueError("q must be finite")
        from . import _clzgraph as _c

        result = _c.fb_pseq_attribution(self.graph._cap, float(q))
        node_probability = np.frombuffer(result["node_probability"], dtype=np.float64)
        edge_probability = np.frombuffer(result["edge_probability"], dtype=np.float64)
        return PseqAttribution(
            analysis=self,
            q=float(result["q"]),
            log_mass=float(result["log_mass"]),
            node_probability=node_probability,
            edge_probability=edge_probability,
            _owner=result["owner"],
        )

    def diversity_under_edge_thresholds(self, thresholds: Any) -> dict[str, np.ndarray]:
        """Compute conditioned Hill diversity after thresholding edge weights.

        For each threshold ``tau``, only original graph edges satisfying
        ``weight > tau`` may be traversed. A node stranded by thresholding is
        a dead end, not a new sink: only root-to-sink paths of the unmodified
        graph remain valid sequences. Their original probabilities are
        renormalized by the surviving mass before D1 and D2 are calculated.

        The native fused dynamic program returns natural-log diversities as
        well as their ordinary values, surviving probability mass, retained
        edge counts, and retained edge fractions. It accepts thresholds in any
        order and returns every array in that same order. ``NaN`` thresholds
        are rejected; infinities are useful exact endpoints.

        Args:
            thresholds: One-dimensional sequence of edge-weight cutoffs.

        Returns:
            A dictionary of float64 arrays ``thresholds``, ``D0``, ``D1``,
            ``D2``, ``log_D0``, ``log_D1``, ``log_D2``,
            ``surviving_mass``, ``kept_edges``, and ``edge_fraction``.
        """
        values = np.asarray(thresholds, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("thresholds must be one-dimensional")
        if np.any(np.isnan(values)):
            raise ValueError("thresholds must not contain NaN")
        order = np.argsort(values, kind="stable")
        sorted_values = np.ascontiguousarray(values[order])

        from . import _clzgraph as _c

        native = _c.fb_edge_threshold_diversity(self.graph._cap, sorted_values.tolist())
        inverse = np.empty(order.size, dtype=np.intp)
        inverse[order] = np.arange(order.size, dtype=np.intp)

        def restored(name: str, dtype=np.float64) -> np.ndarray:
            return np.asarray(native[name], dtype=dtype)[inverse]

        log_d0 = restored("log_d0")
        log_d1 = restored("log_d1")
        log_d2 = restored("log_d2")
        kept_edges = restored("kept_edges", np.int64)
        with np.errstate(over="ignore", invalid="ignore"):
            d0 = np.exp(log_d0)
            d1 = np.exp(log_d1)
            d2 = np.exp(log_d2)
        edge_fraction = (
            kept_edges.astype(np.float64) / self.graph.n_edges
            if self.graph.n_edges
            else np.zeros(kept_edges.size, dtype=np.float64)
        )
        return {
            "thresholds": values.copy(),
            "D0": d0,
            "D1": d1,
            "D2": d2,
            "log_D0": log_d0,
            "log_D1": log_d1,
            "log_D2": log_d2,
            "surviving_mass": restored("surviving_mass"),
            "kept_edges": kept_edges,
            "edge_fraction": edge_fraction,
        }

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

    def _tilted_log_moments(self, q: float, order: int = 4) -> tuple[float, np.ndarray]:
        """Return stable normalized raw log-P moments under weights ``P**q``.

        The native calculation uses log-sum-exp mass merging and central
        moments internally. Orders zero through eight are supported.
        """
        if order < 0 or order > 8:
            raise ValueError("order must be between 0 and 8")
        if not np.isfinite(q):
            raise ValueError("q must be finite")
        from . import _clzgraph as _c

        result = _c.fb_pseq_tilted_moments(self.graph._cap, float(q), int(order))
        return (
            float(result["log_mass"]),
            np.asarray(result["raw_moments"], dtype=np.float64),
        )

    def _tilted_log_moments_python(self, q: float, order: int = 4) -> tuple[float, np.ndarray]:
        """Python normalized-moment oracle used to validate the native DP."""
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
                        comb(r, j) * moments[u, j] * powers[r - j] for j in range(r + 1)
                    )
                candidate_mass = log_mass[u] + q * lp
                log_mass[v], moments[v] = self._merge_raw_moments(
                    log_mass[v], moments[v], candidate_mass, shifted
                )
        return sink_log_mass, sink_moments

    def _cgf_derivatives(self, t: float, order: int = 4) -> tuple[float, ...]:
        """Return ``K(t)=log M(1-t)`` and its surprisal derivatives.

        Variance and higher derivatives are formed from native central
        moments, avoiding subtraction of nearly equal large raw moments.
        """
        if order < 0 or order > 4:
            raise ValueError("order must be between 0 and 4")
        from . import _clzgraph as _c

        result = _c.fb_pseq_tilted_moments(self.graph._cap, 1.0 - float(t), max(order, 1))
        log_z = float(result["log_mass"])
        raw_logp = np.asarray(result["raw_moments"], dtype=np.float64)
        central_logp = np.asarray(result["central_moments"], dtype=np.float64)
        output = [log_z]
        if order >= 1:
            output.append(float(-raw_logp[1]))
        if order >= 2:
            output.append(float(max(central_logp[2], 0.0)))
        if order >= 3:
            output.append(float(-central_logp[3]))
        if order >= 4:
            fourth_cumulant = central_logp[4] - 3 * output[2] ** 2
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
            ((-1) ** r) * np.longdouble(derivatives[r]) / z for r in range(min(5, len(derivatives)))
        ]
        mean = raw[1] if len(raw) > 1 else np.longdouble(0)
        variance = raw[2] - mean * mean if len(raw) > 2 else np.longdouble(0)
        variance = max(variance, np.longdouble(0))
        if variance <= np.longdouble(1e-14) * max(np.longdouble(1), mean * mean):
            variance = np.longdouble(0)
        std = np.sqrt(variance)
        skewness = np.longdouble(0)
        kurtosis = np.longdouble(0)
        if std > 0 and len(raw) > 3:
            central3 = raw[3] - 3 * mean * raw[2] + 2 * mean**3
            skewness = central3 / std**3
        if variance > 0 and len(raw) > 4:
            central4 = raw[4] - 4 * mean * raw[3] + 6 * mean * mean * raw[2] - 3 * mean**4
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
        """Return Mellin derivatives grouped by generated AA length.

        For length ``L``, element ``r`` is
        ``sum_{s: len(s)=L} P(s)**q * log(P(s))**r``. Length is the literal
        reconstructed amino-acid string length, excluding ``@``, ``$``, and
        token metadata; it is not walk depth. The native DAG calculation is
        sampling-free and uses long-double internal accumulation before
        returning float64 arrays. Orders zero through eight are supported.
        """
        if order < 0 or order > 8:
            raise ValueError("order must be between 0 and 8")
        if float(q) == 0.0 and order == 0:
            return {
                length: np.asarray([count], dtype=np.float64)
                for length, count in self.graph.path_count_by_length().items()
            }
        from . import _clzgraph as _c

        result = _c.fb_pseq_length_derivatives(self.graph._cap, float(q), int(order))
        return {int(length): np.asarray(jet, dtype=np.float64) for length, jet in result.items()}

    def _length_derivatives_python(self, q: float, order: int = 4) -> dict[int, np.ndarray]:
        """Python reference implementation used to validate the native DP."""
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
                    transported = self._transport_jet(jet, np.longdouble(self._log_weights[e]), q)
                    if new_length in destination:
                        destination[new_length] += transported
                    else:
                        destination[new_length] = transported
            states[u] = None
        return {length: np.asarray(totals[length], dtype=np.float64) for length in sorted(totals)}

    def length_profile(self) -> dict[int, dict[str, float]]:
        """Exact generated mass and surprisal moments for every AA length."""
        derivatives = self.length_derivatives(1.0, 4)
        return {length: self._moments_from_derivatives(jet) for length, jet in derivatives.items()}

    def length_marginals(self) -> dict[int, dict[str, float]]:
        """Return counting richness and generated mass by AA length.

        For each structurally reachable length, ``counting`` is the number of
        distinct root-to-sink paths and ``generated`` is their total
        p-sequence probability. Length is the literal reconstructed amino-acid
        string length, excluding ``@``, ``$``, and token metadata; it is not
        the number of edges in the walk.

        The native fused dynamic program computes the zeroth-order
        length-resolved Mellin transform at ``q=0`` and ``q=1`` in one graph
        traversal. Counting uses float64 accumulation, while generated mass
        uses long-double accumulation before returning Python floats.

        Returns:
            A dictionary keyed by amino-acid length. Each value contains the
            ``counting`` and ``generated`` marginals for that length.
        """
        from . import _clzgraph as _c

        result = _c.fb_pseq_length_marginals(self.graph._cap)
        return {
            int(length): {
                "counting": float(marginals["counting"]),
                "generated": float(marginals["generated"]),
            }
            for length, marginals in result.items()
        }

    def exact_atoms(self, max_paths: int = 100_000) -> PseqAtoms:
        """Enumerate exact atoms when support size does not exceed ``max_paths``."""
        if max_paths < 1:
            raise ValueError("max_paths must be positive")
        log_count = self.log_mellin(0.0)
        if log_count > log(max_paths + 0.5):
            raise ValueError(
                f"graph has about exp({log_count:.3f}) paths, exceeding max_paths={max_paths}"
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
        """Reconstruct the p-sequence spectrum on a deterministic grid.

        ``measure='generated'`` weights each sequence by ``P(s)`` and, when
        unrestricted, has total mass one for a proper graph.
        ``measure='counting'`` gives every supported sequence unit mass and
        therefore totals to richness. Passing ``length`` restricts the raw
        mass to sequences with exactly that reconstructed amino-acid length,
        not to walks with that many edges.

        Each edge's surprisal increment is transported linearly between its
        two adjacent grid points. This conserves mass and first moment and has
        no sampling variance. ``grid_spacing`` is the numerical resolution;
        ``max_rounding_error`` bounds the accumulated placement error along a
        path. Native accumulation uses long double and the public weights are
        float64.

        Args:
            bins: Number of surprisal grid points. Must exceed the maximum
                root-to-sink edge count plus one.
            measure: ``'generated'`` or ``'counting'``.
            length: Optional literal generated amino-acid sequence length.
        """
        if measure == "generated":
            q = 1.0
        elif measure == "counting":
            q = 0.0
        else:
            raise ValueError("measure must be 'generated' or 'counting'")
        if length is not None:
            if not isinstance(length, (int, np.integer)):
                raise TypeError("length must be an integer or None")
            length = int(length)
            if length < 0:
                raise ValueError("length must be non-negative")
        if self._true_max_surprisal == 0:
            mass = self.mellin(q)
            if length is not None:
                length_mass = self.length_derivatives(q, 0)
                mass = float(length_mass.get(length, np.zeros(1))[0])
            return PseqHistogram(
                surprisal=np.array([0.0]),
                weights=np.array([mass], dtype=np.float64),
                measure=measure,
                q=q,
                grid_spacing=0.0,
                max_rounding_error=0.0,
                true_max_surprisal=0.0,
                length=length,
            )
        if bins <= self._max_edges + 1:
            raise ValueError(f"bins must exceed max path edges + 1 ({self._max_edges + 1})")
        from . import _clzgraph as _c

        result = _c.fb_pseq_histogram(
            self.graph._cap, int(bins), q, -1 if length is None else length
        )
        spacing = float(result["spacing"])
        grid = np.arange(bins, dtype=np.float64) * spacing
        return PseqHistogram(
            surprisal=grid,
            weights=np.asarray(result["weights"], dtype=np.float64),
            measure=measure,
            q=q,
            grid_spacing=spacing,
            max_rounding_error=float(result["max_edges"] * spacing),
            true_max_surprisal=float(result["true_max_surprisal"]),
            length=length,
        )

    def histogram_pair(self, bins: int = 2048) -> dict[str, PseqHistogram]:
        """Reconstruct global counting and generated spectra together.

        This fused native calculation is equivalent to calling
        ``histogram(bins, measure="counting")`` and
        ``histogram(bins, measure="generated")`` separately. It traverses the
        graph once, reusing the topological order, grid bounds, and edge
        shifts. Both returned histograms therefore share exactly the same
        surprisal grid and numerical error bound.

        Use this interface when an analysis needs both measures, as spectrum,
        discovery, and publicness workflows commonly do. Exact-length
        restrictions remain available through the single-measure
        :meth:`histogram` method.

        Args:
            bins: Number of surprisal grid points. Must exceed the maximum
                root-to-sink edge count plus one.

        Returns:
            A dictionary with ``counting`` and ``generated``
            :class:`PseqHistogram` values.
        """
        if self._true_max_surprisal == 0:
            grid = np.array([0.0])
            return {
                "counting": PseqHistogram(
                    surprisal=grid.copy(),
                    weights=np.array([self.mellin(0.0)], dtype=np.float64),
                    measure="counting",
                    q=0.0,
                    grid_spacing=0.0,
                    max_rounding_error=0.0,
                    true_max_surprisal=0.0,
                ),
                "generated": PseqHistogram(
                    surprisal=grid.copy(),
                    weights=np.array([self.mellin(1.0)], dtype=np.float64),
                    measure="generated",
                    q=1.0,
                    grid_spacing=0.0,
                    max_rounding_error=0.0,
                    true_max_surprisal=0.0,
                ),
            }
        if bins <= self._max_edges + 1:
            raise ValueError(f"bins must exceed max path edges + 1 ({self._max_edges + 1})")
        from . import _clzgraph as _c

        result = _c.fb_pseq_histogram_pair(self.graph._cap, int(bins))
        spacing = float(result["spacing"])
        true_max = float(result["true_max_surprisal"])
        max_error = float(result["max_edges"] * spacing)
        grid = np.arange(bins, dtype=np.float64) * spacing

        def make_histogram(measure: str, q: float, key: str) -> PseqHistogram:
            return PseqHistogram(
                surprisal=grid.copy(),
                weights=np.asarray(result[key], dtype=np.float64),
                measure=measure,
                q=q,
                grid_spacing=spacing,
                max_rounding_error=max_error,
                true_max_surprisal=true_max,
            )

        return {
            "counting": make_histogram("counting", 0.0, "counting_weights"),
            "generated": make_histogram("generated", 1.0, "generated_weights"),
        }

    def histograms_by_length(
        self,
        bins: int = 2048,
        *,
        measure: str = "generated",
        max_length: int | None = None,
    ) -> dict[int, PseqHistogram]:
        """Reconstruct every length-conditioned spectrum together.

        This native joint dynamic program is equivalent to calling
        ``histogram(bins, measure=measure, length=L)`` independently for each
        reachable amino-acid length ``L`` through ``max_length``. It carries
        literal reconstructed sequence length and surprisal-grid position in
        one topological traversal. Tight node-length grid bounds and reverse
        reachability pruning keep only states that can terminate within the
        requested length range.

        Length excludes ``@``, ``$``, and token metadata; it is not graph walk
        depth. All returned histograms share the global deterministic grid and
        its pathwise rounding-error bound. Internal joint states use float64
        to keep the two-dimensional state affordable, while sink totals use
        long-double accumulation. The numerical agreement target with the
        independent long-double calculations is relative error below
        ``1e-12``.

        Args:
            bins: Number of surprisal grid points. Must exceed the maximum
                root-to-sink edge count plus one.
            measure: ``'generated'`` or ``'counting'``.
            max_length: Largest amino-acid length to return. The default is
                the graph's largest reachable generated length.

        Returns:
            A dictionary mapping every reachable length at or below
            ``max_length`` to its :class:`PseqHistogram`.
        """
        if measure == "generated":
            q = 1.0
        elif measure == "counting":
            q = 0.0
        else:
            raise ValueError("measure must be 'generated' or 'counting'")
        if max_length is None:
            max_length = max(self.length_marginals(), default=0)
        elif not isinstance(max_length, (int, np.integer)):
            raise TypeError("max_length must be an integer or None")
        max_length = int(max_length)
        if max_length < 0:
            raise ValueError("max_length must be non-negative")

        if self._true_max_surprisal == 0:
            marginals = self.length_marginals()
            return {
                length: PseqHistogram(
                    surprisal=np.array([0.0]),
                    weights=np.array([values[measure]], dtype=np.float64),
                    measure=measure,
                    q=q,
                    grid_spacing=0.0,
                    max_rounding_error=0.0,
                    true_max_surprisal=0.0,
                    length=length,
                )
                for length, values in marginals.items()
                if length <= max_length
            }
        if bins <= self._max_edges + 1:
            raise ValueError(f"bins must exceed max path edges + 1 ({self._max_edges + 1})")

        from . import _clzgraph as _c

        result = _c.fb_pseq_histograms_by_length(
            self.graph._cap,
            int(bins),
            q,
            max_length,
        )
        spacing = float(result["spacing"])
        true_max = float(result["true_max_surprisal"])
        max_error = float(result["max_edges"] * spacing)
        grid = np.arange(bins, dtype=np.float64) * spacing
        return {
            int(length): PseqHistogram(
                surprisal=grid.copy(),
                weights=np.asarray(weights, dtype=np.float64),
                measure=measure,
                q=q,
                grid_spacing=spacing,
                max_rounding_error=max_error,
                true_max_surprisal=true_max,
                length=int(length),
            )
            for length, weights in result["weights"].items()
        }

    def _histogram_python(
        self,
        bins: int = 2048,
        *,
        measure: str = "generated",
        length: int | None = None,
    ) -> PseqHistogram:
        """Python reference implementation of deterministic grid transport.

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
            raise ValueError(f"bins must exceed max path edges + 1 ({self._max_edges + 1})")
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
                shifted = self._shift_grid(source, -float(self._log_weights[e]) / spacing)
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
    ) -> tuple[np.ndarray, np.ndarray, str, float, bool]:
        if self.log_mellin(0.0) <= log(max_exact_paths + 0.5):
            atoms = self.exact_atoms(max_exact_paths)
            probabilities = atoms.probabilities.astype(np.longdouble)
            multiplicities = np.ones(atoms.n_sequences, dtype=np.longdouble)
            method = "exact_atoms"
        else:
            histogram = self.histogram(bins, measure="counting")
            probabilities = np.exp(-histogram.surprisal.astype(np.longdouble))
            multiplicities = histogram.weights.astype(np.longdouble)
            method = "deterministic_grid"
        spectrum_mass = np.sum(probabilities * multiplicities, dtype=np.longdouble)
        if not np.isfinite(spectrum_mass) or spectrum_mass <= 0:
            raise RuntimeError("reconstructed p-sequence spectrum has no mass")
        normalized = method == "deterministic_grid"
        if normalized:
            probabilities = probabilities / spectrum_mass
        return (
            probabilities,
            multiplicities,
            method,
            float(spectrum_mass),
            normalized,
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
        p, counts, method, spectrum_mass, normalized = self._counting_probabilities(
            bins, max_exact_paths
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = -np.expm1(np.longdouble(n) * np.log1p(-p))
        terms[p >= 1] = 1.0 if n > 0 else 0.0
        value = np.sum(counts * terms, dtype=np.longdouble)
        return {
            "expected_richness": float(value),
            "n": int(n),
            "method": method,
            "spectrum_mass_before_normalization": spectrum_mass,
            "spectrum_normalized": normalized,
        }

    def discovery_curve(
        self,
        draw_counts: Any,
        *,
        bins: int = 4096,
        max_exact_paths: int = 100_000,
    ) -> dict[str, Any]:
        """Expected richness and novelty over many sampling depths.

        For sequence probabilities ``P(s)`` and every effective draw count
        ``n >= 1``, this evaluates

        ``R(n) = sum_s (1 - (1 - P(s))**n)``

        and

        ``U(n) = sum_s P(s) * (1 - P(s))**(n - 1)``.

        ``R(n)`` is the expected number of distinct sequences seen by draw
        ``n``. ``U(n)`` is the probability that draw ``n`` is novel relative
        to the preceding draws. Non-integer depths are accepted for smooth
        analytical curves.

        The probability spectrum is constructed once and every requested
        depth is then evaluated in one stable native batch. Small supports use
        exact atoms; larger supports use the deterministic counting histogram
        with its usual controllable grid approximation. Grid probabilities
        are rescaled by their reconstructed total mass so they form a proper
        distribution; the unscaled mass is returned as a numerical diagnostic.
        Inputs may have any shape, which every numerical output preserves.

        Args:
            draw_counts: Finite effective sampling depths, all at least one.
            bins: Grid size when deterministic reconstruction is required.
            max_exact_paths: Largest support enumerated as exact atoms.

        Returns:
            A dictionary containing ``draw_counts``, ``expected_richness``,
            ``novelty_probability``, ``method``,
            ``spectrum_mass_before_normalization``, and
            ``spectrum_normalized``.
        """
        values = np.asarray(draw_counts, dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 1):
            raise ValueError("draw_counts must be finite and at least one")
        flat = np.ascontiguousarray(values.reshape(-1))
        if flat.size == 0:
            return {
                "draw_counts": values.copy(),
                "expected_richness": values.copy(),
                "novelty_probability": values.copy(),
                "method": "empty",
                "spectrum_mass_before_normalization": 0.0,
                "spectrum_normalized": False,
            }
        (
            probabilities,
            multiplicities,
            method,
            spectrum_mass,
            normalized,
        ) = self._counting_probabilities(bins, max_exact_paths)
        from . import _clzgraph as _c

        native = _c.pseq_discovery_curve(
            probabilities.tolist(), multiplicities.tolist(), flat.tolist()
        )
        richness = np.asarray(native["expected_richness"], dtype=np.float64).reshape(values.shape)
        novelty = np.asarray(native["novelty_probability"], dtype=np.float64).reshape(values.shape)
        return {
            "draw_counts": values.copy(),
            "expected_richness": _as_scalar_or_array(draw_counts, richness),
            "novelty_probability": _as_scalar_or_array(draw_counts, novelty),
            "method": method,
            "spectrum_mass_before_normalization": float(spectrum_mass),
            "spectrum_normalized": normalized,
        }

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
        p, counts, method, spectrum_mass, normalized = self._counting_probabilities(
            bins, max_exact_paths
        )
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
            "spectrum_mass_before_normalization": spectrum_mass,
            "spectrum_normalized": normalized,
        }

    def publicness_distribution(
        self,
        depths: Any,
        *,
        levels: Any = None,
        depth_bins: int = 64,
        k_fft: int | None = None,
        tail_sigma: float = _TAIL_SIGMA,
        tail_floor: float = _TAIL_FLOOR,
        mass_tol: float = _MASS_TOL,
        bins: int = 4096,
        max_exact_paths: int = 100_000,
    ) -> dict[str, Any]:
        """Predicted sequences per publicness level for a cohort of depths.

        Publicness is repertoire occupancy: in how many of a cohort's
        repertoires a sequence is present. ``depths`` gives one sampling depth
        per repertoire, in distinct sequences contributed. A sequence of model
        probability ``p`` is present in a repertoire of depth ``N`` with
        probability ``1 - (1-p)**N``. Because the depths differ, the occupancy
        count is Poisson-binomial rather than binomial, and the prediction is
        the counting spectrum's multiplicity at each probability weighted by
        that atom's occupancy PMF.

        This differs from :meth:`expected_frequency_spectrum`, which counts how
        many times a sequence is seen inside *one* pool of ``n`` draws.
        Publicness counts how many *separate* repertoires contain it at all.

        ``levels`` selects the binning (per level by default; see
        :meth:`PublicnessModel.expected_counts`). ``bins`` and
        ``max_exact_paths`` are the usual spectrum-resolution controls: exact
        atoms are used when the support is small enough, and the deterministic
        surprisal grid otherwise.

        The occupancy PMF is inverted by DFT, whose roundoff, amplified by
        spectrum multiplicities that reach 1e29, would otherwise rectify into a
        spurious floor across every publicness bin. ``tail_sigma`` and
        ``tail_floor`` truncate each atom to the support its closed-form
        moments say carries mass, which is what removes that floor. Read the
        :mod:`LZGraphs._publicness` module docstring before changing them.

        Returns the dict from :meth:`PublicnessModel.expected_counts`, plus
        the ``method`` used to obtain the spectrum.
        """
        model = PublicnessModel(
            depths,
            depth_bins=depth_bins,
            k_fft=k_fft,
            tail_sigma=tail_sigma,
            tail_floor=tail_floor,
            mass_tol=mass_tol,
        )
        p, counts, method, spectrum_mass, normalized = self._counting_probabilities(
            bins, max_exact_paths
        )
        result = model.expected_counts(
            np.asarray(p, dtype=np.float64),
            np.asarray(counts, dtype=np.float64),
            levels=levels,
        )
        result["method"] = method
        result["spectrum_mass_before_normalization"] = spectrum_mass
        result["spectrum_normalized"] = normalized
        return result

    def __repr__(self) -> str:
        return f"FlashBackPseqAnalysis(nodes={self._n}, log_D0={self.log_mellin(0.0):.4f})"
