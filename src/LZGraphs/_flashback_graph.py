"""FlashBackGraph — Markovian graph from FlashBack decomposition."""
from __future__ import annotations

import os
from typing import Any, Iterable, overload

import numpy as np

from . import _clzgraph as _c
from ._graph_common import _GraphCommonMixin
from ._simulation_result import SimulationResult
from ._types import (
    DiversityProfile,
    DynamicRange,
    FbasResult,
    PgenDiagnostics,
    PgenMoments,
    Summary,
)


class FlashBackGraph(_GraphCommonMixin):
    """FlashBack decomposition graph for sequence repertoire analysis.

    Builds a Markovian DAG where nodes are FlashBack tokens and edges
    are transitions between consecutive tokens. The FlashBack algorithm
    recursively peels matching character runs from both ends of the
    sentinel-wrapped sequence.

    Since the graph is Markovian, all analytics (path count, diversity,
    entropy, Hill numbers) are computed exactly via forward DP — no
    Monte Carlo approximation needed.

    Args:
        sequences: List of strings.
        abundances: Per-sequence counts. None = all 1.
        smoothing: Laplace smoothing alpha for edge weights.
    """
    # ── Lazy-cache slots (populated by first access) ──
    _all_edges_cache = None
    _all_nodes_cache = None
    _dyn_range_cache = None
    _edges_cache = None
    _eff_div_cache = None
    _length_dist_cache = None
    _nodes_cache = None
    _path_count_cache = None


    def __init__(
        self,
        sequences: Iterable[str],
        *,
        abundances: Iterable[int] | None = None,
        smoothing: float = 0.0,
    ) -> None:
        if not sequences:
            raise ValueError("sequences must be a non-empty list")
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        self._cap = _c.fb_graph_build(
            seqs,
            list(abundances) if abundances is not None else None,
            smoothing,
        )
        self._info = _c.graph_info(self._cap)

    @classmethod
    def _from_capsule(cls, capsule: Any) -> "FlashBackGraph":
        obj = object.__new__(cls)
        obj._cap = capsule
        obj._info = _c.graph_info(capsule)
        return obj

    @classmethod
    def from_file(cls, path: str | os.PathLike, *, smoothing: float = 0.0) -> "FlashBackGraph":
        """Build from a plain text file (streaming, constant memory).

        Supported: one sequence per line, or sequence<TAB>abundance.
        """
        path = os.fspath(path)
        if not path:
            raise ValueError("path must be non-empty")
        return cls._from_capsule(_c.fb_graph_build_file(path, smoothing))

    # ── Dunder ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"FlashBackGraph(nodes={self.n_nodes}, edges={self.n_edges})"

    # ── Basic properties ────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph (FlashBack tokens, including sentinel @ and $ nodes)."""
        return self._info['n_nodes']

    @property
    def n_edges(self) -> int:
        """Number of directed edges in the graph."""
        return self._info['n_edges']

    @property
    def variant(self) -> str:
        """Graph encoding variant; always ``'flashback'`` for this class."""
        return 'flashback'

    @property
    def is_dag(self) -> bool:
        """Whether the graph is a directed acyclic graph (always True for FlashBack)."""
        return self._info['is_dag']

    @property
    def path_count(self) -> float:
        """Exact number of distinct walks (via DP)."""
        if self._path_count_cache is None:
            self._path_count_cache = float(_c.fb_path_count(self._cap))
        return self._path_count_cache

    # ── Structural ──────────────────────────────────────────

    @property
    def n_sequences(self) -> int:
        """Total number of sequences used to build the graph (weighted by abundance)."""
        return sum(self.length_distribution.values())

    @property
    def length_distribution(self) -> dict[int, int]:
        """Sequence length distribution: {length: count}."""
        if self._length_dist_cache is None:
            self._length_dist_cache = _c.graph_length_distribution(self._cap)
        return dict(self._length_dist_cache)

    @property
    def nodes(self) -> list[str]:
        """List of node label strings (excluding @ and $ sentinels)."""
        if self._nodes_cache is None:
            raw = _c.graph_nodes(self._cap)
            self._all_nodes_cache = raw
            self._nodes_cache = [n for n in raw
                                 if not n.startswith('@') and '$' not in n]
        return list(self._nodes_cache)

    @property
    def all_nodes(self) -> list[str]:
        """List of all node label strings, including @ and $ sentinels."""
        if self._all_nodes_cache is None:
            self._all_nodes_cache = _c.graph_nodes(self._cap)
        return list(self._all_nodes_cache)

    @property
    def edges(self) -> list[tuple[str, str, float, int]]:
        """List of (source, target, weight, count) tuples for all non-sentinel edges."""
        if self._edges_cache is None:
            raw = _c.graph_edges(self._cap)
            self._all_edges_cache = raw
            self._edges_cache = [
                e for e in raw
                if not e[0].startswith('@') and '$' not in e[0]
                and not e[1].startswith('@') and '$' not in e[1]
            ]
        return list(self._edges_cache)

    @property
    def all_edges(self) -> list[tuple[str, str, float, int]]:
        """List of all (source, target, weight, count) tuples, including sentinels."""
        if self._all_edges_cache is None:
            self._all_edges_cache = _c.graph_edges(self._cap)
        return list(self._all_edges_cache)

    @property
    def n_initial(self) -> int:
        """Number of initial states (nodes reachable from the @ root)."""
        return self._get_summary()['n_initial']

    @property
    def n_terminal(self) -> int:
        """Number of terminal ($-suffixed) sink nodes."""
        return self._get_summary()['n_terminal']

    @property
    def max_out_degree(self) -> int:
        """Maximum out-degree of any node."""
        return self._get_summary()['max_out_degree']

    @property
    def max_in_degree(self) -> int:
        """Maximum in-degree of any node."""
        return self._get_summary()['max_in_degree']

    @property
    def density(self) -> float:
        """Graph density: ``n_edges / (n_nodes * (n_nodes - 1))``."""
        n = self.n_nodes
        return self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    @property
    def out_degrees(self) -> np.ndarray:
        """Out-degree of each node as a numpy array (indexed by node ID)."""
        return np.array(self._get_degrees()['out_degrees'], dtype=np.uint32)

    @property
    def in_degrees(self) -> np.ndarray:
        """In-degree of each node as a numpy array (indexed by node ID)."""
        return np.array(self._get_degrees()['in_degrees'], dtype=np.uint32)

    # ── Adjacency ───────────────────────────────────────────

    def adjacency_csr(self) -> dict[str, np.ndarray]:
        """CSR (Compressed Sparse Row) adjacency representation."""
        csr = self._get_csr()
        return {
            'row_offsets': csr['row_offsets'].copy(),
            'col_indices': csr['col_indices'].copy(),
            'weights': csr['weights'].copy(),
            'counts': csr['counts'].copy(),
        }

    # ── Simulation ──────────────────────────────────────────

    def simulate(self, n: int, *, seed: int | None = None) -> SimulationResult:
        """Generate n sequences by Markov random walk.

        Returns SimulationResult (iterable of sequences with .log_probs).
        """
        seed_val = seed if seed is not None else -1
        seqs, lps, nts = _c.fb_simulate(self._cap, n, seed=seed_val)
        return SimulationResult(seqs, lps, nts)

    def top_k_sequences(
        self, k: int = 100, *, most_probable: bool = True
    ) -> SimulationResult:
        """Find the K most (or least) probable sequences in the graph.

        Uses exact forward DP on the DAG's topological order — no simulation
        or approximation. Returns a SimulationResult with sequences sorted
        by probability (descending if most_probable, ascending otherwise).

        Args:
            k: Number of sequences to return.
            most_probable: If True, return highest-probability sequences.
                          If False, return lowest-probability sequences.

        Returns:
            SimulationResult with .sequences, .log_probs attributes.
        """
        seqs, lps = _c.fb_top_k_walks(self._cap, k=k, most_probable=most_probable)
        nts = [0] * len(seqs)  # token counts not tracked in top-k
        return SimulationResult(seqs, lps, nts)

    # ── Probability ─────────────────────────────────────────

    @overload
    def pgen(self, sequence: str, *, log: bool = True) -> float: ...
    @overload
    def pgen(self, sequence: list[str], *, log: bool = True) -> np.ndarray: ...

    def pgen(self, sequence, *, log=True):
        """Exact probability of sequence(s) under the FlashBack model.

        Args:
            sequence: A single string, or a list of strings.
            log: If True, return log-probability.

        Returns:
            ``float`` for a single sequence, ``np.ndarray`` for a list.
        """
        raw = _c.fb_pgen(self._cap, sequence)
        if isinstance(raw, float):
            return raw if log else np.exp(raw)
        arr = np.array(raw, dtype=np.float64)
        return arr if log else np.exp(arr)

    def flashback_fbas(self, sequence: str | list[str]) -> FbasResult | list[FbasResult]:
        """FlashBack Anomaly Score for sequence(s) via the C path.

        Returns a dict with keys ``fbas``, ``log_pgen``, ``worst_excess``,
        ``n_missing_tokens``, ``n_missing_edges`` for a single sequence,
        or a list of such dicts for a list input.

        FBAS = max(depth_weighted_excess) + mean(depth_weighted_excess) where
        excess at each transition is ``max(0, -log(edge_weight) - entropy(src))``
        and depth weight linearly decays from 1.0 at depth 0 to 0.4 at the
        final transition. Missing tokens and edges contribute fixed penalties
        (50.0 and 30.0 respectively).

        ~50-100x faster than the equivalent Python ``diagnose_sequence`` loop;
        results match to machine precision.
        """
        return _c.fb_fbas(self._cap, sequence)

    # ── Exact analytics ────────────────────────────────────

    def diversity_profile(self) -> DiversityProfile:
        """Full Shannon diversity breakdown (exact). Cached per instance."""
        if self._eff_div_cache is None:
            self._eff_div_cache = _c.fb_effective_diversity(self._cap)
        return self._eff_div_cache

    def effective_diversity(self) -> float:
        """Exact effective diversity exp(H) via forward DP."""
        return self.diversity_profile()['effective_diversity']

    def hill_number(self, alpha: float) -> float:
        """Exact Hill number D(alpha) via forward DP."""
        return _c.fb_hill_number(self._cap, float(alpha))

    def hill_numbers(self, orders: Iterable[float]) -> np.ndarray:
        """Exact Hill numbers for multiple orders."""
        result = _c.fb_hill_numbers(self._cap, [float(o) for o in orders])
        return np.array(result, dtype=np.float64)

    def hill_curve(self, orders: Iterable[float] | None = None) -> dict[str, np.ndarray]:
        """Hill diversity curve (exact)."""
        if orders is None:
            orders = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
        o_list = [float(o) for o in orders]
        result = _c.fb_hill_numbers(self._cap, o_list)
        return {
            'orders': np.array(o_list, dtype=np.float64),
            'values': np.array(result, dtype=np.float64),
        }

    def power_sum(self, alpha: float) -> float:
        """Exact power sum M(alpha) via forward DP."""
        return _c.fb_power_sum(self._cap, float(alpha))

    def pgen_diagnostics(self, atol: float = 1e-6) -> PgenDiagnostics:
        """Exact absorbed vs leaked mass (via DP)."""
        return _c.fb_pgen_diagnostics(self._cap, atol)

    def pgen_dynamic_range_detail(self) -> DynamicRange:
        """Full dynamic range breakdown (exact). Cached per instance."""
        if self._dyn_range_cache is None:
            self._dyn_range_cache = _c.fb_dynamic_range(self._cap)
        return self._dyn_range_cache

    def pgen_dynamic_range(self) -> float:
        """Exact dynamic range in orders of magnitude."""
        return self.pgen_dynamic_range_detail()['dynamic_range_orders']

    # ── PGEN Distribution ──────────────────────────────────

    def pgen_moments(self) -> PgenMoments:
        """Moments of the forward-DP log-PGEN distribution."""
        return _c.pgen_moments(self._cap)

    def pgen_distribution(self):
        """Analytical Gaussian mixture approximation of log-PGEN."""
        from ._pgen_dist import PgenDistribution
        raw = _c.pgen_analytical(self._cap)
        return PgenDistribution(raw)

    # ── Graph operations ───────────────────────────────────

    def union(self, other: "FlashBackGraph") -> "FlashBackGraph":
        """Union: sum edge counts from both graphs."""
        cap = _c.graph_union(self._cap, other._cap)
        return FlashBackGraph._from_capsule(cap)

    def intersection(self, other: "FlashBackGraph") -> "FlashBackGraph":
        """Intersection: keep shared edges, min counts."""
        cap = _c.graph_intersection(self._cap, other._cap)
        return FlashBackGraph._from_capsule(cap)

    def difference(self, other: "FlashBackGraph") -> "FlashBackGraph":
        """Difference: subtract ``other``'s edge counts."""
        cap = _c.graph_difference(self._cap, other._cap)
        return FlashBackGraph._from_capsule(cap)

    def weighted_merge(
        self, other: "FlashBackGraph", alpha: float = 1.0, beta: float = 1.0
    ) -> "FlashBackGraph":
        """Weighted merge: ``alpha * self + beta * other``."""
        cap = _c.weighted_merge(self._cap, other._cap, alpha, beta)
        return FlashBackGraph._from_capsule(cap)

    def posterior(
        self,
        sequences: Iterable[str],
        *,
        abundances: Iterable[int] | None = None,
        kappa: float = 1.0,
    ) -> "FlashBackGraph":
        """Bayesian posterior graph given new observed sequences.

        Keeps the prior's topology and updates each edge weight via the
        Dirichlet-Multinomial rule:

            w_post(u->v) = (kappa * w_prior(u->v) + c_ind(u->v))
                           / (kappa + n_ind(u))

        where c_ind and n_ind are the individual's edge count and total
        outgoing count at u, derived from FlashBack decomposition of
        ``sequences``. kappa=0 -> pure individual, kappa->inf -> pure prior.
        Sequences inducing edges not present in the prior are ignored.
        """
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        if kappa < 0:
            raise ValueError(f"kappa must be >= 0, got {kappa}")
        cap = _c.fb_posterior(self._cap, seqs, abundances=abs_list, kappa=float(kappa))
        return FlashBackGraph._from_capsule(cap)

    def without(
        self,
        sequences: Iterable[str],
        *,
        abundances: Iterable[int] | None = None,
    ) -> "FlashBackGraph":
        """Return a new graph with the contribution of ``sequences`` removed.

        For each sequence (with abundance ``a``, defaulting to 1) the walk
        through the graph's FlashBack decomposition is traced and ``a`` is
        subtracted from every edge's raw count on that walk. Subtraction is
        clamped at zero. Edges whose count reaches zero are physically
        pruned; isolated nodes are retained for node-index stability and are
        filtered by the Python accessors. Weights are renormalised per
        source node after pruning.

        Use case: leave-donor-out foundation construction from an existing
        graph, in seconds instead of rebuilding from source data.

        Args:
            sequences: list of strings to remove.
            abundances: per-sequence counts. None defaults to 1 each.
        """
        if isinstance(sequences, str):
            raise TypeError("sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != sequences length {len(seqs)}"
            )
        cap = _c.fb_subtract(self._cap, seqs, abundances=abs_list)
        return FlashBackGraph._from_capsule(cap)

    # ── Features ───────────────────────────────────────────

    def feature_stats(self) -> np.ndarray:
        """15-element graph statistics vector for ML pipelines."""
        return np.array(_c.feature_stats(self._cap), dtype=np.float64)

    def feature_mass_profile(self, max_pos: int = 30) -> np.ndarray:
        """Position-based mass distribution profile."""
        return np.array(_c.feature_mass_profile(self._cap, max_pos), dtype=np.float64)

    # ── IO ─────────────────────────────────────────────────

    def save(self, path: str | os.PathLike) -> None:
        """Save to LZG binary format (.lzg). Accepts ``str`` or ``os.PathLike``."""
        _c.save(self._cap, os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike) -> "FlashBackGraph":
        """Load from LZG binary format. Accepts ``str`` or ``os.PathLike``."""
        cap = _c.load(os.fspath(path))
        _c.fb_fix_special_nodes(cap)
        return cls._from_capsule(cap)

    def summary(self) -> Summary:
        """Structural summary dict."""
        return _c.summary(self._cap)


class FlashBackStream:
    """Streaming builder for FlashBackGraph.

    Use when sequences arrive incrementally (e.g., from a generator,
    network stream, or open-ended simulator) and you want to monitor
    running node/edge counts before deciding to stop.

    Internally this calls the same per-sequence accumulator the batch
    and file builders use, so the resulting graph is byte-identical to
    the one you'd get from feeding all sequences at once to
    ``FlashBackGraph(sequences)``.

    Lifecycle::

        stream = FlashBackStream(smoothing=0.0)
        for batch in source:
            stream.add_sequences(batch)            # appends; cheap
            print(stream.n_nodes, stream.n_edges)  # peek; instant
            if some_stop_condition:
                break
        graph = stream.finalize()                  # one-time CSR build
        graph.save('foundation.lzg')

    After ``finalize()`` the stream is consumed: any further calls
    raise :class:`RuntimeError`. To discard a partial build without
    finalizing, call :meth:`abort`.

    Args:
        smoothing: Laplace smoothing alpha for edge weights, applied
            at finalize. Default 0.0.
    """

    def __init__(self, smoothing: float = 0.0):
        self._cap = _c.fb_stream_open(smoothing=smoothing)
        self._consumed = False

    # ── Context-manager protocol ─────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        # If exiting on exception or without finalize(), release
        # resources rather than leak them.
        if not self._consumed:
            self.abort()
        return False

    # ── Operations ───────────────────────────────────────────────

    def add_sequences(self, sequences, abundances=None) -> None:
        """Append ``sequences`` (a list of str) to the running build.

        ``abundances`` is an optional list of equal length giving
        per-sequence counts. Empty / None entries and zero counts are
        silently skipped. Raises :class:`RuntimeError` if the stream
        has been finalized or aborted.
        """
        if self._consumed:
            raise RuntimeError(
                "FlashBackStream is already finalized/aborted; "
                "create a new stream to add more sequences"
            )
        if isinstance(sequences, str):
            raise TypeError(
                "sequences must be a list of strings, not a single string"
            )
        seqs = list(sequences)
        if not seqs:
            return
        ab = list(abundances) if abundances is not None else None
        _c.fb_stream_add(self._cap, seqs, ab)

    @property
    def n_nodes(self) -> int:
        """Current number of distinct nodes in the running accumulator."""
        if self._consumed:
            return 0
        return _c.fb_stream_peek(self._cap)["n_nodes"]

    @property
    def n_edges(self) -> int:
        """Current number of distinct edges in the running accumulator."""
        if self._consumed:
            return 0
        return _c.fb_stream_peek(self._cap)["n_edges"]

    def peek(self) -> dict:
        """Return ``{'n_nodes': int, 'n_edges': int}`` from the running
        accumulator. Returns zeros if the stream is finalized/aborted."""
        if self._consumed:
            return {"n_nodes": 0, "n_edges": 0}
        return _c.fb_stream_peek(self._cap)

    def finalize(self) -> "FlashBackGraph":
        """Convert the running accumulator into a :class:`FlashBackGraph`.

        The stream is consumed by this call: subsequent operations
        raise :class:`RuntimeError`. The returned graph owns its CSR
        storage and is independent of the stream.
        """
        if self._consumed:
            raise RuntimeError("FlashBackStream is already consumed")
        graph_cap = _c.fb_stream_finalize(self._cap)
        self._consumed = True
        return FlashBackGraph._from_capsule(graph_cap)

    def abort(self) -> None:
        """Discard the partial build without producing a graph.

        Idempotent — safe to call after finalize/abort. Useful when an
        early stop condition triggers and you don't want to pay the
        finalize cost just to throw the result away.
        """
        if self._consumed:
            return
        _c.fb_stream_abort(self._cap)
        self._consumed = True

    def snapshot(self) -> "FlashBackGraph":
        """Build a finalized graph from the current accumulator state
        WITHOUT consuming the stream.

        The returned graph owns its own CSR storage but borrows the
        stream's string pool via refcount; it must be destroyed before
        the stream itself. Useful for periodic checkpointing during
        long-running streamed builds — compute metrics on the snapshot,
        save it, drop it, and keep accumulating into the same stream.

        Raises :class:`RuntimeError` if the stream is already finalized
        or aborted.
        """
        if self._consumed:
            raise RuntimeError(
                "FlashBackStream is finalized/aborted; cannot snapshot"
            )
        graph_cap = _c.fb_stream_snapshot(self._cap)
        return FlashBackGraph._from_capsule(graph_cap)
