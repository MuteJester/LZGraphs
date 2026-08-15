"""NaiveGraph — positional baseline for the FlashBack decomposition.

Deliberately the simplest encoding that still produces a sentinel-bounded
DAG: a node is one residue together with its 1-based index, so ``CASS``
walks ``@ -> C_1 -> A_2 -> S_3 -> S_4 -> $``. Nothing about repeat
structure enters the node alphabet.

That is the point. :class:`~LZGraphs.FlashBackGraph` and this class share
the CSR engine, the MLE edge weights, and the exact forward-DP analytics,
and differ in exactly one respect: what a node is. Any advantage
FlashBack shows over a NaiveGraph trained on the same sequences is
therefore attributable to the decomposition rather than to the graph
machinery around it.

Two structural facts make the comparison exact on both sides. The index
strictly increases along a walk, so the graph is acyclic; and the
walk/sequence map is a bijection, so the path count *is* the number of
distinct sequences the model supports and each root-to-sink path product
*is* that sequence's probability. No Monte Carlo enters anywhere.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, overload

import numpy as np

from . import _clzgraph as _c
from ._graph_common import _GraphCommonMixin
from ._simulation_result import SimulationResult
from ._types import (
    DiversityProfile,
    DynamicRange,
    PgenDiagnostics,
    Summary,
)

#: Default longest sequence admitted to a NaiveGraph. Node count grows as
#: ``alphabet x max_length``, and a comparison against another model is
#: only well defined if both are trained on the same length range.
DEFAULT_MAX_LENGTH = 27


def naive_decompose(sequence: str) -> list[str]:
    """Node labels of ``sequence``'s walk, sentinels included.

    The counterpart of :func:`~LZGraphs.flashback_decompose`.

    Example: ``naive_decompose('CASSAY')`` ->
        ``['@', 'C_1', 'A_2', 'S_3', 'S_4', 'A_5', 'Y_6', '$']``
    """
    return _c.naive_decompose(sequence)


class NaiveGraph(_GraphCommonMixin):
    """Positional graph over sequences of bounded length.

    Args:
        sequences: Training sequences.
        abundances: Per-sequence counts. None means all 1.
        max_length: Skip sequences longer than this; None means no limit.
            Sequences over the cap are excluded from the graph *and* from
            its length distribution, since the model cannot represent
            them.
        smoothing: Laplace alpha for edge weights. It reweights observed
            edges only and never creates new ones, so it changes
            calibration without changing support.
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
        max_length: int | None = DEFAULT_MAX_LENGTH,
        smoothing: float = 0.0,
    ) -> None:
        if not sequences:
            raise ValueError("sequences must be a non-empty list")
        if isinstance(sequences, str):
            raise TypeError(
                "sequences must be a list of strings, not a single string")
        seqs = list(sequences)
        abs_list = list(abundances) if abundances is not None else None
        if abs_list is not None and len(abs_list) != len(seqs):
            raise ValueError(
                f"abundances length {len(abs_list)} != "
                f"sequences length {len(seqs)}")
        self._max_length = _normalize_max_length(max_length)
        self._cap = _c.naive_graph_build(
            seqs,
            abs_list,
            self._max_length,
            smoothing,
        )
        self._info = _c.graph_info(self._cap)

    @classmethod
    def _from_capsule(cls, capsule: Any,
                      max_length: int = 0) -> NaiveGraph:
        obj = object.__new__(cls)
        obj._cap = capsule
        obj._max_length = max_length
        obj._info = _c.graph_info(capsule)
        return obj

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike,
        *,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        smoothing: float = 0.0,
    ) -> NaiveGraph:
        """Build from a sequence file, auto-detecting format.

        A clean, uncompressed plain or ``sequence<TAB>abundance`` file is
        streamed straight into the C builder in constant memory, which is
        what makes a multi-GB corpus practical. Anything else
        (compressed, headered, FASTA/FASTQ) is read through
        ``read_sequences`` first and built the same way the list
        constructor would. Gene columns, if present, are ignored: this
        class carries no V/J annotation.
        """
        path = os.fspath(path)
        if not path:
            raise ValueError("path must be non-empty")
        cap_len = _normalize_max_length(max_length)

        from ._io import empty_read_error, plan_streaming_read, read_sequences

        can_stream, _spec = plan_streaming_read(path, variant="aap")
        if can_stream:
            return cls._from_capsule(
                _c.naive_graph_build_file(path, cap_len, smoothing), cap_len)

        data = read_sequences(path, variant="aap", no_genes=True)
        if not data['sequences']:
            raise empty_read_error(path, data['stats'])
        return cls(
            data['sequences'],
            abundances=data['abundances'],
            max_length=max_length,
            smoothing=smoothing,
        )

    # ── Dunder ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"NaiveGraph(nodes={self.n_nodes}, edges={self.n_edges})"

    # ── Basic properties ────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        """Node count, including the ``@`` and ``$`` sentinels."""
        return self._info['n_nodes']

    @property
    def n_edges(self) -> int:
        """Number of directed edges."""
        return self._info['n_edges']

    @property
    def variant(self) -> str:
        """Graph encoding variant; always ``'naive_positional'``."""
        return 'naive_positional'

    @property
    def is_dag(self) -> bool:
        """Whether the graph is acyclic. Always True for this encoding."""
        return self._info['is_dag']

    @property
    def max_length(self) -> int:
        """Length cap applied at build time; 0 means no cap was set."""
        return self._max_length

    @property
    def observed_max_length(self) -> int:
        """Longest sequence actually seen during training."""
        d = self.length_distribution
        return max(d) if d else 0

    @property
    def path_count(self) -> int:
        """Exact number of distinct sequences this model can generate.

        Arbitrary precision. Because walks and sequences are in
        bijection, this is both the DAG path count and the support size
        of the distribution.
        """
        if self._path_count_cache is None:
            self._path_count_cache = _c.naive_path_count_exact(self._cap)
        return self._path_count_cache

    # ── Structural ──────────────────────────────────────────

    @property
    def n_sequences(self) -> int:
        """Total training sequences, weighted by abundance."""
        return sum(self.length_distribution.values())

    @property
    def length_distribution(self) -> dict[int, int]:
        """Sequence length distribution: ``{length: count}``."""
        if self._length_dist_cache is None:
            self._length_dist_cache = _c.graph_length_distribution(self._cap)
        return dict(self._length_dist_cache)

    @property
    def nodes(self) -> list[str]:
        """Node labels, excluding the ``@`` and ``$`` sentinels."""
        if self._nodes_cache is None:
            raw = _c.graph_nodes(self._cap)
            self._all_nodes_cache = raw
            self._nodes_cache = [n for n in raw if n not in ('@', '$')]
        return list(self._nodes_cache)

    @property
    def all_nodes(self) -> list[str]:
        """Node labels, including the sentinels."""
        if self._all_nodes_cache is None:
            self._all_nodes_cache = _c.graph_nodes(self._cap)
        return list(self._all_nodes_cache)

    @property
    def edges(self) -> list[tuple[str, str, float, int]]:
        """``(source, target, weight, count)`` for non-sentinel edges."""
        if self._edges_cache is None:
            raw = _c.graph_edges(self._cap)
            self._all_edges_cache = raw
            self._edges_cache = [
                e for e in raw
                if e[0] not in ('@', '$') and e[1] not in ('@', '$')
            ]
        return list(self._edges_cache)

    @property
    def all_edges(self) -> list[tuple[str, str, float, int]]:
        """``(source, target, weight, count)`` for every edge."""
        if self._all_edges_cache is None:
            self._all_edges_cache = _c.graph_edges(self._cap)
        return list(self._all_edges_cache)

    @property
    def n_initial(self) -> int:
        """Number of nodes reachable from the ``@`` root."""
        return self._get_summary()['n_initial']

    @property
    def n_terminal(self) -> int:
        """Number of sink nodes. Always 1: every walk ends at ``$``."""
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
        """``n_edges / (n_nodes * (n_nodes - 1))``."""
        n = self.n_nodes
        return self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    @property
    def out_degrees(self) -> np.ndarray:
        """Out-degree of each node, indexed by node ID."""
        return np.array(self._get_degrees()['out_degrees'], dtype=np.uint32)

    @property
    def in_degrees(self) -> np.ndarray:
        """In-degree of each node, indexed by node ID."""
        return np.array(self._get_degrees()['in_degrees'], dtype=np.uint32)

    # ── Adjacency ───────────────────────────────────────────

    def adjacency_csr(self) -> dict[str, np.ndarray]:
        """CSR adjacency representation."""
        csr = self._get_csr()
        return {
            'row_offsets': csr['row_offsets'].copy(),
            'col_indices': csr['col_indices'].copy(),
            'weights': csr['weights'].copy(),
            'counts': csr['counts'].copy(),
        }

    # ── Decomposition ───────────────────────────────────────

    def decompose(self, sequence: str) -> list[str]:
        """Node labels of ``sequence``'s walk, sentinels included."""
        return _c.naive_decompose(sequence)

    # ── Simulation ──────────────────────────────────────────

    def simulate(self, n: int, *, seed: int | None = None) -> SimulationResult:
        """Generate ``n`` sequences by Markov random walk from ``@``."""
        seed_val = seed if seed is not None else -1
        seqs, lps, nts = _c.naive_simulate(self._cap, n, seed=seed_val)
        return SimulationResult(seqs, lps, nts)

    # ── Probability ─────────────────────────────────────────

    @overload
    def pseq(self, sequence: str, *, log: bool = True) -> float: ...
    @overload
    def pseq(self, sequence: list[str], *,
             log: bool = True) -> np.ndarray: ...

    def pseq(self, sequence, *, log=True):
        """Exact probability of sequence(s) under the positional model.

        Named Pseq, not Pgen: this is the probability of *observing* the
        sequence under a model fitted to observed repertoires, not the
        probability of the VDJ machinery *generating* it, which is what
        OLGA and the LZ graphs call Pgen. The two are different
        quantities and conflating them is the mistake the name exists to
        prevent.

        Unsupported sequences — an unseen residue/position pair, an
        unseen transition, or a final residue that never ended a training
        sequence — return the library's log floor, matching
        :meth:`FlashBackGraph.pgen`.

        Args:
            sequence: A single string, or a list of strings.
            log: If True, return log-probability.

        Returns:
            ``float`` for a single sequence, ``np.ndarray`` for a list.
        """
        raw = _c.naive_pgen(self._cap, sequence)   # C symbol keeps the
        # library-wide lzg_*_pgen spelling; only the Python API renames.
        if isinstance(raw, float):
            return raw if log else np.exp(raw)
        arr = np.array(raw, dtype=np.float64)
        return arr if log else np.exp(arr)

    # ── Exact analytics ────────────────────────────────────

    def diversity_profile(self) -> DiversityProfile:
        """Full Shannon diversity breakdown (exact). Cached per instance."""
        if self._eff_div_cache is None:
            self._eff_div_cache = _c.naive_effective_diversity(self._cap)
        return self._eff_div_cache

    def effective_diversity(self) -> float:
        """Exact ``exp(H)`` via forward DP."""
        return self.diversity_profile()['effective_diversity']

    def entropy(self) -> float:
        """Shannon entropy of the sequence distribution, in nats."""
        return self.diversity_profile()['entropy_nats']

    def hill_number(self, alpha: float) -> float:
        """Exact Hill number ``D(alpha)`` via forward DP."""
        return _c.naive_hill_number(self._cap, float(alpha))

    def hill_numbers(self, orders: Iterable[float]) -> np.ndarray:
        """Exact Hill numbers for multiple orders."""
        return np.array(_c.naive_hill_numbers(self._cap,
                                              [float(o) for o in orders]),
                        dtype=np.float64)

    def hill_curve(self,
                   orders: Iterable[float] | None = None
                   ) -> dict[str, np.ndarray]:
        """Hill diversity curve (exact)."""
        if orders is None:
            orders = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
        o_list = [float(o) for o in orders]
        return {
            'orders': np.array(o_list, dtype=np.float64),
            'values': self.hill_numbers(o_list),
        }

    def power_sum(self, alpha: float) -> float:
        """Exact power sum ``M(alpha) = sum P(s)**alpha`` via forward DP."""
        return _c.naive_power_sum(self._cap, float(alpha))

    def pseq_diagnostics(self, atol: float = 1e-6) -> PgenDiagnostics:
        """Exact absorbed vs leaked probability mass.

        A well-formed NaiveGraph absorbs all of it: every node lies on at
        least one complete training walk, so none is a dead end.
        """
        return _c.naive_pgen_diagnostics(self._cap, atol)

    def pseq_dynamic_range_detail(self) -> DynamicRange:
        """Full dynamic range breakdown (exact). Cached per instance."""
        if self._dyn_range_cache is None:
            self._dyn_range_cache = _c.naive_dynamic_range(self._cap)
        return self._dyn_range_cache

    def pseq_dynamic_range(self) -> float:
        """Exact dynamic range of ``log10 Pseq`` across the support."""
        return self.pseq_dynamic_range_detail()['dynamic_range_orders']

    def pseq_moments(self):
        """Moments of the forward-DP log-Pseq distribution."""
        return _c.pgen_moments(self._cap)

    def pseq_analysis(self):
        """Sampling-free analytical p-sequence spectrum.

        The same :class:`FlashBackPseqAnalysis` machinery
        :meth:`FlashBackGraph.pseq_analysis` returns. It is generic over
        any sentinel-bounded DAG: the exact Mellin transform
        ``M(q) = sum P(s)**q`` and everything derived from it are forward
        dynamic programs over edges, indifferent to what a node means.
        """
        from ._flashback_pseq import FlashBackPseqAnalysis
        return FlashBackPseqAnalysis(self)

    # ── Cross-class compatibility aliases ──────────────────
    #
    # Pseq is the name for this quantity on this class. These four exist
    # only so code written against FlashBackGraph's surface can take a
    # NaiveGraph unchanged: the shared spectrum and scoring pipelines call
    # .pgen(), .pgen_moments() and .pgen_dynamic_range_detail() on whichever
    # graph they were handed. Prefer the pseq_* names in new code.

    pgen = pseq
    pgen_moments = pseq_moments
    pgen_diagnostics = pseq_diagnostics
    pgen_dynamic_range = pseq_dynamic_range
    pgen_dynamic_range_detail = pseq_dynamic_range_detail

    # ── Graph operations ───────────────────────────────────

    def union(self, other: NaiveGraph) -> NaiveGraph:
        """Union: sum edge counts from both graphs."""
        cap = _c.graph_union(self._cap, other._cap)
        _c.naive_fix_special_nodes(cap)
        return NaiveGraph._from_capsule(cap, self._max_length)

    def intersection(self, other: NaiveGraph) -> NaiveGraph:
        """Intersection: keep shared edges, with the minimum counts."""
        cap = _c.graph_intersection(self._cap, other._cap)
        _c.naive_fix_special_nodes(cap)
        return NaiveGraph._from_capsule(cap, self._max_length)

    def difference(self, other: NaiveGraph) -> NaiveGraph:
        """Difference: subtract ``other``'s edge counts."""
        cap = _c.graph_difference(self._cap, other._cap)
        _c.naive_fix_special_nodes(cap)
        return NaiveGraph._from_capsule(cap, self._max_length)

    def weighted_merge(self, other: NaiveGraph, alpha: float = 1.0,
                       beta: float = 1.0) -> NaiveGraph:
        """Weighted merge: ``alpha * self + beta * other``."""
        cap = _c.weighted_merge(self._cap, other._cap, alpha, beta)
        _c.naive_fix_special_nodes(cap)
        return NaiveGraph._from_capsule(cap, self._max_length)

    # ── Features ───────────────────────────────────────────

    def feature_stats(self) -> np.ndarray:
        """15-element graph statistics vector for ML pipelines."""
        return np.array(_c.feature_stats(self._cap), dtype=np.float64)

    # No feature_mass_profile: the C implementation routes through the
    # LZ76-constrained simulator, which rejects sentinel-DAG variants.
    # It raises for FlashBackGraph too. Use length_distribution, or
    # simulate() and histogram the lengths yourself.

    # ── IO ─────────────────────────────────────────────────

    def save(self, path: str | os.PathLike) -> None:
        """Save to LZG binary format (``.lzg``)."""
        _c.save(self._cap, os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike) -> NaiveGraph:
        """Load from LZG binary format.

        The build-time length cap is not stored in the file, so
        :attr:`max_length` reads 0 on a loaded graph; use
        :attr:`observed_max_length` for what the training data actually
        contained.
        """
        cap = _c.load(os.fspath(path))
        _c.naive_fix_special_nodes(cap)
        return cls._from_capsule(cap)

    def summary(self) -> Summary:
        """Structural summary dict."""
        return _c.summary(self._cap)


def _normalize_max_length(max_length: int | None) -> int:
    """Map the Python-level cap onto the C convention (0 = unlimited)."""
    if max_length is None:
        return 0
    max_length = int(max_length)
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0 or None, got {max_length}")
    return max_length
