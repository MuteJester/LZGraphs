"""FlattenedFlashBackGraph — FlashBack's bilateral scan, minus run compression.

FlashBack peels matching *runs* from both ends, so one token can absorb
several identical residues. This variant makes the same inward pass but
takes exactly one residue from each end per step::

    CASSAYFF  ->  @ -> CF_1 -> AF_2 -> SY_3 -> SA_4 -> $
    CASSLGQ   ->  @ -> CQ_1 -> AG_2 -> SL_3 -> S$_4 -> $

A token is ``{front}{back}_{step}``. An odd-length sequence leaves one
residue unpaired in the middle, written ``{residue}$_{step}``.

It exists to isolate one variable at a time. Against
:class:`~LZGraphs.FlashBackGraph` it holds the bilateral scan constant and
removes only run compression, so the gap between them is what compressing
repeats is worth. Against :class:`~LZGraphs.NaiveGraph` it holds "no
compression" constant and adds only the bilateral scan, so that gap is what
reading from both ends is worth. Neither number is recoverable from
FlashBack and naive alone.

Walks end at an explicit ``$`` sink, which is load-bearing rather than
cosmetic: a pair token cannot signal "stop" on its own, because the same
label is terminal in one sequence and internal in a longer one (``AA_2``
ends ``CAAF`` but continues in ``CAAAAF``). Without the sink the model
could place no probability on stopping and would not be a distribution
over sequences.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, overload

import numpy as np

from . import _clzgraph as _c
from ._graph_common import _GraphCommonMixin
from ._simulation_result import SimulationResult
from ._types import DiversityProfile, DynamicRange, PgenDiagnostics, Summary

#: Default longest sequence admitted. Unlike the naive positional encoding
#: this cap is not needed to bound the node set, but it keeps a comparison
#: against a capped model trained on the same length range.
DEFAULT_MAX_LENGTH = 27


def flat_decompose(sequence: str) -> list[str]:
    """Node labels of ``sequence``'s walk, sentinels included.

    Example: ``flat_decompose('CASSLGQ')`` ->
        ``['@', 'CQ_1', 'AG_2', 'SL_3', 'S$_4', '$']``
    """
    return _c.flat_decompose(sequence)


class FlattenedFlashBackGraph(_GraphCommonMixin):
    """Bilateral pair-token graph over sequences of bounded length.

    Args:
        sequences: Training sequences.
        abundances: Per-sequence counts. None means all 1.
        max_length: Skip sequences longer than this; None means no limit.
        smoothing: Laplace alpha for edge weights.
    """
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
        self._cap = _c.flat_graph_build(seqs, abs_list, self._max_length,
                                        smoothing)
        self._info = _c.graph_info(self._cap)

    @classmethod
    def _from_capsule(cls, capsule: Any,
                      max_length: int = 0) -> FlattenedFlashBackGraph:
        obj = object.__new__(cls)
        obj._cap = capsule
        obj._max_length = max_length
        obj._info = _c.graph_info(capsule)
        return obj

    @classmethod
    def from_file(cls, path: str | os.PathLike, *,
                  max_length: int | None = DEFAULT_MAX_LENGTH,
                  smoothing: float = 0.0) -> FlattenedFlashBackGraph:
        """Build from a sequence file, auto-detecting format."""
        path = os.fspath(path)
        if not path:
            raise ValueError("path must be non-empty")
        cap_len = _normalize_max_length(max_length)

        from ._io import empty_read_error, plan_streaming_read, read_sequences

        can_stream, _spec = plan_streaming_read(path, variant="aap")
        if can_stream:
            return cls._from_capsule(
                _c.flat_graph_build_file(path, cap_len, smoothing), cap_len)

        data = read_sequences(path, variant="aap", no_genes=True)
        if not data['sequences']:
            raise empty_read_error(path, data['stats'])
        return cls(data['sequences'], abundances=data['abundances'],
                   max_length=max_length, smoothing=smoothing)

    # ── Dunder ──────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"FlattenedFlashBackGraph(nodes={self.n_nodes}, "
                f"edges={self.n_edges})")

    # ── Basic properties ────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        """Node count, including the ``@`` and ``$`` sentinels."""
        return self._info['n_nodes']

    @property
    def n_edges(self) -> int:
        return self._info['n_edges']

    @property
    def variant(self) -> str:
        return 'flattened_flashback'

    @property
    def is_dag(self) -> bool:
        """Always True: the step index strictly increases along a walk."""
        return self._info['is_dag']

    @property
    def max_length(self) -> int:
        """Length cap applied at build time; 0 means no cap."""
        return self._max_length

    @property
    def observed_max_length(self) -> int:
        d = self.length_distribution
        return max(d) if d else 0

    @property
    def path_count(self) -> int:
        """Exact number of distinct sequences the model can generate.

        Walks and sequences are in bijection (the step index fixes where
        each residue pair lands), so this is also the support size.
        """
        if self._path_count_cache is None:
            self._path_count_cache = _c.flat_path_count_exact(self._cap)
        return self._path_count_cache

    # ── Structural ──────────────────────────────────────────

    @property
    def n_sequences(self) -> int:
        return sum(self.length_distribution.values())

    @property
    def length_distribution(self) -> dict[int, int]:
        if self._length_dist_cache is None:
            self._length_dist_cache = _c.graph_length_distribution(self._cap)
        return dict(self._length_dist_cache)

    @property
    def nodes(self) -> list[str]:
        """Node labels excluding the bare sentinels.

        A middle token such as ``S$_4`` contains a sentinel character but is
        an ordinary node, so only the bare ``@`` and ``$`` are filtered.
        """
        if self._nodes_cache is None:
            raw = _c.graph_nodes(self._cap)
            self._all_nodes_cache = raw
            self._nodes_cache = [n for n in raw if n not in ('@', '$')]
        return list(self._nodes_cache)

    @property
    def all_nodes(self) -> list[str]:
        if self._all_nodes_cache is None:
            self._all_nodes_cache = _c.graph_nodes(self._cap)
        return list(self._all_nodes_cache)

    @property
    def edges(self) -> list[tuple[str, str, float, int]]:
        if self._edges_cache is None:
            raw = _c.graph_edges(self._cap)
            self._all_edges_cache = raw
            self._edges_cache = [
                e for e in raw
                if e[0] not in ('@', '$') and e[1] not in ('@', '$')]
        return list(self._edges_cache)

    @property
    def all_edges(self) -> list[tuple[str, str, float, int]]:
        if self._all_edges_cache is None:
            self._all_edges_cache = _c.graph_edges(self._cap)
        return list(self._all_edges_cache)

    @property
    def n_initial(self) -> int:
        return self._get_summary()['n_initial']

    @property
    def n_terminal(self) -> int:
        """Always 1: every walk ends at the shared ``$``."""
        return self._get_summary()['n_terminal']

    @property
    def max_out_degree(self) -> int:
        return self._get_summary()['max_out_degree']

    @property
    def max_in_degree(self) -> int:
        return self._get_summary()['max_in_degree']

    @property
    def density(self) -> float:
        n = self.n_nodes
        return self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    @property
    def out_degrees(self) -> np.ndarray:
        return np.array(self._get_degrees()['out_degrees'], dtype=np.uint32)

    @property
    def in_degrees(self) -> np.ndarray:
        return np.array(self._get_degrees()['in_degrees'], dtype=np.uint32)

    def adjacency_csr(self) -> dict[str, np.ndarray]:
        csr = self._get_csr()
        return {k: v.copy() for k, v in csr.items()}

    # ── Decomposition ───────────────────────────────────────

    def decompose(self, sequence: str) -> list[str]:
        return _c.flat_decompose(sequence)

    # ── Simulation ──────────────────────────────────────────

    def simulate(self, n: int, *, seed: int | None = None) -> SimulationResult:
        seed_val = seed if seed is not None else -1
        seqs, lps, nts = _c.flat_simulate(self._cap, n, seed=seed_val)
        return SimulationResult(seqs, lps, nts)

    # ── Probability ─────────────────────────────────────────

    @overload
    def pseq(self, sequence: str, *, log: bool = True) -> float: ...
    @overload
    def pseq(self, sequence: list[str], *,
             log: bool = True) -> np.ndarray: ...

    def pseq(self, sequence, *, log=True):
        """Exact probability of sequence(s) under this encoding.

        Named Pseq rather than Pgen for the same reason as
        :meth:`NaiveGraph.pseq`: this is the probability of *observing* the
        sequence under a model fitted to observed repertoires, not the
        probability of the recombination machinery generating it.
        """
        raw = _c.flat_pseq(self._cap, sequence)
        if isinstance(raw, float):
            return raw if log else np.exp(raw)
        arr = np.array(raw, dtype=np.float64)
        return arr if log else np.exp(arr)

    # ── Exact analytics ────────────────────────────────────

    def diversity_profile(self) -> DiversityProfile:
        if self._eff_div_cache is None:
            self._eff_div_cache = _c.flat_effective_diversity(self._cap)
        return self._eff_div_cache

    def effective_diversity(self) -> float:
        return self.diversity_profile()['effective_diversity']

    def entropy(self) -> float:
        return self.diversity_profile()['entropy_nats']

    def hill_number(self, alpha: float) -> float:
        return _c.flat_hill_number(self._cap, float(alpha))

    def hill_numbers(self, orders: Iterable[float]) -> np.ndarray:
        return np.array(_c.flat_hill_numbers(self._cap,
                                             [float(o) for o in orders]),
                        dtype=np.float64)

    def hill_curve(self, orders: Iterable[float] | None = None
                   ) -> dict[str, np.ndarray]:
        if orders is None:
            orders = [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 10]
        o = [float(x) for x in orders]
        return {'orders': np.array(o, dtype=np.float64),
                'values': self.hill_numbers(o)}

    def power_sum(self, alpha: float) -> float:
        return _c.flat_power_sum(self._cap, float(alpha))

    def pseq_diagnostics(self, atol: float = 1e-6) -> PgenDiagnostics:
        return _c.flat_pseq_diagnostics(self._cap, atol)

    def pseq_dynamic_range_detail(self) -> DynamicRange:
        if self._dyn_range_cache is None:
            self._dyn_range_cache = _c.flat_dynamic_range(self._cap)
        return self._dyn_range_cache

    def pseq_dynamic_range(self) -> float:
        return self.pseq_dynamic_range_detail()['dynamic_range_orders']

    def pseq_moments(self):
        """Moments of the forward-DP log-Pseq distribution."""
        return _c.pgen_moments(self._cap)

    def pseq_analysis(self):
        """Sampling-free analytical p-sequence spectrum."""
        from ._flashback_pseq import FlashBackPseqAnalysis
        return FlashBackPseqAnalysis(self)

    # ── Cross-class compatibility aliases ──────────────────
    #
    # Same reason as NaiveGraph: the shared spectrum and scoring pipelines
    # call .pgen(), .pgen_moments() and .pgen_dynamic_range_detail() on
    # whichever graph they are handed. Prefer the pseq_* names.

    pgen = pseq
    pgen_moments = pseq_moments
    pgen_diagnostics = pseq_diagnostics
    pgen_dynamic_range = pseq_dynamic_range
    pgen_dynamic_range_detail = pseq_dynamic_range_detail

    # ── Graph operations ───────────────────────────────────

    def union(self, other: FlattenedFlashBackGraph) -> FlattenedFlashBackGraph:
        cap = _c.graph_union(self._cap, other._cap)
        _c.flat_fix_special_nodes(cap)
        return FlattenedFlashBackGraph._from_capsule(cap, self._max_length)

    def intersection(self, other) -> FlattenedFlashBackGraph:
        cap = _c.graph_intersection(self._cap, other._cap)
        _c.flat_fix_special_nodes(cap)
        return FlattenedFlashBackGraph._from_capsule(cap, self._max_length)

    def difference(self, other) -> FlattenedFlashBackGraph:
        cap = _c.graph_difference(self._cap, other._cap)
        _c.flat_fix_special_nodes(cap)
        return FlattenedFlashBackGraph._from_capsule(cap, self._max_length)

    # ── IO ─────────────────────────────────────────────────

    def save(self, path: str | os.PathLike) -> None:
        _c.save(self._cap, os.fspath(path))

    @classmethod
    def load(cls, path: str | os.PathLike) -> FlattenedFlashBackGraph:
        cap = _c.load(os.fspath(path))
        _c.flat_fix_special_nodes(cap)
        return cls._from_capsule(cap)

    def summary(self) -> Summary:
        return _c.summary(self._cap)


def _normalize_max_length(max_length: int | None) -> int:
    if max_length is None:
        return 0
    max_length = int(max_length)
    if max_length < 0:
        raise ValueError(f"max_length must be >= 0 or None, got {max_length}")
    return max_length
