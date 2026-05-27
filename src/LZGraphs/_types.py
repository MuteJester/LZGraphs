"""TypedDict definitions for structured dict returns across the library.

These describe the shape of dicts returned by methods like ``summary()``,
``pgen_diagnostics()``, ``flashback_fbas()``, etc. Users who want to type
their own code against these returns can import the relevant TypedDict
from ``LZGraphs``.

All TypedDicts use ``total=False`` for forward compatibility — adding new
keys to a return dict in a future version of the library does not break
existing code that consumes the type.
"""
from __future__ import annotations

from typing import TypedDict


class Summary(TypedDict, total=False):
    """Return type of ``LZGraph.summary()`` / ``FlashBackGraph.summary()``."""
    n_nodes: int
    n_edges: int
    n_initial: int
    n_terminal: int
    max_in_degree: int
    max_out_degree: int
    variant: str
    is_dag: bool


class PgenDiagnostics(TypedDict):
    """Return type of ``pgen_diagnostics(atol)``."""
    total_absorbed: float
    total_leaked: float
    initial_prob_sum: float
    is_proper: bool
    mc_samples: int


class DynamicRange(TypedDict):
    """Return type of ``pgen_dynamic_range_detail()`` (exact-DP, FlashBack)."""
    max_log_prob: float
    min_log_prob: float
    dynamic_range_nats: float
    dynamic_range_orders: float


class PgenMoments(TypedDict):
    """Return type of ``pgen_moments()`` — first four moments of the
    log-probability distribution."""
    mean: float
    variance: float
    std: float
    skewness: float
    kurtosis: float


class DiversityProfile(TypedDict):
    """Return type of ``diversity_profile()`` — Shannon decomposition."""
    entropy_nats: float
    entropy_bits: float
    effective_diversity: float
    uniformity: float


class FbasResult(TypedDict):
    """Return type of ``FlashBackGraph.flashback_fbas(sequence)``."""
    fbas: float
    log_pgen: float
    worst_excess: float
    n_missing_tokens: int
    n_missing_edges: int


class SharingSpectrum(TypedDict):
    """Return type of ``LZGraph.predict_sharing(draw_counts, max_k)``."""
    spectrum: "np.ndarray"            # shape (max_k+1,)
    expected_total: float
    n_donors: int


class ApproximationDiagnostics(TypedDict):
    """Return type of ``LZGraph.approximation_diagnostics(test_sequences, ...)``.

    Reports model coverage and rank correlation against an empirical test
    distribution.
    """
    coverage: float
    cross_entropy_nats: float
    empirical_entropy_nats: float
    kl_divergence_nats: float
    perplexity: float
    mean_log_prob: float
    median_log_prob: float
    rank_correlation: float
    n_unique: int
    n_covered: int


class GrammarRule(TypedDict):
    """Return-element type of ``FlashBackGrammar.rules_at(a, z)``."""
    kind: str
    a_char: str
    z_char: str
    a_run_len: int
    z_run_len: int
    dst_a: str
    dst_z: str
    weight: float


# Forward-reference fix for numpy in SharingSpectrum
import numpy as np  # noqa: E402  (annotation is a string under __future__)
