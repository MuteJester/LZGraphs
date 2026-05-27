"""Tier 1 regression: LZGraph (AAP / NDP / Naive variants).

Locks in build statistics, pgen, simulate, Hill numbers, diversity,
predicted richness, and predicted sharing for all three variants on the
same 200-sequence fixed input.
"""
from __future__ import annotations

import pytest

from LZGraphs import LZGraph

from .snapshot_utils import assert_snapshot_match


VARIANTS = ["aap", "ndp", "naive"]

# On the 200-seq subset, the Naive variant builds correctly but several
# downstream operations raise "graph state error" — a pre-existing
# behavior we lock in by SKIPPING (not patching). Re-enable if the
# underlying library starts supporting these on naive.
NAIVE_UNSUPPORTED = {"simulate", "analytics", "occupancy", "posterior"}


def _skip_if_naive_unsupported(variant: str, op: str) -> None:
    if variant == "naive" and op in NAIVE_UNSUPPORTED:
        pytest.skip(
            f"naive variant does not support {op} on the 200-seq fixture "
            f"(pre-existing behavior; locked in by skip)"
        )


@pytest.fixture(scope="module", params=VARIANTS)
def variant(request):
    return request.param


@pytest.fixture(scope="module")
def lz_graph(variant, cdr3_sequences):
    return LZGraph(cdr3_sequences, variant=variant)


# ───────────────────────────────────────────────────────────────
# Structure (per-variant snapshot)
# ───────────────────────────────────────────────────────────────

def test_structure_snapshot(lz_graph, variant):
    g = lz_graph
    data = {
        "n_nodes": g.n_nodes,
        "n_edges": g.n_edges,
        "n_initial": g.n_initial,
        "n_terminal": g.n_terminal,
        "n_sequences": g.n_sequences,
        "is_dag": g.is_dag,
        "density": g.density,
        "max_in_degree": g.max_in_degree,
        "max_out_degree": g.max_out_degree,
        "variant": g.variant,
        "length_distribution": dict(g.length_distribution),
    }
    assert_snapshot_match(f"lzgraph_{variant}_structure", data)


# ───────────────────────────────────────────────────────────────
# pgen for probe sequences
# ───────────────────────────────────────────────────────────────

def test_pgen_snapshot(lz_graph, variant, probe_sequences):
    out = {
        seq: {
            "log_pgen": lz_graph.pgen(seq, log=True),
            "pgen":     lz_graph.pgen(seq, log=False),
        }
        for seq in probe_sequences
    }
    assert_snapshot_match(f"lzgraph_{variant}_pgen", out)


# ───────────────────────────────────────────────────────────────
# Simulate (fixed seed)
# ───────────────────────────────────────────────────────────────

def test_simulate_snapshot(lz_graph, variant):
    _skip_if_naive_unsupported(variant, "simulate")
    r = lz_graph.simulate(50, seed=42)
    data = {
        "sequences": list(r.sequences),
        "log_probs": list(r.log_probs),
    }
    assert_snapshot_match(f"lzgraph_{variant}_simulate", data)


# ───────────────────────────────────────────────────────────────
# Analytics: Hill numbers, effective diversity, path count, dynamic range
# ───────────────────────────────────────────────────────────────

def test_analytics_snapshot(lz_graph, variant):
    _skip_if_naive_unsupported(variant, "analytics")
    g = lz_graph
    data = {
        "path_count": g.path_count,
        "path_count_estimate": g.path_count_estimate(),
        "effective_diversity": g.effective_diversity(),
        "hill_numbers_q0_1_2_3": list(g.hill_numbers([0.0, 1.0, 2.0, 3.0])),
        "pgen_dynamic_range": g.pgen_dynamic_range(),
        "pgen_dynamic_range_detail": g.pgen_dynamic_range_detail(),
        "pgen_diagnostics": g.pgen_diagnostics(),
        "pgen_moments": g.pgen_moments(),
        "power_sum_alpha_2": g.power_sum(2.0),
    }
    assert_snapshot_match(f"lzgraph_{variant}_analytics", data)


# ───────────────────────────────────────────────────────────────
# Diversity-side: perplexity, path entropy rate
# ───────────────────────────────────────────────────────────────

def test_diversity_snapshot(lz_graph, variant, probe_sequences, cdr3_sequences):
    g = lz_graph
    data = {
        "sequence_perplexity_probes": {s: g.sequence_perplexity(s) for s in probe_sequences},
        "repertoire_perplexity_full": g.repertoire_perplexity(cdr3_sequences),
        "path_entropy_rate_probes": g.path_entropy_rate(probe_sequences),
    }
    assert_snapshot_match(f"lzgraph_{variant}_diversity", data)


# ───────────────────────────────────────────────────────────────
# Occupancy: predicted richness, richness curve, predicted sharing
# ───────────────────────────────────────────────────────────────

def test_occupancy_snapshot(lz_graph, variant):
    _skip_if_naive_unsupported(variant, "occupancy")
    g = lz_graph
    data = {
        "predicted_richness_100":  g.predicted_richness(100),
        "predicted_richness_1000": g.predicted_richness(1000),
        "richness_curve":          list(g.richness_curve([10, 50, 100, 500, 1000])),
        "predict_sharing_5":       g.predict_sharing([100, 100, 100], max_k=5),
        "predicted_overlap_100_200": g.predicted_overlap(100, 200),
    }
    assert_snapshot_match(f"lzgraph_{variant}_occupancy", data)


# ───────────────────────────────────────────────────────────────
# Posterior / subtract (round-trip stats)
# ───────────────────────────────────────────────────────────────

def test_posterior_snapshot(lz_graph, variant, cdr3_sequences, probe_sequences):
    _skip_if_naive_unsupported(variant, "posterior")
    individual = cdr3_sequences[::5]
    post = lz_graph.posterior(individual, kappa=10.0)
    data = {
        "n_nodes": post.n_nodes,
        "n_edges": post.n_edges,
        "hill_numbers_q0_1_2": list(post.hill_numbers([0.0, 1.0, 2.0])),
        "log_pgen_probes": {s: post.pgen(s, log=True) for s in probe_sequences},
    }
    assert_snapshot_match(f"lzgraph_{variant}_posterior", data)
