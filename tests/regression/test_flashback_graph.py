"""Tier 1 regression: FlashBack graph.

Locks in build statistics, pgen, FBAS, top-K walks, posterior+subtract
round-trips, simulate, and analytics on a 200-sequence fixed input.

All MC operations use fixed seeds. Snapshots are split per concern so a
drift shows up as a small, focused diff.
"""
from __future__ import annotations

from LZGraphs import FlashBackGraph

from .snapshot_utils import assert_snapshot_match


# ───────────────────────────────────────────────────────────────
# Fixtures local to this module
# ───────────────────────────────────────────────────────────────
import pytest


@pytest.fixture(scope="module")
def fb_graph(cdr3_sequences):
    return FlashBackGraph(cdr3_sequences)


# A deterministic "individual" subset for posterior / subtract tests.
def _individual_subset(cdr3_sequences):
    # Take every 5th sequence — small, deterministic, in-graph.
    return cdr3_sequences[::5]


# ───────────────────────────────────────────────────────────────
# Structure
# ───────────────────────────────────────────────────────────────

def test_structure_snapshot(fb_graph):
    g = fb_graph
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
    assert_snapshot_match("flashback_graph_structure", data)


# ───────────────────────────────────────────────────────────────
# Per-sequence pgen
# ───────────────────────────────────────────────────────────────

def test_pgen_snapshot(fb_graph, probe_sequences):
    out = {
        seq: {
            "log_pgen": fb_graph.pgen(seq, log=True),
            "pgen":     fb_graph.pgen(seq, log=False),
        }
        for seq in probe_sequences
    }
    assert_snapshot_match("flashback_graph_pgen", out)


# ───────────────────────────────────────────────────────────────
# FBAS
# ───────────────────────────────────────────────────────────────

def test_fbas_snapshot(fb_graph, probe_sequences):
    out = {seq: fb_graph.flashback_fbas(seq) for seq in probe_sequences}
    assert_snapshot_match("flashback_graph_fbas", out)


# ───────────────────────────────────────────────────────────────
# Top-K walks
# ───────────────────────────────────────────────────────────────

def test_topk_snapshot(fb_graph):
    top10 = fb_graph.top_k_sequences(10, most_probable=True)
    bottom10 = fb_graph.top_k_sequences(10, most_probable=False)
    out = {
        "top10_sequences":     list(top10.sequences),
        "top10_log_probs":     list(top10.log_probs),
        "bottom10_sequences":  list(bottom10.sequences),
        "bottom10_log_probs":  list(bottom10.log_probs),
    }
    assert_snapshot_match("flashback_graph_topk", out)


# ───────────────────────────────────────────────────────────────
# Analytics
# ───────────────────────────────────────────────────────────────

def test_analytics_snapshot(fb_graph):
    g = fb_graph
    data = {
        "path_count": g.path_count,
        "effective_diversity": g.effective_diversity(),
        "hill_numbers_q0_1_2_3": list(g.hill_numbers([0.0, 1.0, 2.0, 3.0])),
        "pgen_dynamic_range": g.pgen_dynamic_range(),
        "pgen_dynamic_range_detail": g.pgen_dynamic_range_detail(),
        "pgen_diagnostics": g.pgen_diagnostics(),
        "pgen_moments": g.pgen_moments(),
        "power_sum_alpha_2": g.power_sum(2.0),
    }
    assert_snapshot_match("flashback_graph_analytics", data)


# ───────────────────────────────────────────────────────────────
# Posterior round-trip
# ───────────────────────────────────────────────────────────────

def test_posterior_snapshot(fb_graph, cdr3_sequences, probe_sequences):
    individual = _individual_subset(cdr3_sequences)
    post = fb_graph.posterior(individual, kappa=10.0)
    data = {
        "n_nodes": post.n_nodes,
        "n_edges": post.n_edges,
        "n_sequences": post.n_sequences,
        "hill_numbers_q0_1_2": list(post.hill_numbers([0.0, 1.0, 2.0])),
        "log_pgen_probes": {s: post.pgen(s, log=True) for s in probe_sequences},
    }
    assert_snapshot_match("flashback_graph_posterior", data)


# ───────────────────────────────────────────────────────────────
# Subtract / without round-trip
# ───────────────────────────────────────────────────────────────

def test_subtract_snapshot(fb_graph, cdr3_sequences, probe_sequences):
    individual = _individual_subset(cdr3_sequences)
    sub = fb_graph.without(individual)
    data = {
        "n_nodes": sub.n_nodes,
        "n_edges": sub.n_edges,
        "n_sequences": sub.n_sequences,
        "hill_numbers_q0_1_2": list(sub.hill_numbers([0.0, 1.0, 2.0])),
        "log_pgen_probes": {s: sub.pgen(s, log=True) for s in probe_sequences},
    }
    assert_snapshot_match("flashback_graph_subtract", data)


# ───────────────────────────────────────────────────────────────
# Simulate
# ───────────────────────────────────────────────────────────────

def test_simulate_snapshot(fb_graph):
    r = fb_graph.simulate(50, seed=42)
    data = {
        "sequences": list(r.sequences),
        "log_probs": list(r.log_probs),
        "n_tokens":  list(r.n_tokens),
    }
    assert_snapshot_match("flashback_graph_simulate", data)
