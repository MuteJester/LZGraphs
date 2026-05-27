"""Tier 1 regression: FlashBack grammar.

Locks in grammar structure, pgen (both backoff-respecting `pgen` and `pgen_mle`),
simulate, top-K, posterior+without round-trips, and analytics.
"""
from __future__ import annotations

import pytest

from LZGraphs import FlashBackGrammar

from .snapshot_utils import assert_snapshot_match


@pytest.fixture(scope="module")
def fb_grammar(cdr3_sequences):
    return FlashBackGrammar(cdr3_sequences)


def _individual_subset(cdr3_sequences):
    return cdr3_sequences[::5]


def test_structure_snapshot(fb_grammar):
    g = fb_grammar
    data = {
        "n_rules": g.n_rules,
        "n_internal_rules": g.n_internal_rules,
        "n_leaf_rules": g.n_leaf_rules,
        "n_nonterminals": g.n_nonterminals,
        "n_sequences": g.n_sequences,
        "max_length": g.max_length,
        "abundance_mode": g.abundance_mode,
        "backoff_mode": g.backoff_mode,
        "smoothing": g.smoothing,
        "variant": g.variant,
        "spectral_radius": g.spectral_radius,
        "is_consistent": g.is_consistent,
        "length_counts": dict(g.length_counts),
    }
    assert_snapshot_match("flashback_grammar_structure", data)


def test_pgen_snapshot(fb_grammar, probe_sequences):
    out = {
        seq: {
            "log_pgen":  fb_grammar.pgen(seq, log=True),
            "pgen":      fb_grammar.pgen(seq, log=False),
            "pgen_mle_log":        fb_grammar.pgen_mle(seq, log=True),
            "pgen_mle":            fb_grammar.pgen_mle(seq, log=False),
        }
        for seq in probe_sequences
    }
    assert_snapshot_match("flashback_grammar_pgen", out)


def test_simulate_snapshot(fb_grammar):
    r = fb_grammar.simulate(50, seed=42)
    data = {
        "sequences": list(r.sequences),
        "log_probs": list(r.log_probs),
    }
    assert_snapshot_match("flashback_grammar_simulate", data)


def test_top_k_snapshot(fb_grammar):
    top10 = fb_grammar.top_k_sequences(10, most_probable=True, max_length=30)
    bottom10 = fb_grammar.top_k_sequences(10, most_probable=False, max_length=30)
    data = {
        "top10_sequences":    list(top10.sequences),
        "top10_log_probs":    list(top10.log_probs),
        "bottom10_sequences": list(bottom10.sequences),
        "bottom10_log_probs": list(bottom10.log_probs),
    }
    assert_snapshot_match("flashback_grammar_topk", data)


def test_analytics_snapshot(fb_grammar):
    g = fb_grammar
    data = {
        "path_count_30": g.path_count(30),
        "path_count_series_30": list(g.path_count_series(30)),
        "effective_diversity": g.effective_diversity(),
        "hill_numbers_q0_1_2_3": list(g.hill_numbers([0.0, 1.0, 2.0, 3.0])),
        "pgen_dynamic_range": g.pgen_dynamic_range(max_length=30),
        "pgen_dynamic_range_detail": g.pgen_dynamic_range_detail(max_length=30),
        "power_sum_alpha_2": g.power_sum(2.0),
        "entropy": g.entropy(),
        "diversity_profile": g.diversity_profile(),
    }
    assert_snapshot_match("flashback_grammar_analytics", data)


def test_posterior_snapshot(fb_grammar, cdr3_sequences, probe_sequences):
    individual = _individual_subset(cdr3_sequences)
    post = fb_grammar.posterior(individual, kappa=10.0)
    data = {
        "n_rules": post.n_rules,
        "n_internal_rules": post.n_internal_rules,
        "n_leaf_rules": post.n_leaf_rules,
        "n_sequences": post.n_sequences,
        "spectral_radius": post.spectral_radius,
        "hill_numbers_q0_1_2": list(post.hill_numbers([0.0, 1.0, 2.0])),
        "log_pgen_probes": {s: post.pgen(s, log=True) for s in probe_sequences},
    }
    assert_snapshot_match("flashback_grammar_posterior", data)


def test_subtract_snapshot(fb_grammar, cdr3_sequences, probe_sequences):
    individual = _individual_subset(cdr3_sequences)
    sub = fb_grammar.without(individual)
    data = {
        "n_rules": sub.n_rules,
        "n_internal_rules": sub.n_internal_rules,
        "n_leaf_rules": sub.n_leaf_rules,
        "n_sequences": sub.n_sequences,
        "spectral_radius": sub.spectral_radius,
        "hill_numbers_q0_1_2": list(sub.hill_numbers([0.0, 1.0, 2.0])),
        "log_pgen_probes": {s: sub.pgen(s, log=True) for s in probe_sequences},
    }
    assert_snapshot_match("flashback_grammar_subtract", data)
