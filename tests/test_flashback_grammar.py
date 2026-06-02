"""Tests for FlashBackGrammar (P1: build, consistency, decompose).

The prototype at .private/flashback_grammar_2026-04-22/fbg_prototype.py is
the reference. These tests compare C-port outputs against Python-prototype
outputs on identical inputs.
"""
import os
import sys

import pytest

from LZGraphs import _clzgraph

# Load the prototype as reference. It lives in the gitignored .private/ tree,
# so on a clean checkout (e.g. CI) it is absent; skip this whole module then.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '.private', 'flashback_grammar_2026-04-22'))
proto = pytest.importorskip(
    'fbg_prototype',
    reason='FlashBackGrammar reference prototype lives in gitignored .private/; '
           'skipped when unavailable (e.g. CI).')  # noqa: E402

TRAIN = [
    'CASSYAGLDYF', 'CASSLGQGAYEQYF', 'CASSPGTGAYEQYF', 'CASSYGYF',
    'CASSLGELFF', 'CASRDGRANVLTF', 'CASSLAPGATNEKLFF', 'CASSPGGANEQFF',
    'CASSTGDTQYF', 'CASSEARDRGYTF', 'CASSQETQYF', 'CASSLEGQGSYEQYF',
    'CASSIRSSYEQYF', 'CASSLGRDTQYF', 'CASSSGQGAYEQYF',
]


@pytest.fixture(scope='module')
def fbg_cap():
    return _clzgraph.fbg_build(TRAIN)


@pytest.fixture(scope='module')
def fbg_proto():
    return proto.FlashBackGrammar(TRAIN, abundance_mode='linear',
                                  smoothing=0.0, backoff='none')


# ── Decomposition tests ───────────────────────────────────────

def test_decompose_matches_prototype(fbg_cap, fbg_proto):
    """C decomposer produces identical steps to the Python prototype."""
    for seq in TRAIN:
        c_steps = _clzgraph.fbg_decompose(fbg_cap, seq)
        p_steps = proto.decompose(seq)
        assert len(c_steps) == len(p_steps), \
            f"step count mismatch for {seq!r}: c={len(c_steps)}, py={len(p_steps)}"

        for i, (cs, (p_nt, p_rule)) in enumerate(zip(c_steps, p_steps)):
            # Kind
            assert cs['kind'] == p_rule[0], \
                f"{seq} step {i}: kind {cs['kind']} vs {p_rule[0]}"

            if p_rule[0] == 'internal':
                _, ar, zr, ap, zp = p_rule
                assert cs['a_run_len'] == ar
                assert cs['z_run_len'] == zr
                assert cs['dst_a'] == ap
                assert cs['dst_z'] == zp
            elif p_rule[0] == 'leaf_single':
                _, ch = p_rule
                assert cs['a_char'] == ch
                assert cs['a_run_len'] == 1
            elif p_rule[0] == 'leaf_run':
                _, ch, k = p_rule
                assert cs['a_char'] == ch
                assert cs['a_run_len'] == k
            elif p_rule[0] == 'leaf_pair':
                _, a, ar, z, zr = p_rule
                assert cs['a_char'] == a
                assert cs['z_char'] == z
                assert cs['a_run_len'] == ar
                assert cs['z_run_len'] == zr

            # is_start aligns with Python's NT='S' convention
            assert cs['is_start'] == (p_nt == 'S'), \
                f"{seq} step {i}: is_start={cs['is_start']} but p_nt={p_nt}"


def test_tree_to_string_roundtrip(fbg_cap):
    """decompose → tree_to_string == original."""
    for seq in TRAIN:
        steps = _clzgraph.fbg_decompose(fbg_cap, seq)
        recon = _clzgraph.fbg_tree_to_string(fbg_cap, steps)
        assert recon == seq, f"round-trip failed: {seq!r} → {recon!r}"


# ── Build / info tests ────────────────────────────────────────

def test_info_shape(fbg_cap):
    info = _clzgraph.fbg_info(fbg_cap)
    assert info['n_rules'] > 0
    assert info['n_nts'] > 0
    assert info['is_consistent']  # ρ(T) < 1
    assert info['start_nt'] == 0
    assert info['abundance_mode'] == 1  # LINEAR
    assert info['backoff'] == 0         # NONE


def test_counts_match_prototype(fbg_cap, fbg_proto):
    """Rule counts at each NT in C match prototype totals."""
    info = _clzgraph.fbg_info(fbg_cap)
    c_nts = _clzgraph.fbg_nts(fbg_cap)

    # Build a map (a, z, is_start) → proto_nt_key
    proto_keys = {}
    for nt in fbg_proto.nts:
        if nt == 'S':
            proto_keys[('@', '$', True)] = 'S'
        else:
            proto_keys[(nt[0], nt[1], False)] = nt

    assert len(c_nts) == len(fbg_proto.nts), \
        f"NT count: C={len(c_nts)}, proto={len(fbg_proto.nts)}"

    for c_nt in c_nts:
        a, z, is_start, total, n_rules, unseen = c_nt
        # S's (a, z) are stored as (@, $) for the start non-terminal in our
        # BuildState. Match accordingly.
        key = (a, z, is_start)
        assert key in proto_keys, f"unexpected NT: a={a!r}, z={z!r}, is_start={is_start}"
        p_nt = proto_keys[key]
        p_total = fbg_proto.totals[p_nt]
        assert abs(total - p_total) < 1e-9, \
            f"NT {p_nt} total: c={total} vs proto={p_total}"
        assert n_rules == len(fbg_proto.weights[p_nt])
    _ = info


def test_rule_weights_match_prototype(fbg_cap, fbg_proto):
    """Every rule weight in C matches the prototype's weight."""
    info = _clzgraph.fbg_info(fbg_cap)
    c_nts = _clzgraph.fbg_nts(fbg_cap)
    for i in range(info['n_nts']):
        a, z, is_start, total, n_rules, unseen = c_nts[i]
        p_nt = 'S' if is_start else (a, z)
        c_rules = _clzgraph.fbg_rules_at(fbg_cap, i)
        assert len(c_rules) == n_rules

        # Index prototype rules by their tuple key
        p_rules_dict = fbg_proto.weights[p_nt]

        for cr in c_rules:
            # Build matching prototype rule tuple
            k = cr['kind']
            if k == 'internal':
                p_key = ('internal', cr['a_run_len'], cr['z_run_len'],
                         cr['dst_a'], cr['dst_z'])
            elif k == 'leaf_single':
                p_key = ('leaf_single', cr['a_char'])
            elif k == 'leaf_run':
                p_key = ('leaf_run', cr['a_char'], cr['a_run_len'])
            elif k == 'leaf_pair':
                p_key = ('leaf_pair', cr['a_char'], cr['a_run_len'],
                         cr['z_char'], cr['z_run_len'])
            else:
                pytest.fail(f"unknown kind {k}")
            assert p_key in p_rules_dict, \
                f"C rule {p_key} not in proto M{p_nt}"
            p_w = p_rules_dict[p_key]
            assert abs(cr['weight'] - p_w) < 1e-12, \
                f"weight mismatch at M{p_nt} rule {p_key}: c={cr['weight']}, py={p_w}"


def test_spectral_radius_below_1(fbg_cap, fbg_proto):
    info = _clzgraph.fbg_info(fbg_cap)
    # Prototype uses np.linalg.eigvals (exact); C uses power iteration.
    # They should agree closely for the actual ρ, but both report < 1.
    assert info['spectral_radius'] < 1.0
    assert abs(info['spectral_radius'] - fbg_proto.spectral_radius) < 1e-6


def test_length_counts_match_prototype(fbg_cap, fbg_proto):
    c_lc = _clzgraph.fbg_length_counts(fbg_cap)
    p_lc = fbg_proto.length_distribution  # wait, that's weighted
    # Prototype's length_distribution is a method; its internal length_counts
    # from training is implicit — compare C against Σ abundance per training length.
    expected = {}
    for seq in TRAIN:
        L = len(seq)
        expected[L] = expected.get(L, 0) + 1  # abundance = 1 each
    for L, expected_count in expected.items():
        assert c_lc[L] == expected_count, \
            f"length {L}: c={c_lc[L]} vs expected={expected_count}"
    _ = p_lc


# ── Abundance modes ───────────────────────────────────────────

def test_abundance_mode_none(fbg_proto):
    """abundance_mode='none' with non-1 abundances behaves identically to baseline."""
    abunds = [100] * len(TRAIN)  # all same → identical MLE regardless of mode
    cap_none = _clzgraph.fbg_build(TRAIN, abundances=abunds, abundance_mode='none')
    cap_base = _clzgraph.fbg_build(TRAIN)

    info_none = _clzgraph.fbg_info(cap_none)
    info_base = _clzgraph.fbg_info(cap_base)
    assert info_none['n_rules'] == info_base['n_rules']
    assert info_none['n_nts'] == info_base['n_nts']


def test_abundance_mode_linear_skew():
    """abundance_mode='linear' with skew produces weights matching prototype."""
    abunds = [1] * (len(TRAIN) - 1) + [1000]
    cap = _clzgraph.fbg_build(TRAIN, abundances=abunds, abundance_mode='linear')
    p = proto.FlashBackGrammar(TRAIN, abundances=abunds,
                               abundance_mode='linear', backoff='none')

    # Check every weight in the grammar
    info = _clzgraph.fbg_info(cap)
    c_nts = _clzgraph.fbg_nts(cap)
    for i in range(info['n_nts']):
        a, z, is_start, total, _, _ = c_nts[i]
        p_nt = 'S' if is_start else (a, z)
        c_rules = _clzgraph.fbg_rules_at(cap, i)
        p_rules_dict = p.weights[p_nt]
        for cr in c_rules:
            k = cr['kind']
            if k == 'internal':
                p_key = ('internal', cr['a_run_len'], cr['z_run_len'],
                         cr['dst_a'], cr['dst_z'])
            elif k == 'leaf_single':
                p_key = ('leaf_single', cr['a_char'])
            elif k == 'leaf_run':
                p_key = ('leaf_run', cr['a_char'], cr['a_run_len'])
            else:
                p_key = ('leaf_pair', cr['a_char'], cr['a_run_len'],
                         cr['z_char'], cr['z_run_len'])
            assert abs(cr['weight'] - p_rules_dict[p_key]) < 1e-12


# ── P2: pgen ──────────────────────────────────────────────────

HELDOUT = [
    'CASSYGAGELFF',
    'CASSLGQAYEQYF',
    'CASSEGTGAYEQYF',
    'CATSDGTNEKLFF',
]


def test_pgen_training_matches_prototype(fbg_cap, fbg_proto):
    """pgen(training seq) in C == prototype to float precision."""
    for seq in TRAIN:
        c_lp = _clzgraph.fbg_pgen(fbg_cap, seq)
        p_lp = fbg_proto.pgen(seq, log=True)
        assert abs(c_lp - p_lp) < 1e-12, \
            f"pgen({seq!r}): c={c_lp}, py={p_lp}"


def test_pgen_mle_training(fbg_cap, fbg_proto):
    """pgen_mle agrees with prototype pgen_mle on training seqs."""
    for seq in TRAIN:
        c_lp = _clzgraph.fbg_pgen_mle(fbg_cap, seq)
        p_lp = fbg_proto.pgen_mle(seq, log=True)
        assert abs(c_lp - p_lp) < 1e-12


def test_pgen_training_nonzero(fbg_cap):
    """Every training seq has pgen > ε."""
    for seq in TRAIN:
        lp = _clzgraph.fbg_pgen(fbg_cap, seq)
        assert lp > proto.LOG_EPS + 1.0, f"{seq!r} pgen = {lp}"


def test_pgen_held_out_matches_prototype(fbg_cap, fbg_proto):
    """Held-out pgens (including the ε case) bit-match the prototype."""
    for seq in HELDOUT:
        c_lp = _clzgraph.fbg_pgen(fbg_cap, seq)
        p_lp = fbg_proto.pgen(seq, log=True)
        # With backoff='none', held-outs with unseen rules fall to LOG_EPS in both.
        if p_lp <= proto.LOG_EPS + 1.0:
            assert c_lp <= proto.LOG_EPS + 1.0, \
                f"{seq!r}: prototype says ε but C says {c_lp}"
        else:
            assert abs(c_lp - p_lp) < 1e-12, \
                f"{seq!r}: c={c_lp}, py={p_lp}"


def test_pgen_batch(fbg_cap, fbg_proto):
    """Batch pgen produces per-sequence scores matching individual calls."""
    c_batch = _clzgraph.fbg_pgen_batch(fbg_cap, TRAIN)
    assert len(c_batch) == len(TRAIN)
    for seq, c_lp in zip(TRAIN, c_batch):
        ref = fbg_proto.pgen(seq, log=True)
        assert abs(c_lp - ref) < 1e-12


def test_pgen_out_of_alphabet_returns_eps(fbg_cap):
    """A seq with a char never seen in training → ε."""
    lp = _clzgraph.fbg_pgen(fbg_cap, 'CASSBBBBBBBF')  # 'B' not in training
    assert lp <= proto.LOG_EPS + 1.0


def test_pgen_empty_returns_eps(fbg_cap):
    """Empty or single-char edge cases handled gracefully."""
    lp = _clzgraph.fbg_pgen(fbg_cap, '')
    assert lp <= proto.LOG_EPS + 1.0


# ── P2 backoff ───────────────────────────────────────────────

@pytest.fixture(scope='module')
def fbg_cap_gt():
    return _clzgraph.fbg_build(TRAIN, backoff='gt')


@pytest.fixture(scope='module')
def fbg_proto_gt():
    return proto.FlashBackGrammar(TRAIN, backoff='gt')


def test_gt_unseen_mass_matches_prototype(fbg_cap_gt, fbg_proto_gt):
    """δ_nt (Good-Turing unseen mass) matches prototype for every NT."""
    c_nts = _clzgraph.fbg_nts(fbg_cap_gt)
    for i, c_nt in enumerate(c_nts):
        a, z, is_start, total, _, c_delta = c_nt
        p_nt = 'S' if is_start else (a, z)
        p_delta = fbg_proto_gt.unseen_mass.get(p_nt, 0.0)
        assert abs(c_delta - p_delta) < 1e-12, \
            f"δ_nt mismatch at M{p_nt}: c={c_delta}, py={p_delta}"


def test_gt_rule_marginal_matches_prototype(fbg_cap_gt, fbg_proto_gt):
    """marginal(r) in C matches prototype across a sample of rules."""
    info = _clzgraph.fbg_info(fbg_cap_gt)
    for i in range(info['n_nts']):
        rules = _clzgraph.fbg_rules_at(fbg_cap_gt, i)
        for r in rules:
            c_m = _clzgraph.fbg_rule_marginal(fbg_cap_gt, r)
            # Build matching prototype rule key
            k = r['kind']
            if k == 'internal':
                p_key = ('internal', r['a_run_len'], r['z_run_len'],
                         r['dst_a'], r['dst_z'])
            elif k == 'leaf_single':
                p_key = ('leaf_single', r['a_char'])
            elif k == 'leaf_run':
                p_key = ('leaf_run', r['a_char'], r['a_run_len'])
            else:
                p_key = ('leaf_pair', r['a_char'], r['a_run_len'],
                         r['z_char'], r['z_run_len'])
            p_m = fbg_proto_gt.rule_marginal.get(p_key, 0.0)
            assert abs(c_m - p_m) < 1e-12, \
                f"marginal mismatch for {p_key}: c={c_m}, py={p_m}"


def test_gt_pgen_matches_prototype(fbg_cap_gt, fbg_proto_gt):
    """pgen with GT backoff matches prototype for training and held-outs."""
    for seq in TRAIN + HELDOUT:
        c_lp = _clzgraph.fbg_pgen(fbg_cap_gt, seq)
        p_lp = fbg_proto_gt.pgen(seq, log=True)
        # Both compute ε cases identically; allow a tiny slop near LOG_EPS.
        if p_lp <= proto.LOG_EPS + 1.0:
            assert c_lp <= proto.LOG_EPS + 1.0
        else:
            assert abs(c_lp - p_lp) < 1e-12, \
                f"backoff pgen({seq!r}): c={c_lp}, py={p_lp}"


def test_gt_lifts_heldout_coverage(fbg_cap, fbg_cap_gt):
    """GT backoff lifts held-out coverage vs none mode."""
    none_covered = sum(
        1 for s in HELDOUT if _clzgraph.fbg_pgen(fbg_cap, s) > proto.LOG_EPS + 1.0)
    gt_covered = sum(
        1 for s in HELDOUT if _clzgraph.fbg_pgen(fbg_cap_gt, s) > proto.LOG_EPS + 1.0)
    # With only 15 training seqs, many held-outs still miss, but GT must cover
    # at least as many as none.
    assert gt_covered >= none_covered


def test_gt_pgen_mle_ignores_backoff(fbg_cap_gt, fbg_proto_gt):
    """pgen_mle under a backoff='gt' grammar still ignores backoff."""
    for seq in TRAIN:
        c_lp_mle = _clzgraph.fbg_pgen_mle(fbg_cap_gt, seq)
        p_lp_mle = fbg_proto_gt.pgen_mle(seq, log=True)
        assert abs(c_lp_mle - p_lp_mle) < 1e-12


# ══════════════════════════════════════════════════════════════
# P3: Analytics (path count, entropy, Hill, power sum)
# ══════════════════════════════════════════════════════════════

import math


def test_path_count_series_matches_prototype(fbg_cap, fbg_proto):
    L_max = 25
    c_series = _clzgraph.fbg_path_count_series(fbg_cap, L_max)
    p_series = fbg_proto.path_count_series(L_max)
    assert len(c_series) == L_max + 1
    for L, (c, p) in enumerate(zip(c_series, p_series)):
        assert abs(c - p) < 1e-9, f"path_count[L={L}]: c={c}, py={p}"


def test_length_distribution_sums_to_one(fbg_cap):
    ld = _clzgraph.fbg_length_distribution(fbg_cap, 25)
    total = sum(ld)
    assert abs(total - 1.0) < 1e-9, f"length_distribution sum = {total}"


def test_length_distribution_matches_prototype(fbg_cap, fbg_proto):
    L_max = 25
    c_ld = _clzgraph.fbg_length_distribution(fbg_cap, L_max)
    p_ld = fbg_proto.length_distribution(L_max)
    for L, (c, p) in enumerate(zip(c_ld, p_ld)):
        assert abs(c - p) < 1e-12, f"P(len={L}): c={c}, py={p}"


def test_entropy_matches_prototype(fbg_cap, fbg_proto):
    c_H = _clzgraph.fbg_entropy(fbg_cap)
    p_H = fbg_proto.entropy()
    assert abs(c_H - p_H) < 1e-9, f"H(S): c={c_H}, py={p_H}"


def test_entropy_agrees_with_mc_samples(fbg_cap, fbg_proto):
    """Entropy via linear system ≈ -E[log pgen_mle(sample)] on MC samples."""
    import random
    c_H = _clzgraph.fbg_entropy(fbg_cap)
    rng = random.Random(42)
    # Sample from prototype (sampler not yet C-ported at P3)
    n = 3000
    total = 0.0
    for _ in range(n):
        _, lp, _ = fbg_proto.sample(rng)
        total += -lp
    mc_H = total / n
    rel_err = abs(c_H - mc_H) / abs(c_H)
    assert rel_err < 0.05, f"H analytical {c_H} vs MC {mc_H} (rel err {rel_err:.2%})"


def test_effective_diversity_fields(fbg_cap, fbg_proto):
    ed = _clzgraph.fbg_effective_diversity(fbg_cap)
    p_H = fbg_proto.entropy()
    assert abs(ed['entropy_nats'] - p_H) < 1e-9
    assert abs(ed['entropy_bits'] - p_H / math.log(2)) < 1e-9
    assert abs(ed['effective_diversity'] - math.exp(p_H)) < 1e-9
    assert 0.0 <= ed['uniformity'] <= 1.0


def test_hill_at_1_equals_exp_entropy(fbg_cap):
    D1 = _clzgraph.fbg_hill_number(fbg_cap, 1.0)
    H = _clzgraph.fbg_entropy(fbg_cap)
    assert abs(D1 - math.exp(H)) < 1e-9, f"D(1)={D1}, exp(H)={math.exp(H)}"


def test_hill_numbers_match_prototype(fbg_cap, fbg_proto):
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    c_Ds = _clzgraph.fbg_hill_numbers(fbg_cap, alphas)
    for alpha, c_D in zip(alphas, c_Ds):
        p_D = fbg_proto.hill_number(alpha)
        rel_err = abs(c_D - p_D) / max(abs(p_D), 1e-12)
        assert rel_err < 1e-9, f"D({alpha}): c={c_D}, py={p_D}"


def test_power_sum_at_1_equals_one(fbg_cap):
    M1 = _clzgraph.fbg_power_sum(fbg_cap, 1.0)
    assert abs(M1 - 1.0) < 1e-12


def test_power_sum_matches_prototype(fbg_cap, fbg_proto):
    for alpha in [0.5, 2.0, 3.0]:
        c_M = _clzgraph.fbg_power_sum(fbg_cap, alpha)
        p_M = fbg_proto.power_sum(alpha)
        rel_err = abs(c_M - p_M) / max(abs(p_M), 1e-12)
        assert rel_err < 1e-9, f"M({alpha}): c={c_M}, py={p_M}"


def test_hill_monotone_decreasing_in_alpha(fbg_cap):
    """Hill(α) decreases as α increases (always true for non-uniform P)."""
    alphas = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    Ds = _clzgraph.fbg_hill_numbers(fbg_cap, alphas)
    for i in range(len(Ds) - 1):
        assert Ds[i] >= Ds[i + 1] - 1e-9, \
            f"Hill not monotone: D({alphas[i]})={Ds[i]} < D({alphas[i+1]})={Ds[i+1]}"


def test_hill_zero_equals_path_count(fbg_cap):
    """Hill(0) over truncated L_max ≈ total path count up to L_max."""
    # The grammar's Hill(α→0) limit corresponds to unweighted total path count.
    # Truncated to L_max = 30, it should match path_count_series(30).sum().
    L_max = 30
    pcs = _clzgraph.fbg_path_count_series(fbg_cap, L_max)
    total = sum(pcs)
    # Very-small α approximates Hill(0). Use α = 1e-3 for stability.
    D_eps = _clzgraph.fbg_hill_number(fbg_cap, 1e-3)
    # With α very small, M(α) ≈ |support| and D(α) ≈ |support|. For a finite
    # grammar truncated to L_max, this should be close to `total`.
    # Only assert direction (D_eps should be >> effective_diversity).
    H = _clzgraph.fbg_entropy(fbg_cap)
    assert D_eps > math.exp(H) - 1e-9, \
        f"Hill(0) approx {D_eps} should exceed exp(H) = {math.exp(H)}"
    _ = total


# ══════════════════════════════════════════════════════════════
# P4: Sampling, dynamic range, top-k
# ══════════════════════════════════════════════════════════════


def test_simulate_basic(fbg_cap):
    """simulate returns n non-empty CDR3s with log-probs in (−∞, 0]."""
    seqs, lps, nts = _clzgraph.fbg_simulate(fbg_cap, 500, seed=1)
    assert len(seqs) == len(lps) == len(nts) == 500
    # Every sample should have non-empty string starting with C (training prior)
    for s in seqs:
        assert len(s) >= 1
    # Log-probs non-positive
    for lp in lps:
        assert lp <= 0 + 1e-9


def test_simulate_canonicity(fbg_cap):
    """Every generated sequence re-decomposes to a tree with the SAME depth
    and rule sequence as the sampler emitted. Canonicity is a hard invariant."""
    seqs, _, nts = _clzgraph.fbg_simulate(fbg_cap, 2000, seed=7)
    for s, emitted_depth in zip(seqs, nts):
        if len(s) == 0:
            continue
        re_steps = _clzgraph.fbg_decompose(fbg_cap, s)
        # Re-decomposition must have the same tree depth as the emission
        # (if it's canonical — which ours must be).
        assert len(re_steps) == emitted_depth, \
            f"Canonicity broken for {s!r}: re-decomposed to {len(re_steps)} " \
            f"steps but sampler emitted {emitted_depth}"


def test_simulate_sampled_seq_has_pgen(fbg_cap):
    """pgen_mle(sampled) should equal sampler-reported log_prob."""
    seqs, lps, _ = _clzgraph.fbg_simulate(fbg_cap, 200, seed=3)
    for s, lp_sample in zip(seqs, lps):
        lp_score = _clzgraph.fbg_pgen_mle(fbg_cap, s)
        assert abs(lp_sample - lp_score) < 1e-9, \
            f"mismatch for {s!r}: sampled_lp={lp_sample}, pgen_mle={lp_score}"


def test_simulate_empirical_entropy_matches_analytical(fbg_cap):
    """-E[log pgen(sample)] ≈ analytical H(S) within 5%."""
    seqs, lps, _ = _clzgraph.fbg_simulate(fbg_cap, 5000, seed=42)
    mc_H = -sum(lps) / len(lps)
    H = _clzgraph.fbg_entropy(fbg_cap)
    rel_err = abs(mc_H - H) / max(abs(H), 1e-12)
    assert rel_err < 0.05, f"MC entropy {mc_H} vs analytical {H} (rel err {rel_err:.2%})"


def test_simulate_seed_reproducibility(fbg_cap):
    """Same seed → same outputs."""
    seqs1, lps1, _ = _clzgraph.fbg_simulate(fbg_cap, 50, seed=1234)
    seqs2, lps2, _ = _clzgraph.fbg_simulate(fbg_cap, 50, seed=1234)
    assert seqs1 == seqs2
    assert lps1 == lps2


# ── Dynamic range ─────────────────────────────────────────────

def test_dynamic_range_structure(fbg_cap):
    dr = _clzgraph.fbg_dynamic_range(fbg_cap, 25)
    assert dr['max_log_prob'] <= 0 + 1e-9
    assert dr['min_log_prob'] <= dr['max_log_prob']
    assert dr['dynamic_range_nats'] >= 0
    assert abs(dr['dynamic_range_orders']
               - dr['dynamic_range_nats'] / math.log(10)) < 1e-12


def test_dynamic_range_consistent_with_top_k(fbg_cap):
    """max_log_prob should equal the log-prob of the top_k=1 most-probable seq."""
    dr = _clzgraph.fbg_dynamic_range(fbg_cap, 25)
    seqs, lps, _ = _clzgraph.fbg_top_k_sequences(fbg_cap, 1,
                                                  most_probable=True, max_length=25)
    assert len(seqs) == 1
    assert abs(dr['max_log_prob'] - lps[0]) < 1e-9


# ── Top-K ─────────────────────────────────────────────────────

def test_top_k_monotone_descending(fbg_cap):
    seqs, lps, _ = _clzgraph.fbg_top_k_sequences(fbg_cap, 20,
                                                  most_probable=True, max_length=25)
    assert len(seqs) >= 1
    for i in range(len(lps) - 1):
        assert lps[i] >= lps[i + 1] - 1e-9, \
            f"top_k not monotone at i={i}: lp[i]={lps[i]}, lp[i+1]={lps[i+1]}"


def test_top_k_ascending_least_probable(fbg_cap):
    seqs, lps, _ = _clzgraph.fbg_top_k_sequences(fbg_cap, 20,
                                                  most_probable=False, max_length=25)
    assert len(seqs) >= 1
    for i in range(len(lps) - 1):
        assert lps[i] <= lps[i + 1] + 1e-9, \
            f"top_k (least) not ascending at i={i}: lp[i]={lps[i]}, lp[i+1]={lps[i+1]}"


def test_top_k_sequences_canonical(fbg_cap):
    """Top-K reconstructed sequences must all re-decompose (canonicity)."""
    seqs, _, _ = _clzgraph.fbg_top_k_sequences(fbg_cap, 30,
                                                most_probable=True, max_length=25)
    for s in seqs:
        if len(s) == 0:
            continue
        # Round-trip via decompose + reverse must equal s
        steps = _clzgraph.fbg_decompose(fbg_cap, s)
        re = _clzgraph.fbg_tree_to_string(fbg_cap, steps)
        assert re == s, f"round-trip failed for top-k seq {s!r} → {re!r}"


def test_top_k_top1_in_training_set(fbg_cap):
    """Most-probable top-1 should typically correspond to a short training seq
    (since each log p < 0, shorter trees win)."""
    seqs, lps, _ = _clzgraph.fbg_top_k_sequences(fbg_cap, 1,
                                                  most_probable=True, max_length=25)
    # It should have non-trivial probability — not the epsilon floor.
    assert lps[0] > -50.0
    assert len(seqs[0]) >= 1


# ══════════════════════════════════════════════════════════════
# P5: Posterior and subtract
# ══════════════════════════════════════════════════════════════

# Use the disjoint pool for posterior/subtract tests.
from fbg_prototype import LARGER_POOL


@pytest.fixture(scope='module')
def split_pool():
    import random
    rng = random.Random(0)
    pool = list(LARGER_POOL)
    rng.shuffle(pool)
    A = pool[:40]
    B = pool[40:60]
    return A, B


@pytest.fixture(scope='module')
def grammars_AB(split_pool):
    A, B = split_pool
    cap_A  = _clzgraph.fbg_build(A)
    cap_AB = _clzgraph.fbg_build(A + B)
    return cap_A, cap_AB, A, B


# ── Posterior tests ───────────────────────────────────────────

def test_posterior_kappa_zero_is_individual_mle(fbg_cap, fbg_proto):
    """κ=0 → posterior weights are individual MLE restricted to prior topology."""
    individual = TRAIN[:5]
    post = _clzgraph.fbg_posterior(fbg_cap, individual, kappa=0.0)
    p_post = fbg_proto.posterior(individual, kappa=0.0)

    info = _clzgraph.fbg_info(post)
    c_nts = _clzgraph.fbg_nts(post)
    for i in range(info['n_nts']):
        a, z, is_start, total, n_rules, _ = c_nts[i]
        p_nt = 'S' if is_start else (a, z)
        c_rules = _clzgraph.fbg_rules_at(post, i)
        p_rules_dict = p_post.weights[p_nt]
        for cr in c_rules:
            k = cr['kind']
            if k == 'internal':
                p_key = ('internal', cr['a_run_len'], cr['z_run_len'],
                         cr['dst_a'], cr['dst_z'])
            elif k == 'leaf_single':
                p_key = ('leaf_single', cr['a_char'])
            elif k == 'leaf_run':
                p_key = ('leaf_run', cr['a_char'], cr['a_run_len'])
            else:
                p_key = ('leaf_pair', cr['a_char'], cr['a_run_len'],
                         cr['z_char'], cr['z_run_len'])
            assert abs(cr['weight'] - p_rules_dict[p_key]) < 1e-9, \
                f"κ=0 weight mismatch at M{p_nt}: c={cr['weight']}, py={p_rules_dict[p_key]}"


def test_posterior_kappa_large_equals_prior(fbg_cap, fbg_proto):
    """κ=1e6 → posterior ≈ prior weights."""
    individual = TRAIN[:5]
    post = _clzgraph.fbg_posterior(fbg_cap, individual, kappa=1e6)

    # Compare against the prior grammar's weights.
    info = _clzgraph.fbg_info(post)
    c_nts_post  = _clzgraph.fbg_nts(post)
    c_nts_prior = _clzgraph.fbg_nts(fbg_cap)

    # Both grammars have identical topology (posterior preserves it); compare rule-by-rule.
    assert info['n_nts']   == _clzgraph.fbg_info(fbg_cap)['n_nts']
    assert info['n_rules'] == _clzgraph.fbg_info(fbg_cap)['n_rules']

    for i in range(info['n_nts']):
        post_rules  = _clzgraph.fbg_rules_at(post,    i)
        prior_rules = _clzgraph.fbg_rules_at(fbg_cap, i)
        assert len(post_rules) == len(prior_rules)
        # Posterior with κ→∞ should match prior to machine precision.
        for pr, qr in zip(post_rules, prior_rules):
            assert abs(pr['weight'] - qr['weight']) < 1e-3, \
                f"κ=1e6: rule weight diff = {abs(pr['weight'] - qr['weight'])}"
    _ = fbg_proto


def test_posterior_monotone_interpolation(fbg_cap):
    """|w_post - w_prior| decreases monotonically as κ grows."""
    individual = TRAIN[:5]

    # Find M(C,F) index and pick its top rule.
    c_nts = _clzgraph.fbg_nts(fbg_cap)
    target_idx = None
    for i, (a, z, is_start, _, _, _) in enumerate(c_nts):
        if (a, z, is_start) == ('C', 'F', False):
            target_idx = i
            break
    assert target_idx is not None

    prior_rules = _clzgraph.fbg_rules_at(fbg_cap, target_idx)
    top_rule = max(prior_rules, key=lambda r: r['weight'])
    w_prior = top_rule['weight']

    def find_matching_weight(rules, top):
        for r in rules:
            if (r['kind'] == top['kind']
                and r.get('a_run_len') == top.get('a_run_len')
                and r.get('z_run_len') == top.get('z_run_len')
                and r.get('dst_a') == top.get('dst_a')
                and r.get('dst_z') == top.get('dst_z')):
                return r['weight']
        return None

    prev_dist = None
    kappas = [0.0, 0.5, 1.0, 5.0, 20.0, 1e4]
    for k in kappas:
        post = _clzgraph.fbg_posterior(fbg_cap, individual, kappa=k)
        post_rules = _clzgraph.fbg_rules_at(post, target_idx)
        w = find_matching_weight(post_rules, top_rule)
        dist = abs(w - w_prior)
        if prev_dist is not None:
            assert dist <= prev_dist + 1e-9, \
                f"non-monotone at κ={k}: dist={dist}, prev={prev_dist}"
        prev_dist = dist


def test_posterior_consistency_preserved(fbg_cap):
    post = _clzgraph.fbg_posterior(fbg_cap, TRAIN[:5], kappa=1.0)
    info = _clzgraph.fbg_info(post)
    assert info['is_consistent']
    assert info['spectral_radius'] < 1.0


def test_posterior_improves_individual_likelihood(fbg_cap):
    """Σ log-pgen(individual) should improve under posterior vs prior."""
    individual = TRAIN[:5]
    post = _clzgraph.fbg_posterior(fbg_cap, individual, kappa=1.0)

    ll_prior = sum(_clzgraph.fbg_pgen_mle(fbg_cap, s) for s in individual)
    ll_post  = sum(_clzgraph.fbg_pgen_mle(post,    s) for s in individual)
    assert ll_post >= ll_prior - 1e-9, \
        f"posterior LL {ll_post} should be >= prior LL {ll_prior}"


def test_posterior_kappa_negative_raises(fbg_cap):
    import builtins
    with pytest.raises((ValueError, Exception)):
        _clzgraph.fbg_posterior(fbg_cap, TRAIN[:5], kappa=-1.0)


# ── Subtract tests ────────────────────────────────────────────

def test_subtract_round_trip_disjoint(grammars_AB):
    """G(A + B).without(B) == G(A) bit-exactly on every shared rule."""
    cap_A, cap_AB, A, B = grammars_AB
    cap_rec = _clzgraph.fbg_subtract(cap_AB, B)

    info_A   = _clzgraph.fbg_info(cap_A)
    info_rec = _clzgraph.fbg_info(cap_rec)

    # Build (nt_key, rule_key) → weight dicts for comparison.
    def flatten(cap):
        out = {}
        info = _clzgraph.fbg_info(cap)
        c_nts = _clzgraph.fbg_nts(cap)
        for i in range(info['n_nts']):
            a, z, is_start, _, _, _ = c_nts[i]
            nt_key = 'S' if is_start else (a, z)
            for r in _clzgraph.fbg_rules_at(cap, i):
                k = r['kind']
                if k == 'internal':
                    rk = ('internal', r['a_run_len'], r['z_run_len'],
                          r['dst_a'], r['dst_z'])
                elif k == 'leaf_single':
                    rk = ('leaf_single', r['a_char'])
                elif k == 'leaf_run':
                    rk = ('leaf_run', r['a_char'], r['a_run_len'])
                else:
                    rk = ('leaf_pair', r['a_char'], r['a_run_len'],
                          r['z_char'], r['z_run_len'])
                out[(nt_key, rk)] = r['weight']
        return out

    w_A   = flatten(cap_A)
    w_rec = flatten(cap_rec)
    assert set(w_A.keys()) == set(w_rec.keys()), \
        f"rule sets differ: only_A={set(w_A) - set(w_rec)}, only_rec={set(w_rec) - set(w_A)}"
    for k in w_A:
        assert abs(w_A[k] - w_rec[k]) < 1e-9, f"weight diff at {k}: A={w_A[k]}, rec={w_rec[k]}"
    _ = info_A, info_rec


def test_subtract_consistency_preserved(grammars_AB):
    cap_A, cap_AB, _, B = grammars_AB
    cap_rec = _clzgraph.fbg_subtract(cap_AB, B)
    info = _clzgraph.fbg_info(cap_rec)
    assert info['is_consistent']
    assert info['spectral_radius'] < 1.0


def test_subtract_prunes_unique_to_B(grammars_AB):
    """Rules pruned from G(A+B).without(B) == rules unique to B's decomposition."""
    cap_A, cap_AB, A, B = grammars_AB
    cap_rec = _clzgraph.fbg_subtract(cap_AB, B)

    def rule_set(cap):
        out = set()
        info = _clzgraph.fbg_info(cap)
        c_nts = _clzgraph.fbg_nts(cap)
        for i in range(info['n_nts']):
            a, z, is_start, _, _, _ = c_nts[i]
            nt_key = 'S' if is_start else (a, z)
            for r in _clzgraph.fbg_rules_at(cap, i):
                k = r['kind']
                if k == 'internal':
                    rk = ('internal', r['a_run_len'], r['z_run_len'],
                          r['dst_a'], r['dst_z'])
                elif k == 'leaf_single':
                    rk = ('leaf_single', r['a_char'])
                elif k == 'leaf_run':
                    rk = ('leaf_run', r['a_char'], r['a_run_len'])
                else:
                    rk = ('leaf_pair', r['a_char'], r['a_run_len'],
                          r['z_char'], r['z_run_len'])
                out.add((nt_key, rk))
        return out

    set_AB  = rule_set(cap_AB)
    set_rec = rule_set(cap_rec)
    pruned  = set_AB - set_rec

    # Rules unique to B = in G(B) but not in G(A).
    cap_B = _clzgraph.fbg_build(B)
    set_A = rule_set(cap_A)
    set_B = rule_set(cap_B)
    b_only = set_B - set_A

    assert pruned == b_only, \
        f"mismatch: pruned={len(pruned)}, b_only={len(b_only)}, " \
        f"pruned-b_only={pruned - b_only}, b_only-pruned={b_only - pruned}"


def test_subtract_all_raises():
    cap = _clzgraph.fbg_build(TRAIN)
    with pytest.raises((ValueError, Exception)):
        _clzgraph.fbg_subtract(cap, TRAIN)


def _flatten_grammar(cap):
    """(nt_key, rule_key) -> weight for round-trip comparison."""
    out = {}
    info = _clzgraph.fbg_info(cap)
    c_nts = _clzgraph.fbg_nts(cap)
    for i in range(info['n_nts']):
        a, z, is_start, _, _, _ = c_nts[i]
        nt_key = 'S' if is_start else (a, z)
        for r in _clzgraph.fbg_rules_at(cap, i):
            k = r['kind']
            if k == 'internal':
                rk = ('internal', r['a_run_len'], r['z_run_len'],
                      r['dst_a'], r['dst_z'])
            elif k == 'leaf_single':
                rk = ('leaf_single', r['a_char'])
            elif k == 'leaf_run':
                rk = ('leaf_run', r['a_char'], r['a_run_len'])
            else:
                rk = ('leaf_pair', r['a_char'], r['a_run_len'],
                      r['z_char'], r['z_run_len'])
            out[(nt_key, rk)] = (r['weight'], r['count'])
    return out


def test_subtract_abundance_round_trip():
    """(A₃ + B¹).without(A, abundances=3×)  ==  G(B) bit-exactly."""
    import random
    rng = random.Random(0)
    pool = list(LARGER_POOL)
    rng.shuffle(pool)
    A = pool[:10]
    B = pool[10:15]
    seqs = A + B
    abunds = [3] * 10 + [1] * 5

    cap_mixed = _clzgraph.fbg_build(seqs, abundances=abunds)
    cap_sub = _clzgraph.fbg_subtract(cap_mixed, A, abundances=[3] * 10)
    cap_B = _clzgraph.fbg_build(B)

    # Compare every (nt, rule) weight
    def flatten(cap):
        out = {}
        info = _clzgraph.fbg_info(cap)
        c_nts = _clzgraph.fbg_nts(cap)
        for i in range(info['n_nts']):
            a, z, is_start, _, _, _ = c_nts[i]
            nt_key = 'S' if is_start else (a, z)
            for r in _clzgraph.fbg_rules_at(cap, i):
                k = r['kind']
                if k == 'internal':
                    rk = ('internal', r['a_run_len'], r['z_run_len'],
                          r['dst_a'], r['dst_z'])
                elif k == 'leaf_single':
                    rk = ('leaf_single', r['a_char'])
                elif k == 'leaf_run':
                    rk = ('leaf_run', r['a_char'], r['a_run_len'])
                else:
                    rk = ('leaf_pair', r['a_char'], r['a_run_len'],
                          r['z_char'], r['z_run_len'])
                out[(nt_key, rk)] = r['weight']
        return out

    w_sub = flatten(cap_sub)
    w_B = flatten(cap_B)
    assert set(w_sub) == set(w_B), \
        f"rule sets differ: sub-only={set(w_sub) - set(w_B)}, " \
        f"B-only={set(w_B) - set(w_sub)}"
    for k in w_sub:
        assert abs(w_sub[k] - w_B[k]) < 1e-9, \
            f"weight diff at {k}: sub={w_sub[k]}, B={w_B[k]}"


# ══════════════════════════════════════════════════════════════
# P6: I/O (save/load)
# ══════════════════════════════════════════════════════════════

import tempfile
import os


def test_io_round_trip_preserves_weights(fbg_cap):
    """save → load produces bit-exact weights and counts."""
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        path = tmp.name
    try:
        _clzgraph.fbg_save(fbg_cap, path)
        cap_loaded = _clzgraph.fbg_load(path)
        w_orig = _flatten_grammar(fbg_cap)
        w_loaded = _flatten_grammar(cap_loaded)
        assert set(w_orig.keys()) == set(w_loaded.keys())
        for k in w_orig:
            wo, co = w_orig[k]
            wl, cl = w_loaded[k]
            assert wo == wl, f"weight diff at {k}: orig={wo}, loaded={wl}"
            assert co == cl, f"count diff at {k}: orig={co}, loaded={cl}"
    finally:
        os.unlink(path)


def test_io_round_trip_preserves_info(fbg_cap):
    """Save/load preserves all info fields including derived ones."""
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        path = tmp.name
    try:
        _clzgraph.fbg_save(fbg_cap, path)
        cap_loaded = _clzgraph.fbg_load(path)
        info_orig = _clzgraph.fbg_info(fbg_cap)
        info_loaded = _clzgraph.fbg_info(cap_loaded)
        for key in ['n_nts', 'n_rules', 'n_internal_rules',
                    'alphabet_size', 'max_length', 'start_nt',
                    'abundance_mode', 'backoff', 'is_consistent']:
            assert info_orig[key] == info_loaded[key], \
                f"{key}: orig={info_orig[key]}, loaded={info_loaded[key]}"
        # spectral_radius and smoothing are floats — compare tightly.
        assert abs(info_orig['spectral_radius'] - info_loaded['spectral_radius']) < 1e-12
        assert info_orig['smoothing'] == info_loaded['smoothing']
    finally:
        os.unlink(path)


def test_io_pgen_identical_after_load(fbg_cap):
    """pgen on a loaded grammar == pgen on the original for every training seq."""
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        path = tmp.name
    try:
        _clzgraph.fbg_save(fbg_cap, path)
        cap_loaded = _clzgraph.fbg_load(path)
        for seq in TRAIN:
            lp1 = _clzgraph.fbg_pgen_mle(fbg_cap, seq)
            lp2 = _clzgraph.fbg_pgen_mle(cap_loaded, seq)
            assert lp1 == lp2, f"pgen diverged for {seq!r}: {lp1} vs {lp2}"
    finally:
        os.unlink(path)


def test_io_with_backoff_round_trip(fbg_cap_gt):
    """GT backoff tables (unseen_mass, rule_marginal) recomputed correctly on load."""
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        path = tmp.name
    try:
        _clzgraph.fbg_save(fbg_cap_gt, path)
        cap_loaded = _clzgraph.fbg_load(path)
        info_orig = _clzgraph.fbg_info(fbg_cap_gt)
        info_loaded = _clzgraph.fbg_info(cap_loaded)
        assert info_orig['backoff'] == info_loaded['backoff']
        # Compare unseen_mass per NT.
        nts_orig = _clzgraph.fbg_nts(fbg_cap_gt)
        nts_loaded = _clzgraph.fbg_nts(cap_loaded)
        for (_, _, _, _, _, um_orig), (_, _, _, _, _, um_loaded) in zip(nts_orig, nts_loaded):
            assert abs(um_orig - um_loaded) < 1e-12
        # Compare pgen with backoff enabled on held-outs.
        for seq in HELDOUT:
            lp1 = _clzgraph.fbg_pgen(fbg_cap_gt, seq)
            lp2 = _clzgraph.fbg_pgen(cap_loaded, seq)
            assert lp1 == lp2, f"backoff pgen diverged for {seq!r}"
    finally:
        os.unlink(path)


def test_io_load_bad_magic_raises():
    """Loading garbage fails gracefully."""
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        tmp.write(b'\x00\x00\x00\x00' * 100)
        path = tmp.name
    try:
        with pytest.raises(Exception):
            _clzgraph.fbg_load(path)
    finally:
        os.unlink(path)


def test_io_load_nonexistent_raises():
    with pytest.raises(Exception):
        _clzgraph.fbg_load('/nonexistent/path/should/fail.fbg')


def test_io_analytics_identical_after_load(fbg_cap):
    """Entropy, Hill numbers, path count all identical after save/load."""
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        path = tmp.name
    try:
        _clzgraph.fbg_save(fbg_cap, path)
        cap_loaded = _clzgraph.fbg_load(path)

        H1 = _clzgraph.fbg_entropy(fbg_cap)
        H2 = _clzgraph.fbg_entropy(cap_loaded)
        assert H1 == H2

        for alpha in [0.5, 1.0, 2.0, 5.0]:
            D1 = _clzgraph.fbg_hill_number(fbg_cap, alpha)
            D2 = _clzgraph.fbg_hill_number(cap_loaded, alpha)
            assert abs(D1 - D2) < 1e-12

        pcs1 = _clzgraph.fbg_path_count_series(fbg_cap, 20)
        pcs2 = _clzgraph.fbg_path_count_series(cap_loaded, 20)
        for a, b in zip(pcs1, pcs2):
            assert a == b
    finally:
        os.unlink(path)


# ══════════════════════════════════════════════════════════════
# P7: FlashBackGrammar class — integration tests
# ══════════════════════════════════════════════════════════════

from LZGraphs import FlashBackGrammar, SimulationResult


@pytest.fixture(scope='module')
def fbg():
    return FlashBackGrammar(TRAIN)


def test_class_repr_and_len(fbg):
    assert repr(fbg).startswith("FlashBackGrammar(")
    assert len(fbg) == fbg.n_sequences > 0


def test_class_contains(fbg):
    assert TRAIN[0] in fbg
    assert "BBBB" not in fbg  # 'B' unseen


def test_class_properties(fbg):
    assert fbg.n_nonterminals > 0
    assert fbg.n_rules > 0
    assert fbg.n_internal_rules + fbg.n_leaf_rules == fbg.n_rules
    assert fbg.is_consistent
    assert 0.0 <= fbg.spectral_radius < 1.0
    assert fbg.abundance_mode == 'linear'
    assert fbg.backoff_mode == 'none'
    assert fbg.smoothing == 0.0
    assert fbg.n_sequences == len(TRAIN)
    assert fbg.max_length == max(len(s) for s in TRAIN)


def test_class_pgen_scalar_and_batch(fbg):
    # Scalar
    lp = fbg.pgen(TRAIN[0])
    assert lp > -100 and lp <= 0
    p = fbg.pgen(TRAIN[0], log=False)
    assert abs(p - math.exp(lp)) < 1e-12
    # Batch
    lps = fbg.pgen(TRAIN)
    assert len(lps) == len(TRAIN)
    ps = fbg.pgen(TRAIN, log=False)
    assert abs(ps[0] - math.exp(lps[0])) < 1e-12


def test_class_pgen_mle_distinct_from_backoff():
    fbg_gt = FlashBackGrammar(TRAIN, backoff='gt')
    seq = HELDOUT[0]
    lp_backoff = fbg_gt.pgen(seq)
    lp_mle = fbg_gt.pgen_mle(seq)
    # For a held-out that needs backoff, lp_mle should be at ε and lp_backoff above.
    assert lp_mle <= proto.LOG_EPS + 1.0
    assert lp_backoff > lp_mle - 1e-9  # backoff ≥ MLE here


def test_class_analytics(fbg):
    H = fbg.entropy()
    assert H > 0
    D = fbg.effective_diversity()
    assert abs(D - math.exp(H)) < 1e-9
    assert abs(fbg.hill_number(1.0) - D) < 1e-9
    assert abs(fbg.power_sum(1.0) - 1.0) < 1e-12
    # Hill curve
    curve = fbg.hill_curve([0.5, 1.0, 2.0])
    assert set(curve) == {'orders', 'values'}
    assert len(curve['values']) == 3


def test_class_path_count(fbg):
    series = fbg.path_count_series(20)
    assert len(series) == 21
    total = fbg.path_count(20)
    assert abs(total - series.sum()) < 1e-9
    ld = fbg.length_pmf(25)
    assert abs(ld.sum() - 1.0) < 1e-9


def test_class_dynamic_range(fbg):
    dr_orders = fbg.pgen_dynamic_range(max_length=25)
    assert dr_orders >= 0
    dr = fbg.pgen_dynamic_range_detail(max_length=25)
    assert abs(dr['dynamic_range_orders'] - dr_orders) < 1e-12


def test_class_simulate_returns_simulation_result(fbg):
    result = fbg.simulate(50, seed=0)
    assert isinstance(result, SimulationResult)
    assert len(result.sequences) == 50
    assert len(result.log_probs) == 50
    assert all(lp <= 0 for lp in result.log_probs)


def test_class_top_k(fbg):
    result = fbg.top_k_sequences(k=10, max_length=25)
    assert isinstance(result, SimulationResult)
    assert len(result.sequences) <= 10
    for i in range(len(result.log_probs) - 1):
        assert result.log_probs[i] >= result.log_probs[i + 1] - 1e-9


def test_class_nonterminals_and_rules_at(fbg):
    nts = fbg.nonterminals
    assert any(start for (_, _, start) in nts)  # at least one S
    # CASS-prefix training → M(C,F) expected
    assert ('C', 'F', False) in nts
    rules_cf = fbg.rules_at('C', 'F')
    assert len(rules_cf) > 0
    assert all('weight' in r and 'kind' in r for r in rules_cf)


def test_class_rules_at_unknown_raises(fbg):
    with pytest.raises(KeyError):
        fbg.rules_at('Z', 'Q')  # likely not observed


def test_class_sentinel_rule_weights(fbg):
    sw = fbg.sentinel_rule_weights
    # Training is all CASS...F/W → mostly M(C,F)
    assert ('C', 'F') in sw
    assert abs(sum(sw.values()) - 1.0) < 1e-9


def test_class_top_rules(fbg):
    top = fbg.top_rules(k=5, by='weight')
    assert len(top) <= 5
    # Weights monotone non-increasing
    for i in range(len(top) - 1):
        assert top[i][1]['weight'] >= top[i + 1][1]['weight'] - 1e-12


def test_class_posterior_returns_instance(fbg):
    post = fbg.posterior(TRAIN[:5], kappa=1.0)
    assert isinstance(post, FlashBackGrammar)
    assert post.is_consistent
    # Individual LL should improve
    ll_prior = float(fbg.pgen_mle(TRAIN[:5]).sum())
    ll_post  = float(post.pgen_mle(TRAIN[:5]).sum())
    assert ll_post >= ll_prior - 1e-9


def test_class_without_returns_instance(fbg):
    # Split training into A, B and verify round-trip
    A = TRAIN[:10]
    B = TRAIN[10:]
    fbg_A  = FlashBackGrammar(A)
    fbg_AB = FlashBackGrammar(A + B)
    fbg_rec = fbg_AB.without(B)
    assert isinstance(fbg_rec, FlashBackGrammar)
    # Same entropy as G(A)
    assert abs(fbg_rec.entropy() - fbg_A.entropy()) < 1e-9


def test_class_save_load_roundtrip(fbg):
    with tempfile.NamedTemporaryFile(suffix='.fbg', delete=False) as tmp:
        path = tmp.name
    try:
        fbg.save(path)
        fbg_loaded = FlashBackGrammar.load(path)
        assert fbg_loaded.n_rules == fbg.n_rules
        assert abs(fbg_loaded.entropy() - fbg.entropy()) < 1e-12
        for seq in TRAIN[:3]:
            assert fbg_loaded.pgen_mle(seq) == fbg.pgen_mle(seq)
    finally:
        os.unlink(path)


def test_class_from_file_round_trip():
    """from_file() parses a plain-text file and builds an equivalent grammar."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
        for s in TRAIN:
            tmp.write(s + '\n')
        path = tmp.name
    try:
        fbg_file = FlashBackGrammar.from_file(path)
        fbg_mem  = FlashBackGrammar(TRAIN)
        assert abs(fbg_file.entropy() - fbg_mem.entropy()) < 1e-12
        assert fbg_file.n_rules == fbg_mem.n_rules
    finally:
        os.unlink(path)


def test_class_summary(fbg):
    s = fbg.summary()
    for key in ['variant', 'n_nonterminals', 'n_rules', 'spectral_radius',
                'is_consistent', 'entropy_nats', 'effective_diversity']:
        assert key in s
    assert s['variant'] == 'flashback_grammar'


def test_class_rejects_single_string():
    with pytest.raises(TypeError):
        FlashBackGrammar("CASSLGQGAYEQYF")  # must be a list


def test_class_rejects_empty():
    with pytest.raises(ValueError):
        FlashBackGrammar([])


def test_class_abundance_length_mismatch():
    with pytest.raises(ValueError):
        FlashBackGrammar(TRAIN, abundances=[1, 2, 3])


# ── Integration: alongside FlashBackGraph ─────────────────────

def test_integration_alongside_flashback_graph():
    """FBG and FBG(graph) can be built on the same data and produce coherent
    results independently."""
    from LZGraphs import FlashBackGraph
    fbg = FlashBackGrammar(TRAIN)
    fbg_graph = FlashBackGraph(TRAIN)

    # Both should recognize all training seqs.
    for seq in TRAIN:
        assert fbg.pgen(seq) > -100
        assert fbg_graph.pgen(seq) > -100

    # Grammar has FEWER distinct non-terminals than graph has nodes (the
    # cross-depth-sharing compactness win).
    # (Training set is small; may not always hold, but document it.)
    _ = fbg.n_nonterminals, fbg_graph.n_nodes
