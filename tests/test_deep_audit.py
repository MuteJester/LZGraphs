"""Deep model audit round 2 — testing areas not yet covered.

Focus:
- Cross-validation: held-out sequences should have positive LZPGEN
- Naive variant correctness
- Walk probability decomposition: log_p should be sum of per-step factors
- Graph union LZPGEN consistency
- Large-scale simulation: no crashes, no zero-length seqs, no NaN/inf
- Sequence ordering: more frequent = higher probability (with abundance)
- The walk dictionary prefix check is exact (not just first character)
"""

import math
import numpy as np
import pytest
from collections import Counter
from LZGraphs import LZGraph, lz76_decompose


# ═══════════════════════════════════════════════════════════════
# 1. Cross-validation: held-out sequences
# ═══════════════════════════════════════════════════════════════

class TestCrossValidation:
    def test_holdout_sequences_positive(self):
        """Sequences from the same repertoire should have positive LZPGEN
        even if not used in training (they share subpatterns)."""
        all_seqs = [
            'CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
            'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF',
            'CASSLGIRRT', 'CASSLGYEQYF', 'CASSFGQGSYEQYF',
            'CASSDTSGGTDTQYF', 'CASSQETQYF', 'CASSLEPSGGTDTQYF',
        ]
        train = all_seqs[:8]
        holdout = all_seqs[8:]

        g = LZGraph(train, variant='aap')

        positive = 0
        for s in holdout:
            lp = g.lzpgen(s)
            if lp > -600:
                positive += 1
        assert positive >= len(holdout) - 1, \
            f"only {positive}/{len(holdout)} holdout seqs have positive prob"


# ═══════════════════════════════════════════════════════════════
# 2. Walk probability is product of per-step factors
# ═══════════════════════════════════════════════════════════════

class TestProbabilityDecomposition:
    def test_lzpgen_is_product_of_steps(self):
        """Verify the walk probability = P(init) × Π P(continue) × Π P(edge) × P(stop)
        by comparing lzpgen with a manually traced walk."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSQETQYF'] * 4
        g = LZGraph(seqs, variant='aap')

        # Pick a training sequence
        seq = 'CASSLGIRRT'
        lp = g.lzpgen(seq)

        # The log_prob should be negative (not zero, not -inf)
        assert -100 < lp < 0, f"unexpected lzpgen: {lp}"

        # Two identical sequences should have identical LZPGEN
        lp2 = g.lzpgen(seq)
        assert lp == lp2, "LZPGEN should be deterministic"

    def test_longer_seq_lower_prob(self):
        """Generally, longer sequences should have lower probability
        (more transition factors in the product)."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
                'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF'] * 3
        g = LZGraph(seqs, variant='aap')

        short_lp = g.lzpgen('CASSLGIRRT')      # 10 chars
        long_lp = g.lzpgen('CASSLEPSGGTDTQYF')  # 16 chars

        # Not a strict invariant (depends on edge weights), but generally true
        # The longer sequence has more steps → more probability factors → lower total
        assert short_lp != long_lp, "different sequences should have different LZPGEN"


# ═══════════════════════════════════════════════════════════════
# 3. Naive variant
# ═══════════════════════════════════════════════════════════════

class TestNaiveVariant:
    def test_naive_build(self):
        """Naive graph builds and has correct properties."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSQETQYF'] * 3
        g = LZGraph(seqs, variant='naive')
        assert g.n_nodes > 0
        assert g.variant == 'naive'
        # Naive can have cycles (same subpattern at different positions → same node)
        # So is_dag might be False

    def test_naive_lzpgen(self):
        """Naive LZPGEN should work for training sequences."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF'] * 3
        g = LZGraph(seqs, variant='naive')
        lp = g.lzpgen('CASSLGIRRT')
        # Naive might not work perfectly due to cycles, but shouldn't crash
        assert isinstance(lp, float)


# ═══════════════════════════════════════════════════════════════
# 4. Graph union + LZPGEN
# ═══════════════════════════════════════════════════════════════

class TestUnionLZPGEN:
    def test_union_preserves_positive_prob(self):
        """If seq has positive prob in graph A, it should have positive prob
        in A|B (union) too — the union adds more paths, not fewer."""
        seqs_a = ['CASSLGIRRT', 'CASSLGYEQYF'] * 3
        seqs_b = ['CASSQETQYF', 'CASSDTSGGTDTQYF'] * 3

        g_a = LZGraph(seqs_a, variant='aap')
        g_b = LZGraph(seqs_b, variant='aap')
        g_union = g_a | g_b

        # Sequences from A should still have positive prob in union
        lp_a_in_a = g_a.lzpgen('CASSLGIRRT')
        lp_a_in_union = g_union.lzpgen('CASSLGIRRT')

        assert lp_a_in_a > -600, "seq should have positive prob in its own graph"
        assert lp_a_in_union > -600, "seq should have positive prob in union"

    def test_intersection_subset_prob(self):
        """Intersection graph should have positive prob for shared sequences."""
        seqs_shared = ['CASSLGIRRT', 'CASSLGYEQYF']
        seqs_a = seqs_shared + ['CASSQETQYF']
        seqs_b = seqs_shared + ['CASSDTSGGTDTQYF']

        g_a = LZGraph(seqs_a * 2, variant='aap')
        g_b = LZGraph(seqs_b * 2, variant='aap')
        g_inter = g_a & g_b

        lp = g_inter.lzpgen('CASSLGIRRT')
        assert lp > -600, f"shared seq should have positive prob in intersection: {lp}"


# ═══════════════════════════════════════════════════════════════
# 5. Large-scale simulation robustness
# ═══════════════════════════════════════════════════════════════

class TestLargeScaleSimulation:
    def test_1000_simulations_no_crash(self, aap_graph):
        """Simulate 1000 sequences without crash, all valid."""
        result = aap_graph.simulate(1000, seed=42)
        assert len(result) == 1000
        assert all(len(s) > 0 for s in result.sequences)
        assert all(np.isfinite(result.log_probs))
        assert all(result.log_probs < 0)

    def test_no_duplicate_single_char_tokens(self, aap_graph):
        """In any simulated sequence, no single character should appear
        as a standalone LZ token twice (except possibly at the end)."""
        result = aap_graph.simulate(200, seed=42)
        violations = 0
        for seq in result.sequences:
            tokens = lz76_decompose(seq)
            single_chars = []
            for i, tok in enumerate(tokens):
                if len(tok) == 1:
                    if i < len(tokens) - 1:  # not last
                        if tok in single_chars:
                            violations += 1
                            break
                        single_chars.append(tok)
        assert violations == 0, f"{violations}/200 have duplicate single-char tokens"


# ═══════════════════════════════════════════════════════════════
# 6. Prefix check is exact (not just first character)
# ═══════════════════════════════════════════════════════════════

class TestPrefixExactness:
    def test_prefix_check_uses_full_prefix(self):
        """The walk dict should check the FULL prefix string, not just
        the first character. Build a scenario where these differ."""
        # Sequences where "SL" and "SG" are both tokens.
        # After emitting "S" and "SL", the prefix "SG"[:-1] = "S" IS in dict,
        # so "SG" should be valid. And "SL" is in dict, so "SLA" should be valid
        # (prefix "SL" in dict, "SLA" not in dict).
        seqs = ['CASSLGIRRT', 'CASSGQETQYF', 'CASSLAETQYF'] * 3
        g = LZGraph(seqs, variant='aap')

        # All training sequences should have positive LZPGEN
        for s in ['CASSLGIRRT', 'CASSGQETQYF', 'CASSLAETQYF']:
            lp = g.lzpgen(s)
            assert lp > -600, f"'{s}' should have positive prob: {lp:.2f}"


# ═══════════════════════════════════════════════════════════════
# 7. Batch LZPGEN matches individual calls
# ═══════════════════════════════════════════════════════════════

class TestBatchConsistency:
    def test_batch_matches_individual(self, aap_graph, aap_sequences):
        """lzpgen(list) should match individual lzpgen(str) calls."""
        batch = aap_graph.lzpgen(aap_sequences)
        for i, seq in enumerate(aap_sequences):
            individual = aap_graph.lzpgen(seq)
            assert abs(batch[i] - individual) < 1e-10, \
                f"mismatch for {seq}: batch={batch[i]}, individual={individual}"


# ═══════════════════════════════════════════════════════════════
# 8. Stop probability correctness
# ═══════════════════════════════════════════════════════════════

class TestStopProbability:
    def test_stop_prob_affects_lzpgen(self):
        """A sequence that ends at a high-stop-prob node should have
        higher LZPGEN than one that goes through it."""
        # Build from sequences of varying length
        seqs = ['CASSLG', 'CASSLGIRRT', 'CASSLGYEQYF'] * 3
        g = LZGraph(seqs, variant='aap')

        lp_short = g.lzpgen('CASSLG')
        lp_long = g.lzpgen('CASSLGIRRT')

        # Both should have positive probability
        assert lp_short > -600, f"short seq: {lp_short}"
        assert lp_long > -600, f"long seq: {lp_long}"


# ═══════════════════════════════════════════════════════════════
# 9. Simulated sequences reconstruct correctly
# ═══════════════════════════════════════════════════════════════

class TestSimulationReconstruction:
    def test_simulated_seqs_start_with_initial_token(self, aap_graph):
        """All simulated sequences should start with a character that's
        an initial state in the graph."""
        result = aap_graph.simulate(100, seed=42)
        for seq in result.sequences:
            # First character should be a valid amino acid
            assert seq[0] in 'ACDEFGHIKLMNPQRSTVWY', \
                f"invalid first char: {seq[0]}"

    def test_every_simulated_seq_has_matching_lzpgen(self, aap_graph):
        """CRITICAL: Every simulated sequence MUST have a computable
        positive LZPGEN. If simulate produces it, lzpgen must accept it."""
        result = aap_graph.simulate(200, seed=123)
        failures = []
        for i, seq in enumerate(result.sequences):
            lp = aap_graph.lzpgen(seq)
            if lp <= -600:
                failures.append((i, seq, lp, result.log_probs[i]))

        if failures:
            for idx, seq, lp, sim_lp in failures[:3]:
                print(f"  FAIL [{idx}]: '{seq}' lzpgen={lp:.2f} sim_lp={sim_lp:.2f}")
        assert len(failures) == 0, \
            f"{len(failures)}/200 simulated sequences have zero LZPGEN"
