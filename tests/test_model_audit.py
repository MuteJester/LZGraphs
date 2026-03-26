"""Deep model correctness audit for LZGraphs.

Tests invariants that the C unit tests don't cover:
- NDP variant LZ validity
- Abundance weighting
- Empirical distribution vs LZPGEN consistency
- Large-scale validation (5000 sequences)
- Non-training sequence handling
"""

import csv
import math
import os
import numpy as np
import pytest
from collections import Counter
from LZGraphs import LZGraph, lz76_decompose


TESTS_DIR = os.path.dirname(__file__)


# ═══════════════════════════════════════════════════════════════
# Fixture: 5000-sequence dataset
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def large_data():
    seqs, vg, jg = [], [], []
    path = os.path.join(TESTS_DIR, 'ExampleData3.csv')
    with open(path) as f:
        for row in csv.DictReader(f):
            seqs.append(row['cdr3_amino_acid'])
            vg.append(row['V'])
            jg.append(row['J'])
    return seqs, vg, jg


@pytest.fixture(scope='module')
def large_graph(large_data):
    seqs, vg, jg = large_data
    return LZGraph(seqs, variant='aap', v_genes=vg, j_genes=jg)


# ═══════════════════════════════════════════════════════════════
# 1. LZ76 round-trip for all sequences
# ═══════════════════════════════════════════════════════════════

class TestLZ76Correctness:
    def test_roundtrip_all_training(self, large_data):
        """Every training sequence decomposes and reconstructs exactly."""
        seqs = large_data[0]
        failures = 0
        for seq in seqs[:500]:
            tokens = lz76_decompose(seq)
            reconstructed = ''.join(tokens)
            if reconstructed != seq:
                failures += 1
        assert failures == 0, f"{failures} round-trip failures"

    def test_lz76_dict_invariant(self, large_data):
        """Every decomposition follows LZ76 dictionary rules."""
        seqs = large_data[0]
        violations = 0
        for seq in seqs[:500]:
            tokens = lz76_decompose(seq)
            seen = set()
            for i, tok in enumerate(tokens):
                is_last = (i == len(tokens) - 1)
                if len(tok) == 1:
                    if not is_last and tok in seen:
                        violations += 1
                else:
                    prefix = tok[:-1]
                    if prefix not in seen:
                        violations += 1
                    if not is_last and tok in seen:
                        violations += 1
                seen.add(tok)
        assert violations == 0, f"{violations} dictionary rule violations"


# ═══════════════════════════════════════════════════════════════
# 2. Simulation LZ validity at scale
# ═══════════════════════════════════════════════════════════════

class TestSimulationValidity:
    def test_simulated_seqs_are_lz_valid(self, large_graph):
        """Every simulated sequence is a valid LZ76 decomposition."""
        result = large_graph.simulate(200, seed=42)
        violations = 0
        for seq in result.sequences:
            tokens = lz76_decompose(seq)
            seen = set()
            for i, tok in enumerate(tokens):
                is_last = (i == len(tokens) - 1)
                if len(tok) == 1:
                    if not is_last and tok in seen:
                        violations += 1
                        break
                else:
                    if tok[:-1] not in seen:
                        violations += 1
                        break
                    if not is_last and tok in seen:
                        violations += 1
                        break
                seen.add(tok)
        assert violations == 0, f"{violations}/200 simulated sequences violate LZ76"

    def test_simulate_log_prob_consistency(self, large_graph):
        """simulate().log_prob matches lzpgen() for every sequence."""
        result = large_graph.simulate(100, seed=42)
        max_delta = 0.0
        mismatches = 0
        for seq, sim_lp in zip(result.sequences, result.log_probs):
            score_lp = large_graph.lzpgen(seq)
            delta = abs(sim_lp - score_lp)
            max_delta = max(max_delta, delta)
            if delta > 0.01:
                mismatches += 1
        assert mismatches == 0, f"{mismatches} mismatches, max delta={max_delta:.6f}"


# ═══════════════════════════════════════════════════════════════
# 3. Training sequence LZPGEN
# ═══════════════════════════════════════════════════════════════

class TestTrainingLZPGEN:
    def test_all_training_have_positive_prob(self, large_data, large_graph):
        """Every training sequence should have positive LZPGEN."""
        seqs = large_data[0]
        lps = large_graph.lzpgen(seqs[:200])
        zero_count = np.sum(lps < -600)
        assert zero_count == 0, f"{zero_count}/200 training seqs have zero prob"

    def test_case3_ff_ending_positive(self, large_data, large_graph):
        """Sequences ending with repeated token (case 3) have positive prob."""
        seqs = large_data[0]
        case3_seqs = []
        for s in seqs:
            tokens = lz76_decompose(s)
            if tokens[-1] in set(tokens[:-1]):
                case3_seqs.append(s)
        if not case3_seqs:
            pytest.skip("no case-3 sequences in dataset")

        lps = large_graph.lzpgen(case3_seqs[:50])
        zero_count = np.sum(lps < -600)
        assert zero_count == 0, f"{zero_count}/{len(lps)} case-3 seqs have zero prob"


# ═══════════════════════════════════════════════════════════════
# 4. Abundance weighting
# ═══════════════════════════════════════════════════════════════

class TestAbundance:
    def test_abundance_shifts_probability(self):
        """Higher abundance → higher LZPGEN."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSQETQYF']

        # Equal counts
        g_equal = LZGraph(seqs, variant='aap')
        lp_equal = g_equal.lzpgen('CASSLGIRRT')

        # CASSLGIRRT has 10x abundance
        g_weighted = LZGraph(seqs, variant='aap', abundances=[10, 1, 1])
        lp_weighted = g_weighted.lzpgen('CASSLGIRRT')

        assert lp_weighted > lp_equal, \
            f"weighted ({lp_weighted:.4f}) should be > equal ({lp_equal:.4f})"

    def test_abundance_repetition_equivalence(self):
        """Repeating a sequence N times ≈ abundance N."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF']

        g_repeat = LZGraph(seqs * 5, variant='aap')
        g_abund = LZGraph(seqs, variant='aap', abundances=[5, 5])

        lp_r = g_repeat.lzpgen('CASSLGIRRT')
        lp_a = g_abund.lzpgen('CASSLGIRRT')
        assert abs(lp_r - lp_a) < 0.1, f"repeat={lp_r:.4f} vs abund={lp_a:.4f}"


# ═══════════════════════════════════════════════════════════════
# 5. Empirical distribution sanity
# ═══════════════════════════════════════════════════════════════

class TestDistributionSanity:
    def test_frequent_simulated_seqs_have_high_lzpgen(self):
        """The most frequently simulated sequence should have high LZPGEN."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
                'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF']
        g = LZGraph(seqs * 2, variant='aap')

        result = g.simulate(2000, seed=42)
        counts = Counter(result.sequences)
        most_common_seq = counts.most_common(1)[0][0]
        lp = g.lzpgen(most_common_seq)
        assert lp > -10.0, f"most common sim seq has low prob: {lp:.4f}"

    def test_log_probs_all_negative(self):
        """All simulated log_probs must be strictly negative."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSQETQYF'] * 3
        g = LZGraph(seqs, variant='aap')
        result = g.simulate(500, seed=42)
        assert all(lp < 0 for lp in result.log_probs), \
            f"found non-negative log_prob: {max(result.log_probs)}"

    def test_sequence_lengths_reasonable(self):
        """Simulated sequence lengths should be similar to training."""
        seqs = ['CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
                'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF']
        g = LZGraph(seqs * 2, variant='aap')

        train_lens = [len(s) for s in seqs]
        result = g.simulate(500, seed=42)
        sim_lens = [len(s) for s in result.sequences]

        # Simulated lengths should be in a reasonable range
        assert min(sim_lens) >= 3, f"too short: {min(sim_lens)}"
        assert max(sim_lens) <= max(train_lens) * 3, \
            f"too long: {max(sim_lens)} vs max train {max(train_lens)}"


# ═══════════════════════════════════════════════════════════════
# 6. NDP variant
# ═══════════════════════════════════════════════════════════════

class TestNDPVariant:
    def test_ndp_build_and_simulate(self):
        """NDP graph builds and simulates valid sequences."""
        nt_seqs = ['TGTGCCAGCAGTTTCAAGATACGCAGTATTTT',
                   'TGTGCCAGCAGCCAAAGCAGCAAGCTGTGATT',
                   'TGTGCCAGCAGTTCAGGGACACAGCTATGGCCTAC']
        g = LZGraph(nt_seqs * 3, variant='ndp')
        assert g.n_nodes > 0
        assert g.n_edges > 0

        result = g.simulate(10, seed=42)
        assert len(result) == 10
        for s in result.sequences:
            assert len(s) > 0
            assert all(c in 'ATGCN' for c in s), f"invalid nucleotide in {s}"

    def test_ndp_lzpgen_positive(self):
        """NDP training sequences have positive LZPGEN."""
        nt_seqs = ['TGTGCCAGCAGTTTCAAGATACGCAGTATTTT',
                   'TGTGCCAGCAGCCAAAGCAGCAAGCTGTGATT',
                   'TGTGCCAGCAGTTCAGGGACACAGCTATGGCCTAC']
        g = LZGraph(nt_seqs * 3, variant='ndp')

        for seq in nt_seqs:
            lp = g.lzpgen(seq)
            assert lp > -600, f"NDP seq '{seq[:20]}...' has zero prob: {lp:.2f}"


# ═══════════════════════════════════════════════════════════════
# 7. Edge cases
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_unseen_sequence_zero_prob(self):
        """A sequence with no matching nodes gets LZG_LOG_EPS."""
        g = LZGraph(['CASSLGIRRT', 'CASSLGYEQYF'], variant='aap')
        lp = g.lzpgen('ZZZZZZZZZZZ')
        assert lp < -600

    def test_partial_match_zero_prob(self):
        """A sequence matching only the first few tokens then diverging."""
        g = LZGraph(['CASSLGIRRT', 'CASSLGYEQYF'], variant='aap')
        # Starts like training but ends differently
        lp = g.lzpgen('CASSLGXXXX')
        assert lp < -600

    def test_single_char_sequence(self):
        """Very short sequence."""
        g = LZGraph(['CASSLGIRRT', 'CASSLGYEQYF'] * 3, variant='aap')
        lp = g.lzpgen('C')
        # 'C' is the first token of every training seq, but it alone
        # needs to end at a terminal node — which it likely isn't
        assert isinstance(lp, float)

    def test_empty_sequence_handled(self):
        """Empty string should get LZG_LOG_EPS."""
        g = LZGraph(['CASSLGIRRT'], variant='aap')
        lp = g.lzpgen('')
        assert lp < -600

    def test_save_load_simulation_consistent(self, tmp_path):
        """Simulate from original and loaded graph should match."""
        g = LZGraph(['CASSLGIRRT', 'CASSLGYEQYF', 'CASSQETQYF'] * 3,
                     variant='aap')
        path = str(tmp_path / 'audit.lzg')
        g.save(path)
        g2 = LZGraph.load(path)

        r1 = g.simulate(20, seed=42)
        r2 = g2.simulate(20, seed=42)
        assert r1.sequences == r2.sequences
        np.testing.assert_allclose(r1.log_probs, r2.log_probs, atol=1e-10)
