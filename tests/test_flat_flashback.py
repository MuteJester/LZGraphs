"""Tests for FlattenedFlashBackGraph: the bilateral scan without run compression.

Every analytic result is checked against brute-force enumeration of the
fixture graph's entire support, not against a golden number, since that is
the only check that would catch a wrong recursion.
"""

import math

import pytest

from LZGraphs import (
    FlashBackGraph,
    FlattenedFlashBackGraph,
    NaiveGraph,
    flat_decompose,
)
from LZGraphs._constants import LOG_EPS_AT_FLOOR, LOG_EPS_THRESHOLD

SEQS = [
    'CASSLGIRRT', 'CASSLGYEQYF', 'CASSQETQYF',
    'CASSDTSGGTDTQYF', 'CASSAYFF', 'CASSLGQ',
]


@pytest.fixture(scope='module')
def graph():
    return FlattenedFlashBackGraph(SEQS)


def reverse_tokens(tokens):
    """Rebuild a sequence from its non-sentinel tokens."""
    front = ''.join(t[0] for t in tokens if t[1] != '$')
    middle = ''.join(t[0] for t in tokens if t[1] == '$')
    back = ''.join(t[1] for t in tokens if t[1] != '$')[::-1]
    return front + middle + back


def enumerate_support(g):
    """Every (sequence, log_prob) the graph can generate, by brute force."""
    adj = {}
    for src, dst, w, _ in g.all_edges:
        adj.setdefault(src, []).append((dst, w))

    out = []

    def walk(node, tokens, lp):
        if node == '$':
            out.append((reverse_tokens(tokens), lp))
            return
        for dst, w in adj.get(node, []):
            walk(dst, tokens if dst == '$' else tokens + [dst],
                 lp + math.log(w))

    walk('@', [], 0.0)
    return out


# ═══════════════════════════════════════════════════════════════
# Encoding
# ═══════════════════════════════════════════════════════════════

class TestEncoding:

    def test_even_length_is_all_pairs(self):
        assert flat_decompose('CASSAYFF') == [
            '@', 'CF_1', 'AF_2', 'SY_3', 'SA_4', '$']

    def test_odd_length_leaves_a_middle_residue(self):
        assert flat_decompose('CASSLGQ') == [
            '@', 'CQ_1', 'AG_2', 'SL_3', 'S$_4', '$']

    def test_single_residue(self):
        assert flat_decompose('A') == ['@', 'A$_1', '$']

    def test_two_residues(self):
        assert flat_decompose('AB') == ['@', 'AB_1', '$']

    @pytest.mark.parametrize("seq", [
        'A', 'AB', 'ABC', 'CASSLGIRRT', 'CASSAYFF', 'CASSLGQ',
        'FFFFFFFFF', 'CASSFSTCSANYGYTF', 'CASSLEPSGGTDTQYF',
    ])
    def test_roundtrip(self, seq):
        tokens = flat_decompose(seq)
        assert tokens[0] == '@' and tokens[-1] == '$'
        assert reverse_tokens(tokens[1:-1]) == seq

    def test_no_run_compression(self):
        """The whole point: FlashBack collapses the repeated residues into
        one token, this takes them one step at a time."""
        seq = 'CAAAAF'
        flat = [t for t in flat_decompose(seq) if t not in ('@', '$')]
        assert len(flat) == 3
        assert flat == ['CF_1', 'AA_2', 'AA_3']

    def test_step_index_disambiguates_repeats(self):
        """AA_2 and AA_3 are the same residue pair at different steps, and
        must be different nodes or the walk could not be reversed."""
        toks = flat_decompose('CAAAAF')
        assert toks[2] != toks[3]

    def test_method_matches_module_function(self, graph):
        assert graph.decompose('CASSLG') == flat_decompose('CASSLG')


# ═══════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════

class TestConstruction:

    def test_basic_properties(self, graph):
        assert graph.variant == 'flattened_flashback'
        assert graph.is_dag
        assert graph.n_sequences == len(SEQS)

    def test_sentinels_present_exactly_once(self, graph):
        assert graph.all_nodes.count('@') == 1
        assert graph.all_nodes.count('$') == 1

    def test_middle_token_is_not_a_sentinel(self, graph):
        """'S$_4' contains a sentinel character but is an ordinary node."""
        middles = [n for n in graph.nodes if n[1] == '$']
        assert middles, "fixture should contain an odd-length sequence"
        assert all(n in graph.nodes for n in middles)

    def test_single_sink(self, graph):
        assert graph.summary()['n_terminal'] == 1

    def test_length_distribution(self, graph):
        from collections import Counter
        assert graph.length_distribution == Counter(len(s) for s in SEQS)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            FlattenedFlashBackGraph([])

    def test_single_string_rejected(self):
        with pytest.raises(TypeError):
            FlattenedFlashBackGraph('CASSLGIRRT')

    def test_abundance_equals_repetition(self):
        a = FlattenedFlashBackGraph(['CASSLG', 'CASSQE'], abundances=[3, 1])
        b = FlattenedFlashBackGraph(['CASSLG'] * 3 + ['CASSQE'])
        assert a.pseq('CASSLG') == pytest.approx(b.pseq('CASSLG'))

    def test_max_length_cap(self):
        long_seq = 'A' * 40
        g = FlattenedFlashBackGraph(SEQS + [long_seq], max_length=27)
        assert g.n_sequences == len(SEQS)
        assert g.pseq(long_seq) < LOG_EPS_AT_FLOOR


# ═══════════════════════════════════════════════════════════════
# Probability and properness
# ═══════════════════════════════════════════════════════════════

class TestPseq:

    def test_training_sequences_are_supported(self, graph):
        for s in SEQS:
            assert graph.pseq(s) > LOG_EPS_THRESHOLD

    def test_unsupported_sequences_hit_the_floor(self, graph):
        for s in ['WWWWWWWW', '', 'Q']:
            assert graph.pseq(s) < LOG_EPS_AT_FLOOR

    def test_matches_brute_force(self, graph):
        for seq, lp in enumerate_support(graph):
            assert graph.pseq(seq) == pytest.approx(lp, abs=1e-12)

    def test_support_sums_to_one(self, graph):
        total = sum(math.exp(lp) for _, lp in enumerate_support(graph))
        assert total == pytest.approx(1.0)

    def test_no_leaked_mass(self, graph):
        assert graph.pseq_diagnostics()['is_proper']

    def test_walks_and_sequences_are_in_bijection(self, graph):
        seqs = [s for s, _ in enumerate_support(graph)]
        assert len(seqs) == len(set(seqs))

    def test_explicit_sink_makes_stopping_representable(self):
        """The reason the '$' sink exists: AA_2 ends CAAF and continues in
        CAAAAF, so a pair token cannot signal 'stop' by itself. Both must be
        scoreable, which is only possible if stopping is its own edge."""
        g = FlattenedFlashBackGraph(['CAAF', 'CAAAAF'])
        assert g.pseq('CAAF') > LOG_EPS_THRESHOLD
        assert g.pseq('CAAAAF') > LOG_EPS_THRESHOLD
        total = sum(math.exp(lp) for _, lp in enumerate_support(g))
        assert total == pytest.approx(1.0)

    def test_batch_matches_scalar(self, graph):
        batch = graph.pseq(SEQS)
        assert batch.tolist() == pytest.approx([graph.pseq(s) for s in SEQS])

    def test_pgen_alias(self):
        assert FlattenedFlashBackGraph.pgen is FlattenedFlashBackGraph.pseq


# ═══════════════════════════════════════════════════════════════
# Exact analytics
# ═══════════════════════════════════════════════════════════════

class TestAnalytics:

    def test_path_count_is_support_size(self, graph):
        assert graph.path_count == len(enumerate_support(graph))

    def test_entropy_matches_enumeration(self, graph):
        expected = -sum(math.exp(lp) * lp for _, lp in enumerate_support(graph))
        assert graph.entropy() == pytest.approx(expected)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 3.0])
    def test_power_sum_matches_enumeration(self, graph, alpha):
        expected = sum(math.exp(lp) ** alpha
                       for _, lp in enumerate_support(graph))
        assert graph.power_sum(alpha) == pytest.approx(expected)

    def test_hill_is_non_increasing(self, graph):
        import numpy as np
        assert np.all(np.diff(graph.hill_curve()['values']) <= 1e-9)

    def test_dynamic_range_matches_enumeration(self, graph):
        lps = [lp for _, lp in enumerate_support(graph)]
        d = graph.pseq_dynamic_range_detail()
        assert d['max_log_prob'] == pytest.approx(max(lps))
        assert d['min_log_prob'] == pytest.approx(min(lps))


# ═══════════════════════════════════════════════════════════════
# P-sequence analysis
# ═══════════════════════════════════════════════════════════════

class TestPseqAnalysis:

    def test_atoms_match_enumeration(self, graph):
        atoms = graph.pseq_analysis().exact_atoms()
        support = enumerate_support(graph)
        assert atoms.n_sequences == len(support)
        assert atoms.probability_mass == pytest.approx(1.0)

    def test_reconstructed_lengths_are_right(self, graph):
        """The '{residue}$_{k}' token contributes one residue, not two."""
        atoms = graph.pseq_analysis().exact_atoms()
        expected = sorted(len(s) for s, _ in enumerate_support(graph))
        assert sorted(atoms.lengths.tolist()) == expected

    def test_mellin_matches_power_sum(self, graph):
        a = graph.pseq_analysis()
        for q in (0.5, 1.0, 2.0):
            assert a.mellin(q) == pytest.approx(graph.power_sum(q), rel=1e-9)


# ═══════════════════════════════════════════════════════════════
# Simulation, IO, set ops
# ═══════════════════════════════════════════════════════════════

class TestSimulationAndIO:

    def test_simulated_sequences_are_in_support(self, graph):
        sim = graph.simulate(50, seed=7)
        for s in sim.sequences:
            assert graph.pseq(s) > LOG_EPS_THRESHOLD

    def test_reported_log_prob_matches_pseq(self, graph):
        sim = graph.simulate(30, seed=11)
        for s, lp in zip(sim.sequences, sim.log_probs):
            assert graph.pseq(s) == pytest.approx(lp, abs=1e-9)

    def test_seed_is_reproducible(self, graph):
        assert list(graph.simulate(20, seed=3).sequences) == \
               list(graph.simulate(20, seed=3).sequences)

    def test_save_load_roundtrip(self, graph, tmp_path):
        path = tmp_path / 'flat.lzg'
        graph.save(path)
        loaded = FlattenedFlashBackGraph.load(path)
        assert loaded.variant == graph.variant
        assert set(loaded.all_nodes) == set(graph.all_nodes)
        assert loaded.pseq(SEQS).tolist() == pytest.approx(
            graph.pseq(SEQS).tolist())
        assert loaded.pseq_diagnostics()['is_proper']

    def test_set_ops_keep_labels_intact(self, graph):
        u = graph.union(graph)
        assert set(u.all_nodes) == set(graph.all_nodes)
        assert u.pseq_diagnostics()['is_proper']


# ═══════════════════════════════════════════════════════════════
# The three encodings side by side
# ═══════════════════════════════════════════════════════════════

class TestThreeWayComparison:
    """FlattenedFlashBack sits between the other two by construction, and
    each pair differs in exactly one property."""

    def test_all_three_are_proper_dags(self):
        for g in (NaiveGraph(SEQS), FlattenedFlashBackGraph(SEQS),
                  FlashBackGraph(SEQS)):
            assert g.is_dag
            assert g.pgen_diagnostics()['is_proper']

    def test_all_three_support_their_training_data(self):
        for g in (NaiveGraph(SEQS), FlattenedFlashBackGraph(SEQS),
                  FlashBackGraph(SEQS)):
            for s in SEQS:
                assert g.pgen(s) > LOG_EPS_THRESHOLD

    def test_encodings_are_distinct(self):
        n = set(NaiveGraph(SEQS).nodes)
        f = set(FlattenedFlashBackGraph(SEQS).nodes)
        b = set(FlashBackGraph(SEQS).nodes)
        assert n != f and f != b and n != b

    def test_flattened_uses_about_half_as_many_steps_as_naive(self):
        """Two residues per token against one, so a walk is half as long."""
        seq = 'CASSLGIRRT'
        flat = len([t for t in flat_decompose(seq) if t not in ('@', '$')])
        from LZGraphs import naive_decompose
        naive = len([t for t in naive_decompose(seq) if t not in ('@', '$')])
        assert flat == (naive + 1) // 2

    def test_same_length_distribution(self):
        a = NaiveGraph(SEQS).length_distribution
        b = FlattenedFlashBackGraph(SEQS).length_distribution
        c = FlashBackGraph(SEQS).length_distribution
        assert a == b == c
