"""Tests for NaiveGraph: positional encoding, construction, probability,
exact analytics, length capping, file streaming, and abundance weighting.

The analytics tests all work the same way: the fixture graph is small
enough to enumerate its entire support by brute force, so every DP result
is checked against the exhaustive answer rather than against a golden
number. That is the only check that would actually catch a wrong
recursion.
"""

import math
from collections import Counter

import numpy as np
import pytest

from LZGraphs import FlashBackGraph, NaiveGraph, naive_decompose
from LZGraphs._constants import LOG_EPS_AT_FLOOR, LOG_EPS_THRESHOLD

SEQS = [
    'CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
    'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF',
]


@pytest.fixture(scope='module')
def graph():
    return NaiveGraph(SEQS)


@pytest.fixture(scope='module')
def graph_weighted():
    return NaiveGraph(SEQS, abundances=[10, 5, 3, 3, 2, 1])


def enumerate_support(g):
    """Every (sequence, log_prob) the graph can generate, by brute force.

    Walks the DAG exhaustively from '@'. Only usable on small graphs, which
    is exactly what the fixtures are for.
    """
    adj = {}
    for src, dst, w, _ in g.all_edges:
        adj.setdefault(src, []).append((dst, w))

    out = []

    def walk(node, seq, lp):
        if node == '$':
            out.append((seq, lp))
            return
        for dst, w in adj.get(node, []):
            walk(dst, seq if dst == '$' else seq + dst[0], lp + math.log(w))

    walk('@', '', 0.0)
    return out


# ═══════════════════════════════════════════════════════════════
# Encoding
# ═══════════════════════════════════════════════════════════════

class TestEncoding:
    """The node alphabet is residue + 1-based index, nothing else."""

    def test_label_format(self):
        assert naive_decompose('CASS') == [
            '@', 'C_1', 'A_2', 'S_3', 'S_4', '$']

    @pytest.mark.parametrize("seq", [
        'A', 'AA', 'CASSLGIRRT', 'CASSAYFF', 'FFFFFFFFF',
        'CASSFSTCSANYGYTF', 'CASSLEPSGGTDTQYF',
    ])
    def test_roundtrip(self, seq):
        """Dropping the sentinels and taking each label's residue rebuilds
        the sequence, so the encoding loses nothing."""
        tokens = naive_decompose(seq)
        assert tokens[0] == '@' and tokens[-1] == '$'
        assert ''.join(t[0] for t in tokens[1:-1]) == seq

    def test_position_is_one_based_index(self):
        tokens = naive_decompose('MKV')
        assert tokens[1:-1] == ['M_1', 'K_2', 'V_3']

    def test_repeated_residue_gets_distinct_nodes(self):
        """The two S's in CASS are different nodes; that is the entire
        difference from a bag-of-residues model."""
        tokens = naive_decompose('CASS')
        assert tokens[3] != tokens[4]
        assert tokens[3][0] == tokens[4][0] == 'S'

    def test_method_matches_module_function(self, graph):
        assert graph.decompose('CASSLG') == naive_decompose('CASSLG')


# ═══════════════════════════════════════════════════════════════
# Construction
# ═══════════════════════════════════════════════════════════════

class TestConstruction:

    def test_basic_properties(self, graph):
        assert graph.variant == 'naive_positional'
        assert graph.is_dag
        assert graph.n_nodes > 0
        assert graph.n_edges > 0
        assert graph.n_sequences == len(SEQS)

    def test_sentinels_present_exactly_once(self, graph):
        assert graph.all_nodes.count('@') == 1
        assert graph.all_nodes.count('$') == 1
        assert '@' not in graph.nodes and '$' not in graph.nodes

    def test_single_sink(self, graph):
        """Every walk ends at the one shared '$'."""
        assert graph.summary()['n_terminal'] == 1

    def test_node_count_is_distinct_residue_positions(self, graph):
        """Nodes are exactly the (residue, index) pairs the data contains,
        plus the two sentinels."""
        expected = {(c, i + 1) for s in SEQS for i, c in enumerate(s)}
        assert graph.n_nodes == len(expected) + 2

    def test_length_distribution(self, graph):
        assert graph.length_distribution == Counter(len(s) for s in SEQS)

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            NaiveGraph([])

    def test_single_string_rejected(self):
        with pytest.raises(TypeError):
            NaiveGraph('CASSLGIRRT')

    def test_abundance_length_mismatch_rejected(self):
        with pytest.raises(ValueError):
            NaiveGraph(SEQS, abundances=[1, 2])

    def test_abundances_change_weights(self, graph, graph_weighted):
        assert graph.n_nodes == graph_weighted.n_nodes
        assert graph.n_edges == graph_weighted.n_edges
        assert graph_weighted.n_sequences == 24
        assert graph.pseq(SEQS[0]) != graph_weighted.pseq(SEQS[0])

    def test_abundance_equals_repetition(self):
        """Abundance k must be identical to repeating the sequence k times."""
        a = NaiveGraph(['CASSLG', 'CASSQE'], abundances=[3, 1])
        b = NaiveGraph(['CASSLG'] * 3 + ['CASSQE'])
        assert a.pseq('CASSLG') == pytest.approx(b.pseq('CASSLG'))
        assert a.pseq('CASSQE') == pytest.approx(b.pseq('CASSQE'))


# ═══════════════════════════════════════════════════════════════
# Length capping
# ═══════════════════════════════════════════════════════════════

class TestMaxLength:

    def test_default_cap(self, graph):
        assert graph.max_length == 27

    def test_over_cap_sequences_are_excluded(self):
        long_seq = 'A' * 40
        g = NaiveGraph(SEQS + [long_seq], max_length=27)
        assert g.n_sequences == len(SEQS)
        assert 40 not in g.length_distribution
        assert g.pseq(long_seq) < LOG_EPS_AT_FLOOR

    def test_cap_excludes_from_length_distribution(self):
        """A skipped sequence must not appear in the length histogram: the
        model cannot generate it, so it is not part of the distribution."""
        g = NaiveGraph(['CASSLG', 'CASSQETQYFAAAA'], max_length=8)
        assert g.length_distribution == {6: 1}

    def test_no_cap(self):
        long_seq = 'A' * 40
        g = NaiveGraph(SEQS + [long_seq], max_length=None)
        assert g.max_length == 0
        assert g.n_sequences == len(SEQS) + 1
        assert g.pseq(long_seq) > LOG_EPS_THRESHOLD

    def test_all_filtered_out_raises(self):
        with pytest.raises(ValueError):
            NaiveGraph(['CASSLGIRRT'], max_length=3)

    def test_negative_cap_rejected(self):
        with pytest.raises(ValueError):
            NaiveGraph(SEQS, max_length=-1)

    def test_observed_max_length(self, graph):
        assert graph.observed_max_length == max(len(s) for s in SEQS)

    def test_uncapped_skips_beyond_walk_buffer(self):
        """With no cap, a sequence too long for the encoder's walk buffer
        is skipped like an over-cap one, rather than failing the build."""
        g = NaiveGraph(['CASSLGIRRT', 'A' * 5000], max_length=None)
        assert g.n_sequences == 1
        assert g.pseq('CASSLGIRRT') > LOG_EPS_THRESHOLD


# ═══════════════════════════════════════════════════════════════
# Probability (Pseq)
# ═══════════════════════════════════════════════════════════════

class TestPseq:

    def test_training_sequences_are_supported(self, graph):
        for s in SEQS:
            assert graph.pseq(s) > LOG_EPS_THRESHOLD

    def test_unsupported_sequences_hit_the_floor(self, graph):
        for s in ['WWWWWWWW', 'CASSLGIRRTXYZ', '', 'Q']:
            assert graph.pseq(s) < LOG_EPS_AT_FLOOR

    def test_right_residues_wrong_positions(self, graph):
        """A residue the graph knows, at a position it never occupied,
        must not score."""
        assert graph.pseq('ACSS') < LOG_EPS_AT_FLOOR

    def test_batch_matches_scalar(self, graph):
        batch = graph.pseq(SEQS)
        assert isinstance(batch, np.ndarray)
        assert batch.tolist() == pytest.approx([graph.pseq(s) for s in SEQS])

    def test_log_false_returns_probability(self, graph):
        p = graph.pseq(SEQS[0], log=False)
        assert 0.0 < p <= 1.0
        assert math.log(p) == pytest.approx(graph.pseq(SEQS[0]))

    def test_pgen_aliases_are_the_pseq_methods(self):
        """The quantity is Pseq, not the recombination Pgen OLGA and the LZ
        graphs report. The pgen_* names survive only so code that accepts
        either graph class keeps working."""
        for old, new in (
            ('pgen', 'pseq'),
            ('pgen_moments', 'pseq_moments'),
            ('pgen_diagnostics', 'pseq_diagnostics'),
            ('pgen_dynamic_range', 'pseq_dynamic_range'),
            ('pgen_dynamic_range_detail', 'pseq_dynamic_range_detail'),
        ):
            assert getattr(NaiveGraph, old) is getattr(NaiveGraph, new), old

    def test_contains(self, graph):
        assert SEQS[0] in graph
        assert 'WWWWWWWW' not in graph

    def test_matches_brute_force_enumeration(self, graph):
        """pgen must agree with the product of weights along the walk, for
        every sequence in the support."""
        for seq, lp in enumerate_support(graph):
            assert graph.pseq(seq) == pytest.approx(lp, abs=1e-12)


# ═══════════════════════════════════════════════════════════════
# The model is a proper distribution
# ═══════════════════════════════════════════════════════════════

class TestProperDistribution:

    def test_support_sums_to_one(self, graph):
        total = sum(math.exp(lp) for _, lp in enumerate_support(graph))
        assert total == pytest.approx(1.0)

    def test_weighted_support_sums_to_one(self, graph_weighted):
        total = sum(math.exp(lp) for _, lp in enumerate_support(graph_weighted))
        assert total == pytest.approx(1.0)

    def test_no_leaked_mass(self, graph):
        diag = graph.pseq_diagnostics()
        assert diag['is_proper']
        assert diag['total_absorbed'] == pytest.approx(1.0)
        assert diag['total_leaked'] == pytest.approx(0.0)

    def test_walks_and_sequences_are_in_bijection(self, graph):
        """Distinct walks must give distinct sequences, otherwise the path
        count would not be a support size and the DP would double-count."""
        seqs = [s for s, _ in enumerate_support(graph)]
        assert len(seqs) == len(set(seqs))

    def test_outgoing_weights_sum_to_one_per_node(self, graph):
        totals = {}
        for src, _dst, w, _c in graph.all_edges:
            totals[src] = totals.get(src, 0.0) + w
        for src, total in totals.items():
            assert total == pytest.approx(1.0), f"node {src} sums to {total}"


# ═══════════════════════════════════════════════════════════════
# Exact analytics, checked against enumeration
# ═══════════════════════════════════════════════════════════════

class TestAnalytics:

    def test_path_count_is_support_size(self, graph):
        assert graph.path_count == len(enumerate_support(graph))

    def test_path_count_is_exact_integer(self, graph):
        assert isinstance(graph.path_count, int)

    def test_entropy_matches_enumeration(self, graph):
        expected = -sum(math.exp(lp) * lp for _, lp in enumerate_support(graph))
        assert graph.entropy() == pytest.approx(expected)

    def test_effective_diversity_is_exp_entropy(self, graph):
        assert graph.effective_diversity() == pytest.approx(
            math.exp(graph.entropy()))

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 2.0, 3.0, 5.0])
    def test_power_sum_matches_enumeration(self, graph, alpha):
        expected = sum(math.exp(lp) ** alpha
                       for _, lp in enumerate_support(graph))
        assert graph.power_sum(alpha) == pytest.approx(expected)

    def test_hill_zero_is_path_count(self, graph):
        assert graph.hill_number(0) == pytest.approx(float(graph.path_count))

    def test_hill_one_is_effective_diversity(self, graph):
        assert graph.hill_number(1) == pytest.approx(
            graph.effective_diversity())

    def test_hill_is_non_increasing(self, graph):
        values = graph.hill_curve()['values']
        assert np.all(np.diff(values) <= 1e-9)

    def test_hill_numbers_match_individual_calls(self, graph):
        orders = [0.0, 1.0, 2.0, 4.0]
        assert graph.hill_numbers(orders).tolist() == pytest.approx(
            [graph.hill_number(a) for a in orders])

    def test_dynamic_range_matches_enumeration(self, graph):
        lps = [lp for _, lp in enumerate_support(graph)]
        detail = graph.pseq_dynamic_range_detail()
        assert detail['max_log_prob'] == pytest.approx(max(lps))
        assert detail['min_log_prob'] == pytest.approx(min(lps))
        assert graph.pseq_dynamic_range() == pytest.approx(
            (max(lps) - min(lps)) / math.log(10))

    def test_diversity_profile_fields(self, graph):
        prof = graph.diversity_profile()
        assert prof['entropy_bits'] == pytest.approx(
            prof['entropy_nats'] / math.log(2))
        assert prof['effective_diversity'] == pytest.approx(
            math.exp(prof['entropy_nats']))


# ═══════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════

class TestSimulation:

    def test_simulated_sequences_are_in_support(self, graph):
        sim = graph.simulate(50, seed=7)
        assert len(sim.sequences) == 50
        for s in sim.sequences:
            assert graph.pseq(s) > LOG_EPS_THRESHOLD

    def test_reported_log_prob_matches_pgen(self, graph):
        sim = graph.simulate(30, seed=11)
        for s, lp in zip(sim.sequences, sim.log_probs):
            assert graph.pseq(s) == pytest.approx(lp, abs=1e-9)

    def test_seed_is_reproducible(self, graph):
        a = graph.simulate(20, seed=3)
        b = graph.simulate(20, seed=3)
        assert list(a.sequences) == list(b.sequences)

    def test_different_seeds_differ(self, graph):
        a = graph.simulate(40, seed=1)
        b = graph.simulate(40, seed=2)
        assert list(a.sequences) != list(b.sequences)

    def test_simulated_lengths_within_cap(self, graph):
        sim = graph.simulate(50, seed=5)
        assert all(1 <= len(s) <= graph.observed_max_length
                   for s in sim.sequences)


# ═══════════════════════════════════════════════════════════════
# Smoothing
# ═══════════════════════════════════════════════════════════════

class TestSmoothing:

    def test_smoothing_preserves_support(self, graph):
        smoothed = NaiveGraph(SEQS, smoothing=1.0)
        assert smoothed.n_nodes == graph.n_nodes
        assert smoothed.n_edges == graph.n_edges
        assert smoothed.path_count == graph.path_count

    def test_smoothing_still_normalised(self):
        smoothed = NaiveGraph(SEQS, smoothing=1.0)
        total = sum(math.exp(lp) for _, lp in enumerate_support(smoothed))
        assert total == pytest.approx(1.0)

    def test_smoothing_flattens_weights(self):
        """Laplace mass moves probability toward the rarer branch."""
        seqs = ['CASSLG'] * 10 + ['CASSQE']
        sharp = NaiveGraph(seqs, smoothing=0.0)
        flat = NaiveGraph(seqs, smoothing=5.0)
        assert flat.pseq('CASSQE') > sharp.pseq('CASSQE')
        assert flat.pseq('CASSLG') < sharp.pseq('CASSLG')


# ═══════════════════════════════════════════════════════════════
# Set operations
# ═══════════════════════════════════════════════════════════════

class TestSetOperations:

    def test_union_with_self_is_identity(self, graph):
        u = graph.union(graph)
        assert u.n_nodes == graph.n_nodes
        assert u.n_edges == graph.n_edges
        assert u.pseq(SEQS).tolist() == pytest.approx(
            graph.pseq(SEQS).tolist())

    def test_set_ops_keep_labels_intact(self, graph):
        """Regression: a set op must not re-append node_pos and turn 'A_2'
        into 'A_2_4294967295'."""
        u = graph.union(graph)
        assert set(u.all_nodes) == set(graph.all_nodes)

    def test_union_stays_proper(self, graph):
        other = NaiveGraph(['CASSWWTQYF', 'CASSLGIRRT'])
        u = graph.union(other)
        assert u.pseq_diagnostics()['is_proper']
        assert u.pseq('CASSWWTQYF') > LOG_EPS_THRESHOLD
        assert u.pseq(SEQS[0]) > LOG_EPS_THRESHOLD

    def test_intersection_stays_proper(self, graph):
        other = NaiveGraph(SEQS[:3])
        i = graph.intersection(other)
        assert i.pseq_diagnostics()['is_proper']

    def test_operators(self, graph):
        other = NaiveGraph(SEQS[:3])
        assert (graph | other).n_edges == graph.union(other).n_edges
        assert (graph & other).n_edges == graph.intersection(other).n_edges


# ═══════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════

class TestSerialization:

    def test_save_load_roundtrip(self, graph, tmp_path):
        path = tmp_path / 'naive.lzg'
        graph.save(path)
        loaded = NaiveGraph.load(path)
        assert loaded.n_nodes == graph.n_nodes
        assert loaded.n_edges == graph.n_edges
        assert loaded.variant == graph.variant
        assert set(loaded.all_nodes) == set(graph.all_nodes)
        assert loaded.pseq(SEQS).tolist() == pytest.approx(
            graph.pseq(SEQS).tolist())
        assert loaded.entropy() == pytest.approx(graph.entropy())
        assert loaded.path_count == graph.path_count

    def test_loaded_graph_is_still_proper(self, graph, tmp_path):
        path = tmp_path / 'naive.lzg'
        graph.save(path)
        assert NaiveGraph.load(path).pseq_diagnostics()['is_proper']


# ═══════════════════════════════════════════════════════════════
# File input
# ═══════════════════════════════════════════════════════════════

class TestFromFile:

    def test_plain_file(self, graph, tmp_path):
        path = tmp_path / 'seqs.txt'
        path.write_text('\n'.join(SEQS) + '\n')
        g = NaiveGraph.from_file(path)
        assert g.n_nodes == graph.n_nodes
        assert g.n_edges == graph.n_edges
        assert g.pseq(SEQS).tolist() == pytest.approx(
            graph.pseq(SEQS).tolist())

    def test_abundance_file(self, graph_weighted, tmp_path):
        path = tmp_path / 'seqs.tsv'
        counts = [10, 5, 3, 3, 2, 1]
        path.write_text(
            '\n'.join(f'{s}\t{c}' for s, c in zip(SEQS, counts)) + '\n')
        g = NaiveGraph.from_file(path)
        assert g.n_sequences == sum(counts)
        assert g.pseq(SEQS).tolist() == pytest.approx(
            graph_weighted.pseq(SEQS).tolist())

    def test_file_respects_max_length(self, tmp_path):
        path = tmp_path / 'seqs.txt'
        path.write_text('\n'.join(SEQS + ['A' * 40]) + '\n')
        g = NaiveGraph.from_file(path, max_length=27)
        assert g.n_sequences == len(SEQS)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            NaiveGraph.from_file(tmp_path / 'nope.txt')


# ═══════════════════════════════════════════════════════════════
# P-sequence analysis
# ═══════════════════════════════════════════════════════════════

class TestPseqAnalysis:
    """The Mellin machinery is generic over sentinel-bounded DAGs."""

    def test_atoms_match_enumeration(self, graph):
        atoms = graph.pseq_analysis().exact_atoms()
        support = enumerate_support(graph)
        assert atoms.n_sequences == len(support)
        assert atoms.probability_mass == pytest.approx(1.0)
        assert sorted(atoms.surprisal.tolist()) == pytest.approx(
            sorted(-lp for _, lp in support))

    def test_reconstructed_lengths_are_right(self, graph):
        """Regression: a bare '$' sink contributes no character. Counting it
        as one made every reconstructed length one too long."""
        atoms = graph.pseq_analysis().exact_atoms()
        expected = sorted(len(s) for s, _ in enumerate_support(graph))
        assert sorted(atoms.lengths.tolist()) == expected

    def test_surprisal_bounds_match_atoms(self, graph):
        a = graph.pseq_analysis()
        atoms = a.exact_atoms()
        assert a.true_min_surprisal == pytest.approx(atoms.surprisal.min())
        assert a.true_max_surprisal == pytest.approx(atoms.surprisal.max())

    def test_mellin_matches_power_sum(self, graph):
        a = graph.pseq_analysis()
        for q in (0.5, 1.0, 2.0):
            assert a.mellin(q) == pytest.approx(graph.power_sum(q), rel=1e-9)

    def test_flashback_lengths_still_right(self):
        """The sentinel fix must not disturb FlashBack, whose root carries
        both sentinels inside one label."""
        atoms = FlashBackGraph(SEQS).pseq_analysis().exact_atoms()
        assert sorted(atoms.lengths.tolist()) == sorted(
            len(s) for s in set(SEQS))


# ═══════════════════════════════════════════════════════════════
# Side-by-side with FlashBack
# ═══════════════════════════════════════════════════════════════

class TestAgainstFlashBack:
    """The two variants must be directly comparable: same training data,
    same guarantees, different node alphabet."""

    def test_both_are_proper_dags(self):
        n = NaiveGraph(SEQS)
        f = FlashBackGraph(SEQS)
        assert n.is_dag and f.is_dag
        assert n.pseq_diagnostics()['is_proper']
        assert f.pgen_diagnostics()['is_proper']

    def test_both_support_their_training_data(self):
        n = NaiveGraph(SEQS)
        f = FlashBackGraph(SEQS)
        for s in SEQS:
            assert n.pseq(s) > LOG_EPS_THRESHOLD
            assert f.pgen(s) > LOG_EPS_THRESHOLD

    def test_same_length_distribution(self):
        assert (NaiveGraph(SEQS).length_distribution ==
                FlashBackGraph(SEQS).length_distribution)

    def test_encodings_differ(self):
        """Sanity check that this is actually a different model."""
        n = NaiveGraph(SEQS)
        f = FlashBackGraph(SEQS)
        assert set(n.nodes) != set(f.nodes)
        assert n.pseq(SEQS[0]) != f.pgen(SEQS[0])
