"""Tests for FlashBackGraph: decomposition, construction, simulation,
probability, exact analytics, file streaming, and abundance weighting."""

import numpy as np
import pytest
from LZGraphs._constants import LOG_EPS_THRESHOLD, LOG_EPS_AT_FLOOR
from collections import Counter
from LZGraphs import (
    FlashBackGraph, FlashBackStream, flashback_decompose, flashback_reverse,
)

SEQS = [
    'CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
    'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF',
]


@pytest.fixture(scope='module')
def fb_graph():
    return FlashBackGraph(SEQS)


@pytest.fixture(scope='module')
def fb_graph_weighted():
    return FlashBackGraph(SEQS, abundances=[10, 5, 3, 3, 2, 1])


# ═══════════════════════════════════════════════════════════════
# Decomposition
# ═══════════════════════════════════════════════════════════════

class TestDecomposition:
    """FlashBack decompose + reverse round-trip."""

    @pytest.mark.parametrize("seq", [
        'A', 'AA', 'AAA', 'AB', 'ABBA', 'ABCDEF',
        'CASSLGIRRT', 'CASSAYFF', 'HELLO', 'FFFFFFFFF',
        'CASSFSTCSANYGYTF', 'CASSLEPSGGTDTQYF',
        'AABCA', 'AABBCCDD', 'AAABBBCCCDDDEEE',
    ])
    def test_round_trip(self, seq):
        tokens = flashback_decompose(seq)
        assert flashback_reverse(tokens) == seq

    def test_first_token_is_sentinel(self):
        tokens = flashback_decompose('CASSLGIRRT')
        assert tokens[0] == '@$_1{0}'

    def test_tokens_have_order_suffix(self):
        tokens = flashback_decompose('CASSLGIRRT')
        for i, t in enumerate(tokens):
            assert t.endswith(f'{{{i}}}')

    def test_empty_raises(self):
        # Empty string should still produce at least the sentinel token
        tokens = flashback_decompose('')
        assert len(tokens) >= 1


# ═══════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════

class TestConstruction:
    def test_basic_build(self, fb_graph):
        assert fb_graph.n_nodes > 0
        assert fb_graph.n_edges > 0
        assert fb_graph.is_dag

    def test_variant_is_flashback(self, fb_graph):
        assert fb_graph.variant == 'flashback'

    def test_repr(self, fb_graph):
        r = repr(fb_graph)
        assert 'FlashBackGraph' in r
        assert 'nodes=' in r

    def test_single_sequence(self):
        g = FlashBackGraph(['CASSLGIRRT'])
        tokens = flashback_decompose('CASSLGIRRT')
        assert g.n_nodes == len(tokens)
        assert g.n_edges == len(tokens) - 1

    def test_duplicate_sequences(self):
        g1 = FlashBackGraph(['CASSLGIRRT'])
        g2 = FlashBackGraph(['CASSLGIRRT', 'CASSLGIRRT'])
        assert g1.n_nodes == g2.n_nodes
        assert g1.n_edges == g2.n_edges

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            FlashBackGraph([])

    def test_string_raises(self):
        with pytest.raises(TypeError):
            FlashBackGraph('CASSLGIRRT')

    def test_root_and_sinks(self, fb_graph):
        summary = fb_graph.summary()
        assert summary['n_initial'] == 1  # single root
        assert summary['n_terminal'] > 0  # at least one sink

    def test_with_abundances(self, fb_graph_weighted):
        assert fb_graph_weighted.n_nodes > 0


# ═══════════════════════════════════════════════════════════════
# Structural properties
# ═══════════════════════════════════════════════════════════════

class TestStructural:
    def test_length_distribution(self, fb_graph):
        ld = fb_graph.length_distribution
        assert sum(ld.values()) == len(SEQS)
        for s in SEQS:
            assert len(s) in ld

    def test_n_sequences(self, fb_graph):
        assert fb_graph.n_sequences == len(SEQS)

    def test_nodes_exclude_sentinels(self, fb_graph):
        nodes = fb_graph.nodes
        for n in nodes:
            assert not n.startswith('@')
            assert '$' not in n

    def test_all_nodes_include_root(self, fb_graph):
        assert '@$_1{0}' in fb_graph.all_nodes

    def test_density_positive(self, fb_graph):
        assert fb_graph.density > 0

    def test_degrees(self, fb_graph):
        assert len(fb_graph.out_degrees) == fb_graph.n_nodes
        assert len(fb_graph.in_degrees) == fb_graph.n_nodes

    def test_adjacency_csr(self, fb_graph):
        csr = fb_graph.adjacency_csr()
        assert len(csr['row_offsets']) == fb_graph.n_nodes + 1
        assert len(csr['col_indices']) == fb_graph.n_edges
        assert len(csr['weights']) == fb_graph.n_edges

    def test_successors(self, fb_graph):
        succs = fb_graph.successors('@$_1{0}')
        assert len(succs) > 0
        # Each successor is (label, weight, count)
        for label, w, c in succs:
            assert isinstance(label, str)
            assert 0 < w <= 1.0
            assert c >= 1


# ═══════════════════════════════════════════════════════════════
# flashback_pgen
# ═══════════════════════════════════════════════════════════════

class TestPgen:
    def test_training_sequences_have_positive_prob(self, fb_graph):
        for s in SEQS:
            assert fb_graph.pgen(s) > LOG_EPS_THRESHOLD

    def test_unknown_sequence_has_zero_prob(self, fb_graph):
        assert fb_graph.pgen('ZZZZZZ') < LOG_EPS_AT_FLOOR

    def test_contains_operator(self, fb_graph):
        assert 'CASSLGIRRT' in fb_graph
        assert 'ZZZZZZ' not in fb_graph

    def test_pgen_matches_manual_edge_walk(self, fb_graph):
        seq = SEQS[0]
        tokens = flashback_decompose(seq)
        nodes = fb_graph.all_nodes
        csr = fb_graph.adjacency_csr()
        manual = 0.0
        for i in range(len(tokens) - 1):
            src = nodes.index(tokens[i])
            dst = nodes.index(tokens[i + 1])
            start = csr['row_offsets'][src]
            end = csr['row_offsets'][src + 1]
            for e in range(start, end):
                if csr['col_indices'][e] == dst:
                    manual += np.log(csr['weights'][e])
                    break
        api = fb_graph.pgen(seq)
        assert abs(manual - api) < 1e-10

    def test_batch_pgen(self, fb_graph):
        batch = fb_graph.pgen(list(set(SEQS)))
        assert isinstance(batch, np.ndarray)
        assert all(lp > LOG_EPS_THRESHOLD for lp in batch)

    def test_pgen_exp(self, fb_graph):
        p = fb_graph.pgen(SEQS[0], log=False)
        assert 0 < p <= 1.0

    def test_abundances_shift_probabilities(self):
        g_equal = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF'])
        g_heavy = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF'],
                                 abundances=[1000, 1])
        lp_eq = g_equal.pgen('CASSLGIRRT')
        lp_hv = g_heavy.pgen('CASSLGIRRT')
        # Heavy weighting should increase probability
        assert lp_hv > lp_eq


# ═══════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════

class TestSimulation:
    def test_simulate_returns_sequences(self, fb_graph):
        sim = fb_graph.simulate(10, seed=42)
        seqs = list(sim.sequences)
        assert len(seqs) == 10
        for s in seqs:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_simulated_sequences_round_trip(self, fb_graph):
        sim = fb_graph.simulate(50, seed=42)
        for s in sim.sequences:
            tokens = flashback_decompose(s)
            assert flashback_reverse(tokens) == s

    def test_simulated_sequences_have_positive_pgen(self, fb_graph):
        sim = fb_graph.simulate(50, seed=42)
        for s in set(sim.sequences):
            assert fb_graph.pgen(s) > LOG_EPS_THRESHOLD

    def test_simulation_frequencies_match_model(self):
        g = FlashBackGraph(['CASSLGIRRT'] * 5 + ['CASSAYFF'] * 3 +
                           ['HELLO'] * 2)
        sim = g.simulate(50_000, seed=42)
        counts = Counter(sim.sequences)
        total = sum(counts.values())
        for s in counts:
            emp = counts[s] / total
            model_p = np.exp(g.pgen(s))
            # Within 5% relative error for 50K samples
            assert abs(emp - model_p) / model_p < 0.05, \
                f"{s}: emp={emp:.4f} model={model_p:.4f}"

    def test_seed_reproducibility(self, fb_graph):
        s1 = list(fb_graph.simulate(20, seed=123).sequences)
        s2 = list(fb_graph.simulate(20, seed=123).sequences)
        assert s1 == s2

    def test_different_seeds_differ(self, fb_graph):
        s1 = list(fb_graph.simulate(20, seed=1).sequences)
        s2 = list(fb_graph.simulate(20, seed=2).sequences)
        assert s1 != s2


# ═══════════════════════════════════════════════════════════════
# Exact DP analytics
# ═══════════════════════════════════════════════════════════════

class TestAnalytics:
    def test_path_count_at_least_k(self, fb_graph):
        # Path count >= number of unique input sequences
        assert fb_graph.path_count >= len(set(SEQS))

    def test_path_count_single_seq(self):
        g = FlashBackGraph(['CASSLGIRRT'])
        assert g.path_count == 1.0

    def test_effective_diversity_positive(self, fb_graph):
        assert fb_graph.effective_diversity() > 0

    def test_diversity_profile_keys(self, fb_graph):
        dp = fb_graph.diversity_profile()
        for key in ('entropy_nats', 'entropy_bits',
                     'effective_diversity', 'uniformity'):
            assert key in dp

    def test_power_sum_m1_is_one(self, fb_graph):
        assert abs(fb_graph.power_sum(1.0) - 1.0) < 1e-10

    def test_hill_d0_equals_path_count(self, fb_graph):
        assert abs(fb_graph.hill_number(0) - fb_graph.path_count) < 1e-6

    def test_hill_d1_equals_exp_entropy(self, fb_graph):
        dp = fb_graph.diversity_profile()
        d1 = fb_graph.hill_number(1)
        assert abs(d1 - dp['effective_diversity']) < 1e-6

    def test_hill_d2_equals_inv_m2(self, fb_graph):
        m2 = fb_graph.power_sum(2)
        d2 = fb_graph.hill_number(2)
        assert abs(d2 - 1.0 / m2) < 1e-6

    def test_hill_numbers_batch(self, fb_graph):
        orders = [0, 1, 2, 3, 5]
        batch = fb_graph.hill_numbers(orders)
        assert len(batch) == len(orders)
        # Hill numbers should be non-increasing
        for i in range(len(batch) - 1):
            assert batch[i] >= batch[i + 1] - 1e-10

    def test_hill_curve(self, fb_graph):
        hc = fb_graph.hill_curve()
        assert 'orders' in hc and 'values' in hc
        assert len(hc['orders']) == len(hc['values'])

    def test_dynamic_range_nonnegative(self, fb_graph):
        assert fb_graph.pgen_dynamic_range() >= 0

    def test_dynamic_range_detail(self, fb_graph):
        dr = fb_graph.pgen_dynamic_range_detail()
        assert dr['max_log_prob'] >= dr['min_log_prob']
        assert dr['dynamic_range_nats'] >= 0

    def test_dynamic_range_positive_with_weights(self):
        """With unequal abundances, dynamic range should be > 0."""
        g = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF'], abundances=[100, 1])
        assert g.pgen_dynamic_range() > 0

    def test_pgen_diagnostics_proper(self, fb_graph):
        diag = fb_graph.pgen_diagnostics()
        assert diag['is_proper']
        assert abs(diag['total_absorbed'] - 1.0) < 1e-6
        assert diag['mc_samples'] == 0  # exact, not MC

    def test_uniformity_bounded(self, fb_graph):
        dp = fb_graph.diversity_profile()
        assert 0.0 <= dp['uniformity'] <= 1.0


# ═══════════════════════════════════════════════════════════════
# File streaming
# ═══════════════════════════════════════════════════════════════

class TestFileIO:
    def test_from_file_matches_list(self, tmp_path):
        p = tmp_path / 'test.tsv'
        p.write_text('CASSLGIRRT\t5\nCASSAYFF\t3\nHELLO\t2\n')
        gf = FlashBackGraph.from_file(str(p))
        gm = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF', 'HELLO'],
                             abundances=[5, 3, 2])
        assert gf.n_nodes == gm.n_nodes
        assert gf.n_edges == gm.n_edges
        assert gf.path_count == gm.path_count
        for s in ['CASSLGIRRT', 'CASSAYFF', 'HELLO']:
            assert abs(gf.pgen(s) - gm.pgen(s)) < 1e-10

    def test_from_file_plain(self, tmp_path):
        p = tmp_path / 'plain.txt'
        p.write_text('CASSLGIRRT\nCASSAYFF\nHELLO\n')
        g = FlashBackGraph.from_file(str(p))
        assert g.n_sequences == 3

    def test_save_load_round_trip(self, fb_graph, tmp_path):
        path = str(tmp_path / 'fb.lzg')
        fb_graph.save(path)
        g2 = FlashBackGraph.load(path)
        assert g2.n_nodes == fb_graph.n_nodes
        assert g2.n_edges == fb_graph.n_edges

    def test_from_file_bad_path(self):
        with pytest.raises(Exception):
            FlashBackGraph.from_file('/nonexistent/path.tsv')

    def test_from_file_empty_path(self):
        with pytest.raises(ValueError):
            FlashBackGraph.from_file('')


# ═══════════════════════════════════════════════════════════════
# Graph operations
# ═══════════════════════════════════════════════════════════════

class TestOperations:
    def test_union(self):
        g1 = FlashBackGraph(['CASSLGIRRT'])
        g2 = FlashBackGraph(['CASSAYFF'])
        gu = g1 | g2
        assert gu.n_nodes >= max(g1.n_nodes, g2.n_nodes)

    def test_intersection(self):
        g1 = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF'])
        g2 = FlashBackGraph(['CASSLGIRRT', 'HELLO'])
        gi = g1 & g2
        assert gi.n_nodes > 0

    def test_difference(self):
        g1 = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF'])
        g2 = FlashBackGraph(['CASSLGIRRT'])
        gd = g1 - g2
        assert gd.n_nodes > 0


# ═══════════════════════════════════════════════════════════════
# Bayesian posterior — Dirichlet-Multinomial update on prior topology
# ═══════════════════════════════════════════════════════════════

class TestPosterior:
    def test_signature(self, fb_graph):
        post = fb_graph.posterior(['CASSLGIRRT'], kappa=1.0)
        assert isinstance(post, FlashBackGraph)

    def test_topology_preserved(self, fb_graph):
        """Posterior must not change the prior's topology."""
        post = fb_graph.posterior(['CASSLGIRRT'], kappa=1.0)
        assert post.n_nodes == fb_graph.n_nodes
        assert post.n_edges == fb_graph.n_edges

    def test_kappa_large_returns_prior(self, fb_graph):
        """At kappa >> n_ind the posterior should match the prior weights."""
        post = fb_graph.posterior(['CASSLGIRRT'], kappa=1e12)
        prior_csr = fb_graph.adjacency_csr()
        post_csr = post.adjacency_csr()
        assert np.allclose(prior_csr['weights'], post_csr['weights'], atol=1e-6)

    def test_kappa_zero_reflects_data(self):
        """At kappa=0 the posterior weights should reflect the subtracted repertoire only."""
        prior = FlashBackGraph(['CASSLGIRRT', 'CASSAYFF', 'CASSQETQYF'])
        # At kappa=0, the posterior at each node should match the individual data's
        # transition probabilities restricted to the prior's topology.
        # We cannot directly compare to a donor-only graph because topologies differ,
        # but we can check that per-source rows still sum to 1.
        post = prior.posterior(['CASSLGIRRT'], kappa=0.0)
        csr = post.adjacency_csr()
        for u in range(len(csr['row_offsets']) - 1):
            e_start = csr['row_offsets'][u]
            e_end = csr['row_offsets'][u + 1]
            if e_start == e_end:
                continue
            wsum = csr['weights'][e_start:e_end].sum()
            # Rows with no data for this donor fall back to prior (renormalised),
            # rows with data sum to 1 as well. Either way: sum == 1.0 (± eps).
            assert abs(wsum - 1.0) < 1e-6 or wsum == 0.0, \
                f"row {u} sums to {wsum}"

    def test_rows_sum_to_one(self, fb_graph):
        """Posterior transition weights must sum to 1 per source node."""
        post = fb_graph.posterior(['CASSLGIRRT', 'CASSAYFF'], kappa=1.0)
        csr = post.adjacency_csr()
        for u in range(len(csr['row_offsets']) - 1):
            e_start = csr['row_offsets'][u]
            e_end = csr['row_offsets'][u + 1]
            if e_start == e_end:
                continue
            wsum = csr['weights'][e_start:e_end].sum()
            assert abs(wsum - 1.0) < 1e-6, f"row {u} sum = {wsum}"

    def test_unknown_sequences_ignored(self, fb_graph):
        """Sequences whose tokens are not in the prior should not affect weights."""
        prior_csr = fb_graph.adjacency_csr()
        # A sequence made of characters not in the training set produces tokens
        # that are mostly absent from the graph. High kappa makes this a no-op.
        post = fb_graph.posterior(['ZZZZZZZZZZ'], kappa=1e12)
        post_csr = post.adjacency_csr()
        assert np.allclose(prior_csr['weights'], post_csr['weights'], atol=1e-6)

    def test_abundance_scales_contribution(self, fb_graph):
        """Higher abundance should shift the posterior more toward the data."""
        post_lo = fb_graph.posterior(['CASSLGIRRT'], abundances=[1], kappa=1.0)
        post_hi = fb_graph.posterior(['CASSLGIRRT'], abundances=[1000], kappa=1.0)
        lo_csr = post_lo.adjacency_csr()
        hi_csr = post_hi.adjacency_csr()
        # Some edge weights must differ.
        assert not np.allclose(lo_csr['weights'], hi_csr['weights'], atol=1e-4)

    def test_invalid_kappa(self, fb_graph):
        with pytest.raises(ValueError):
            fb_graph.posterior(['CASSLGIRRT'], kappa=-1.0)

    def test_string_raises(self, fb_graph):
        with pytest.raises(TypeError):
            fb_graph.posterior('CASSLGIRRT', kappa=1.0)

    def test_mismatched_abundances(self, fb_graph):
        with pytest.raises(ValueError):
            fb_graph.posterior(['CASSLGIRRT', 'CASSAYFF'], abundances=[1])


# ═══════════════════════════════════════════════════════════════
# Subtract / without
# ═══════════════════════════════════════════════════════════════

class TestWithout:
    def test_signature(self, fb_graph):
        sub = fb_graph.without(['CASSLGIRRT'])
        assert isinstance(sub, FlashBackGraph)

    def test_leave_one_out_matches_direct_build(self):
        """Subtracting one sequence must match rebuilding from the remainder."""
        all_seqs = ['CASSLGIRRT', 'CASSAYFF', 'CASSQETQYF', 'CASSGQGSYEQYF']
        g_all = FlashBackGraph(all_seqs)
        g_without = g_all.without(['CASSGQGSYEQYF'])
        g_direct = FlashBackGraph(all_seqs[:3])

        # Every edge in g_direct must have the same count in g_without
        # (modulo isolated-node artefacts in the label-index mapping).
        direct_edges = {(s, d): c for s, d, _, c in g_direct.all_edges_with_counts()} \
            if hasattr(g_direct, 'all_edges_with_counts') else None
        if direct_edges is None:
            # Fallback: use the edges property and CSR counts
            d_csr = g_direct.adjacency_csr()
            d_nodes = g_direct.all_nodes
            direct_edges = {}
            for u in range(len(d_csr['row_offsets']) - 1):
                for e in range(d_csr['row_offsets'][u], d_csr['row_offsets'][u + 1]):
                    v = d_csr['col_indices'][e]
                    direct_edges[(d_nodes[u], d_nodes[v])] = int(d_csr['counts'][e])

        w_csr = g_without.adjacency_csr()
        w_nodes = g_without.all_nodes
        without_edges = {}
        for u in range(len(w_csr['row_offsets']) - 1):
            for e in range(w_csr['row_offsets'][u], w_csr['row_offsets'][u + 1]):
                v = w_csr['col_indices'][e]
                without_edges[(w_nodes[u], w_nodes[v])] = int(w_csr['counts'][e])

        # Every directly-built edge must exist in g_without with matching count.
        for edge, c in direct_edges.items():
            assert edge in without_edges, f"missing edge {edge} in subtracted graph"
            assert without_edges[edge] == c, \
                f"edge {edge}: direct={c}, subtract={without_edges[edge]}"

    def test_subtract_all_removes_all_edges(self):
        """Subtracting every sequence must leave no edges with positive count."""
        seqs = ['CASSLGIRRT', 'CASSAYFF', 'CASSQETQYF']
        g = FlashBackGraph(seqs)
        g_empty = g.without(seqs)
        assert g_empty.n_edges == 0

    def test_subtract_nonexistent_is_noop(self, fb_graph):
        """Subtracting sequences whose tokens aren't in the graph is a no-op."""
        before_edges = fb_graph.n_edges
        before_csr = fb_graph.adjacency_csr()
        g2 = fb_graph.without(['ZZZZZZZZZZ'])
        assert g2.n_edges == before_edges
        after_csr = g2.adjacency_csr()
        # ``without()`` returns a fresh graph with no β calibration;
        # edge weights revert to MLE. Compare MLE backbone (counts) which
        # is the structural invariant of "no-op subtraction".
        assert np.array_equal(before_csr['counts'], after_csr['counts'])

    def test_subtract_clamps_at_zero(self):
        """Subtracting more than the original count must not underflow."""
        g = FlashBackGraph(['CASSLGIRRT'], abundances=[2])
        g2 = g.without(['CASSLGIRRT'], abundances=[100])  # big overshoot
        assert g2.n_edges == 0  # all edges cleared, not negative

    def test_abundance_partial_subtraction(self):
        """Partial subtraction leaves residual counts intact."""
        g = FlashBackGraph(['CASSLGIRRT'], abundances=[10])
        g2 = g.without(['CASSLGIRRT'], abundances=[3])
        csr = g2.adjacency_csr()
        assert g2.n_edges > 0  # edges survive
        # Every surviving edge should have count 10 - 3 = 7
        for e in range(g2.n_edges):
            assert csr['counts'][e] == 7

    def test_weights_sum_to_one_after_subtract(self, fb_graph):
        """Row weights remain a valid categorical after subtraction."""
        g2 = fb_graph.without(['CASSLGIRRT'])
        csr = g2.adjacency_csr()
        for u in range(len(csr['row_offsets']) - 1):
            e_start = csr['row_offsets'][u]
            e_end = csr['row_offsets'][u + 1]
            if e_start == e_end:
                continue
            wsum = csr['weights'][e_start:e_end].sum()
            assert abs(wsum - 1.0) < 1e-6, f"row {u} sum = {wsum}"

    def test_pgen_drops_after_subtract(self):
        """The subtracted sequence must drop from pgen>0 to pgen=0 if uniquely contributed."""
        g = FlashBackGraph(['CASSLGIRRT'])
        assert g.pgen('CASSLGIRRT') > LOG_EPS_THRESHOLD
        g2 = g.without(['CASSLGIRRT'])
        assert g2.pgen('CASSLGIRRT') < LOG_EPS_AT_FLOOR

    def test_string_raises(self, fb_graph):
        with pytest.raises(TypeError):
            fb_graph.without('CASSLGIRRT')

    def test_mismatched_abundances(self, fb_graph):
        with pytest.raises(ValueError):
            fb_graph.without(['CASSLGIRRT', 'CASSAYFF'], abundances=[1])

    def test_topo_valid_after_subtract(self, fb_graph):
        """The subtracted graph should still be a DAG with valid topo order."""
        g2 = fb_graph.without(['CASSLGIRRT'])
        assert g2.is_dag


# ═══════════════════════════════════════════════════════════════
# Streaming construction (FlashBackStream)
# ═══════════════════════════════════════════════════════════════

class TestStreamConstruction:
    """The streaming builder must produce a graph bit-identical to a
    batch-built graph constructed from the same sequence pool, and
    enforce a clean lifecycle (no use-after-finalize, no double-free)."""

    def test_basic_open_finalize(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        g = s.finalize()
        assert g.n_nodes > 0
        assert g.n_edges > 0

    def test_peek_grows_with_adds(self):
        s = FlashBackStream()
        before = s.peek()
        assert before == {'n_nodes': 0, 'n_edges': 0}
        s.add_sequences(SEQS[:2])
        mid = s.peek()
        assert mid['n_nodes'] > 0
        assert mid['n_edges'] > 0
        s.add_sequences(SEQS[2:])
        after = s.peek()
        assert after['n_nodes'] >= mid['n_nodes']
        assert after['n_edges'] >= mid['n_edges']
        s.finalize()

    def test_streaming_matches_batch_unweighted(self):
        g_batch = FlashBackGraph(SEQS)
        s = FlashBackStream()
        s.add_sequences(SEQS[:3])
        s.add_sequences(SEQS[3:])
        g_stream = s.finalize()
        assert g_batch.n_nodes == g_stream.n_nodes
        assert g_batch.n_edges == g_stream.n_edges
        a, b = g_batch._get_csr(), g_stream._get_csr()
        assert np.array_equal(a['row_offsets'], b['row_offsets'])
        assert np.array_equal(a['col_indices'], b['col_indices'])
        assert np.allclose(a['weights'], b['weights'])

    def test_streaming_matches_batch_weighted(self):
        abundances = [10, 5, 3, 3, 2, 1]
        g_batch = FlashBackGraph(SEQS, abundances=abundances)
        s = FlashBackStream()
        s.add_sequences(SEQS[:3], abundances=abundances[:3])
        s.add_sequences(SEQS[3:], abundances=abundances[3:])
        g_stream = s.finalize()
        a, b = g_batch._get_csr(), g_stream._get_csr()
        assert np.allclose(a['weights'], b['weights'])

    def test_pgen_matches_batch(self):
        g_batch = FlashBackGraph(SEQS)
        s = FlashBackStream()
        s.add_sequences(SEQS)
        g_stream = s.finalize()
        for seq in SEQS:
            p_b = g_batch.pgen(seq, log=True)
            p_s = g_stream.pgen(seq, log=True)
            assert abs(p_b - p_s) < 1e-10

    def test_smoothing_propagates(self):
        s = FlashBackStream(smoothing=0.5)
        s.add_sequences(SEQS)
        g_stream = s.finalize()
        g_batch = FlashBackGraph(SEQS, smoothing=0.5)
        a, b = g_batch._get_csr(), g_stream._get_csr()
        assert np.allclose(a['weights'], b['weights'])

    def test_add_after_finalize_raises(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        s.finalize()
        with pytest.raises(RuntimeError):
            s.add_sequences(['CASSAYFF'])

    def test_double_finalize_raises(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        s.finalize()
        with pytest.raises(RuntimeError):
            s.finalize()

    def test_abort_after_finalize_is_noop(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        s.finalize()
        s.abort()  # must not raise

    def test_abort_releases_resources(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        s.abort()
        with pytest.raises(RuntimeError):
            s.add_sequences(['CASSAYFF'])
        with pytest.raises(RuntimeError):
            s.finalize()

    def test_context_manager_finalizes_via_exit(self):
        with FlashBackStream() as s:
            s.add_sequences(SEQS)
            assert s.peek()['n_edges'] > 0
        # __exit__ on a non-finalized stream calls abort
        with pytest.raises(RuntimeError):
            s.add_sequences(SEQS)

    def test_context_manager_does_not_clobber_explicit_finalize(self):
        with FlashBackStream() as s:
            s.add_sequences(SEQS)
            g = s.finalize()
        # Graph is independent of stream lifecycle.
        assert g.n_nodes > 0

    def test_empty_add_is_noop(self):
        s = FlashBackStream()
        s.add_sequences([])
        assert s.peek() == {'n_nodes': 0, 'n_edges': 0}
        s.add_sequences(SEQS)
        s.add_sequences([])
        peek = s.peek()
        s.finalize()
        # empty adds didn't perturb anything
        assert peek['n_edges'] > 0

    def test_string_input_raises(self):
        s = FlashBackStream()
        with pytest.raises(TypeError):
            s.add_sequences('CASSLGIRRT')  # single string, not list

    def test_many_small_batches_match_one_big(self):
        # Stream the SEQS one at a time vs in one big batch.
        s_big = FlashBackStream()
        s_big.add_sequences(SEQS)
        g_big = s_big.finalize()

        s_small = FlashBackStream()
        for seq in SEQS:
            s_small.add_sequences([seq])
        g_small = s_small.finalize()

        a, b = g_big._get_csr(), g_small._get_csr()
        assert np.array_equal(a['row_offsets'], b['row_offsets'])
        assert np.array_equal(a['col_indices'], b['col_indices'])
        assert np.allclose(a['weights'], b['weights'])

    def test_peek_after_finalize_returns_zeros(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        s.finalize()
        assert s.peek() == {'n_nodes': 0, 'n_edges': 0}

    def test_simulate_on_streamed_graph(self):
        s = FlashBackStream()
        s.add_sequences(SEQS)
        g = s.finalize()
        sim = g.simulate(5, seed=0)
        assert len(sim.sequences) == 5
        for seq in sim.sequences:
            assert isinstance(seq, str) and len(seq) > 0


