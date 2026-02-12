"""
Tests for abundance-weighted graph construction.

Mathematical grounding:
  An LZGraph is a first-order Markov chain on LZ76 subpatterns.
  Transition probabilities are P(B|A) = count(A->B) / freq(A).
  Inserting a sequence with abundance k is equivalent to inserting it k times.
  Therefore:
    - Graph topology (which edges exist) is unchanged by abundance.
    - Edge counts scale linearly with abundance.
    - After normalization, transition probabilities shift toward paths
      used by high-abundance sequences.
    - P(stop|t) = T(t) / (T(t) + f(t)) changes because both T(t) and f(t)
      scale with abundance.
    - walk_log_probability of a high-abundance sequence becomes LESS negative
      (higher probability) relative to low-abundance sequences.
    - The LZPgen distribution shifts: its mean moves toward the log-probability
      of high-abundance sequences.

All of these properties are tested below.
"""

import math
import pytest
import numpy as np
import pandas as pd

from LZGraphs import AAPLZGraph, generate_kmer_dictionary
from LZGraphs.graphs.naive import NaiveLZGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A small controlled set of amino acid sequences
SEQUENCES = [
    "CASSLGQAYEQYF",
    "CASSLAGTYEQYF",
    "CASSLGQGYNEQF",
    "CASSQAGTYNEQF",
    "CASRTGQAYEQYF",
    "CASSLGQAYEQYF",   # duplicate of [0]
    "CASSLAGTYEQYF",   # duplicate of [1]
    "CASSLGQGYNEQF",   # duplicate of [2]
    "CASSLGRAYEQYF",
    "CASSLAGRYEQYF",
]


def make_aap_df(sequences, abundances=None):
    """Create a DataFrame suitable for AAPLZGraph."""
    df = pd.DataFrame({"cdr3_amino_acid": sequences})
    if abundances is not None:
        df["abundance"] = abundances
    return df


# ---------------------------------------------------------------------------
# Test 1: Abundance=1 is equivalent to no abundance column
# ---------------------------------------------------------------------------

class TestAbundanceEquivalence:
    """Inserting with abundance=1 for all must be identical to no abundance column."""

    def test_aap_abundance_ones_equals_no_abundance(self):
        """Graph built with all abundance=1 == graph built without abundance column."""
        df_no_ab = make_aap_df(SEQUENCES)
        df_ab_ones = make_aap_df(SEQUENCES, abundances=[1] * len(SEQUENCES))

        g1 = AAPLZGraph(df_no_ab, verbose=False, min_initial_state_count=0)
        g2 = AAPLZGraph(df_ab_ones, verbose=False, min_initial_state_count=0)

        # Same topology
        assert set(g1.graph.nodes()) == set(g2.graph.nodes())
        assert set(g1.graph.edges()) == set(g2.graph.edges())

        # Same edge counts
        for a, b in g1.graph.edges():
            assert g1.graph[a][b]['data'].count == g2.graph[a][b]['data'].count

        # Same transition probabilities
        for a, b in g1.graph.edges():
            assert abs(g1.graph[a][b]['data'].weight - g2.graph[a][b]['data'].weight) < 1e-12

        # Same terminal/initial state counts
        assert g1.terminal_state_counts == g2.terminal_state_counts
        assert g1.initial_state_counts == g2.initial_state_counts


# ---------------------------------------------------------------------------
# Test 2: Abundance=k is equivalent to repeating the sequence k times
# ---------------------------------------------------------------------------

class TestAbundanceEqualsRepetition:
    """Inserting seq with abundance=k must equal inserting seq k times with abundance=1."""

    def test_aap_abundance_k_equals_k_repetitions(self):
        """A single sequence with abundance=5 == that sequence repeated 5 times."""
        seq = "CASSLGQAYEQYF"
        k = 5

        # Build with repetition
        df_repeat = make_aap_df([seq] * k)
        g_repeat = AAPLZGraph(df_repeat, verbose=False, min_initial_state_count=0)

        # Build with abundance
        df_abund = make_aap_df([seq], abundances=[k])
        g_abund = AAPLZGraph(df_abund, verbose=False, min_initial_state_count=0)

        # Same topology
        assert set(g_repeat.graph.nodes()) == set(g_abund.graph.nodes())
        assert set(g_repeat.graph.edges()) == set(g_abund.graph.edges())

        # Same edge counts
        for a, b in g_repeat.graph.edges():
            assert g_repeat.graph[a][b]['data'].count == g_abund.graph[a][b]['data'].count

        # Same per-node frequencies
        for node in g_repeat.graph.nodes():
            assert g_repeat.node_outgoing_counts.get(node, 0) == \
                   g_abund.node_outgoing_counts.get(node, 0)

        # Same transition probabilities (both are trivially 1.0 for a single unique seq)
        for a, b in g_repeat.graph.edges():
            assert abs(g_repeat.graph[a][b]['data'].weight - g_abund.graph[a][b]['data'].weight) < 1e-12

        # Same terminal states
        assert g_repeat.terminal_state_counts == g_abund.terminal_state_counts

    def test_aap_mixed_abundance_equals_repetition(self):
        """Multiple sequences with different abundances == expanding each by its count."""
        seqs = ["CASSLGQAYEQYF", "CASSLAGTYEQYF", "CASSLGQGYNEQF"]
        abundances = [3, 1, 7]

        # Build with abundance
        df_abund = make_aap_df(seqs, abundances=abundances)
        g_abund = AAPLZGraph(df_abund, verbose=False, min_initial_state_count=0)

        # Build with repetition
        expanded = []
        for seq, ab in zip(seqs, abundances):
            expanded.extend([seq] * ab)
        df_repeat = make_aap_df(expanded)
        g_repeat = AAPLZGraph(df_repeat, verbose=False, min_initial_state_count=0)

        # Same edge counts
        for a, b in g_repeat.graph.edges():
            assert g_repeat.graph[a][b]['data'].count == g_abund.graph[a][b]['data'].count, \
                f"Edge {a}->{b}: repeat={g_repeat.graph[a][b]['data'].count}, " \
                f"abund={g_abund.graph[a][b]['data'].count}"

        # Same transition probabilities
        for a, b in g_repeat.graph.edges():
            assert abs(g_repeat.graph[a][b]['data'].weight - g_abund.graph[a][b]['data'].weight) < 1e-12

        # Same stop probabilities
        for state in g_repeat.terminal_state_counts:
            p_repeat = g_repeat._stop_probability_cache.get(state, 0)
            p_abund = g_abund._stop_probability_cache.get(state, 0)
            assert abs(p_repeat - p_abund) < 1e-12, \
                f"Stop prob at {state}: repeat={p_repeat}, abund={p_abund}"


# ---------------------------------------------------------------------------
# Test 3: Transition probabilities shift correctly
# ---------------------------------------------------------------------------

class TestTransitionProbabilityShift:
    """
    Given two sequences sharing a common prefix but diverging at a node,
    boosting one's abundance should increase the transition probability
    toward that sequence's path.
    """

    def test_abundance_shifts_transition_probability(self):
        """
        Consider sequences that share a prefix:
          S1 = "CASSLGQAYEQYF" (common prefix up to some node, then diverges)
          S2 = "CASSLGRAYEQYF" (diverges differently)

        With equal abundance, the branching node has equal probability to
        each branch. With S1 having higher abundance, the branch toward S1
        should have higher probability.
        """
        s1 = "CASSLGQAYEQYF"
        s2 = "CASSLGRAYEQYF"

        # Equal weight
        df_equal = make_aap_df([s1, s2], abundances=[1, 1])
        g_equal = AAPLZGraph(df_equal, verbose=False, min_initial_state_count=0)

        # S1 has 10x abundance
        df_boosted = make_aap_df([s1, s2], abundances=[10, 1])
        g_boosted = AAPLZGraph(df_boosted, verbose=False, min_initial_state_count=0)

        # Find the walk for s1
        walk_s1 = AAPLZGraph.encode_sequence(s1)

        # The walk_probability of s1 should be higher (less negative log) in boosted graph
        lp_equal = g_equal.walk_log_probability(walk_s1, verbose=False)
        lp_boosted = g_boosted.walk_log_probability(walk_s1, verbose=False)

        assert lp_boosted > lp_equal, \
            f"Expected boosted log-prob ({lp_boosted}) > equal log-prob ({lp_equal})"

    def test_abundance_does_not_change_topology(self):
        """Abundance should never create or remove edges, only change weights."""
        seqs = ["CASSLGQAYEQYF", "CASSLAGTYEQYF"]

        df_a = make_aap_df(seqs, abundances=[1, 1])
        df_b = make_aap_df(seqs, abundances=[100, 1])

        g_a = AAPLZGraph(df_a, verbose=False, min_initial_state_count=0)
        g_b = AAPLZGraph(df_b, verbose=False, min_initial_state_count=0)

        assert set(g_a.graph.nodes()) == set(g_b.graph.nodes())
        assert set(g_a.graph.edges()) == set(g_b.graph.edges())


# ---------------------------------------------------------------------------
# Test 4: Stop probability correctness
# ---------------------------------------------------------------------------

class TestStopProbability:
    """
    P(stop|t) = T(t) / (T(t) + f(t))
    With abundance, T(t) and f(t) scale accordingly.
    """

    def test_stop_probability_with_abundance(self):
        """
        Sequence "ABCDE" with abundance k:
          - Terminal node gets T(t) = k
          - If this node is also an intermediate node in other sequences,
            f(t) reflects the total outgoing edge count.

        For a single sequence with abundance k:
          - The terminal node has T(t)=k, f(t)=0
          - So P(stop) = k/(k+0) = 1.0
        """
        seq = "CASSLGQAYEQYF"

        for k in [1, 5, 100]:
            df = make_aap_df([seq], abundances=[k])
            g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

            walk = AAPLZGraph.encode_sequence(seq)
            terminal_node = walk[-1]

            # For a single-sequence graph, P(stop|terminal) = 1.0
            assert abs(g._stop_probability_cache[terminal_node] - 1.0) < 1e-12

    def test_stop_probability_two_sequences(self):
        """
        Two sequences: one ends at node T, the other passes through T.
        Abundance controls the MLE stop probability.

        s1 = "CASSLGQ" ends at some terminal T1
        s2 = "CASSLGQAYEQYF" passes through a prefix that may share nodes with s1

        We verify: P(stop|T1) = T1_terminal_count / (T1_terminal_count + T1_outgoing_count)
        """
        s1 = "CASSLGQAYEQYF"
        s2 = "CASSLGQGYNEQF"

        ab_s1 = 3
        ab_s2 = 7
        df = make_aap_df([s1, s2], abundances=[ab_s1, ab_s2])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        # Verify the MLE formula holds for every terminal state
        for state in g.terminal_state_counts:
            t_count = g.terminal_state_counts[state]
            f_count = g.node_outgoing_counts.get(state, 0)
            expected = t_count / (t_count + f_count) if (t_count + f_count) > 0 else 1.0
            actual = g._stop_probability_cache[state]
            assert abs(expected - actual) < 1e-12, \
                f"Stop prob at {state}: expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# Test 5: walk_log_probability mathematical evaluation
# ---------------------------------------------------------------------------

class TestWalkLogProbability:
    """
    P(seq) = P(init) * prod(P(edge_i)) * P(stop|last_node)

    With abundance, all three factors change:
      - P(init) = initial_count / total_initial_count
      - P(edge) = edge_count / source_node_freq
      - P(stop) = terminal_count / (terminal_count + outgoing_count)
    """

    def test_walk_probability_manual_computation(self):
        """
        Build a graph with known abundances and manually verify the
        walk_log_probability computation.
        """
        s1 = "CASSLGQAYEQYF"
        s2 = "CASSLAGTYEQYF"
        ab1, ab2 = 4, 6

        df = make_aap_df([s1, s2], abundances=[ab1, ab2])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        walk = AAPLZGraph.encode_sequence(s1)

        # Manual computation
        # P(init)
        init_node = walk[0]
        p_init = g.initial_state_probabilities[init_node]

        # P(edges)
        log_p = math.log(p_init)
        for i in range(len(walk) - 1):
            a, b = walk[i], walk[i + 1]
            p_edge = g.graph[a][b]['data'].weight
            log_p += math.log(p_edge)

        # P(stop)
        last_node = walk[-1]
        p_stop = g._stop_probability_cache.get(last_node, 1.0)
        log_p += math.log(p_stop)

        # Compare
        computed = g.walk_log_probability(walk, verbose=False)
        assert abs(computed - log_p) < 1e-10, \
            f"Manual: {log_p}, Computed: {computed}"

    def test_high_abundance_sequence_has_higher_pgen(self):
        """
        Given two sequences in the same graph, the one with higher abundance
        should have a higher (less negative) log-probability, because its
        transitions are more probable.
        """
        s1 = "CASSLGQAYEQYF"
        s2 = "CASSLAGTYEQYF"

        # s1 has 10x the abundance
        df = make_aap_df([s1, s2], abundances=[10, 1])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        walk_s1 = AAPLZGraph.encode_sequence(s1)
        walk_s2 = AAPLZGraph.encode_sequence(s2)

        lp_s1 = g.walk_log_probability(walk_s1, verbose=False)
        lp_s2 = g.walk_log_probability(walk_s2, verbose=False)

        assert lp_s1 > lp_s2, \
            f"Expected abundant seq log-prob ({lp_s1}) > rare seq log-prob ({lp_s2})"


# ---------------------------------------------------------------------------
# Test 6: Edge count linearity
# ---------------------------------------------------------------------------

class TestEdgeCountLinearity:
    """Edge counts should scale linearly with abundance."""

    def test_edge_counts_scale_linearly(self):
        """
        If we double all abundances, all edge counts should exactly double.
        """
        seqs = ["CASSLGQAYEQYF", "CASSLAGTYEQYF", "CASSLGQGYNEQF"]
        ab_base = [2, 3, 5]
        ab_doubled = [4, 6, 10]

        df_base = make_aap_df(seqs, abundances=ab_base)
        df_doubled = make_aap_df(seqs, abundances=ab_doubled)

        g_base = AAPLZGraph(df_base, verbose=False, min_initial_state_count=0)
        g_doubled = AAPLZGraph(df_doubled, verbose=False, min_initial_state_count=0)

        # Same topology
        assert set(g_base.graph.edges()) == set(g_doubled.graph.edges())

        # Counts double
        for a, b in g_base.graph.edges():
            assert g_doubled.graph[a][b]['data'].count == 2 * g_base.graph[a][b]['data'].count

        # But transition probabilities stay the same (normalization cancels)
        for a, b in g_base.graph.edges():
            assert abs(g_doubled.graph[a][b]['data'].weight - g_base.graph[a][b]['data'].weight) < 1e-12

    def test_uniform_abundance_does_not_change_probabilities(self):
        """
        Multiplying all abundances by a constant k should not change any
        transition probability, stop probability, or walk_log_probability.

        This is because all counts scale by k, and normalization divides by
        the sum which also scales by k.
        """
        seqs = SEQUENCES[:5]
        k = 17

        df_a = make_aap_df(seqs, abundances=[1] * len(seqs))
        df_b = make_aap_df(seqs, abundances=[k] * len(seqs))

        g_a = AAPLZGraph(df_a, verbose=False, min_initial_state_count=0)
        g_b = AAPLZGraph(df_b, verbose=False, min_initial_state_count=0)

        # Transition probabilities unchanged
        for a, b in g_a.graph.edges():
            assert abs(g_a.graph[a][b]['data'].weight - g_b.graph[a][b]['data'].weight) < 1e-12

        # Stop probabilities unchanged
        for state in g_a._stop_probability_cache:
            assert abs(g_a._stop_probability_cache[state] - g_b._stop_probability_cache[state]) < 1e-12

        # walk_log_probability unchanged
        for seq in seqs:
            walk = AAPLZGraph.encode_sequence(seq)
            lp_a = g_a.walk_log_probability(walk, verbose=False)
            lp_b = g_b.walk_log_probability(walk, verbose=False)
            assert abs(lp_a - lp_b) < 1e-10, \
                f"walk_log_prob changed for {seq}: {lp_a} vs {lp_b}"


# ---------------------------------------------------------------------------
# Test 7: Normalization invariants
# ---------------------------------------------------------------------------

class TestNormalizationInvariants:
    """The Markov chain structure must be maintained with abundance."""

    def test_outgoing_probabilities_sum_to_one(self):
        """For every non-terminal node, outgoing edge weights must sum to 1.0."""
        seqs = SEQUENCES[:6]
        df = make_aap_df(seqs, abundances=[5, 1, 3, 2, 8, 4])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        for node in g.graph.nodes():
            successors = list(g.graph.successors(node))
            if not successors:
                continue
            total = sum(g.graph[node][nb]['data'].weight for nb in successors)
            assert abs(total - 1.0) < 1e-10, \
                f"Node {node}: outgoing weights sum to {total}, expected 1.0"

    def test_initial_state_probabilities_sum_to_one(self):
        """Initial state probabilities must sum to 1.0."""
        seqs = SEQUENCES[:6]
        df = make_aap_df(seqs, abundances=[5, 1, 3, 2, 8, 4])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        total = sum(g.initial_state_probabilities.values())
        assert abs(total - 1.0) < 1e-10, \
            f"Initial state probabilities sum to {total}"

    def test_stop_probabilities_in_zero_one(self):
        """All stop probabilities must be in [0, 1]."""
        seqs = SEQUENCES[:6]
        df = make_aap_df(seqs, abundances=[5, 1, 3, 2, 8, 4])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        for state, p in g._stop_probability_cache.items():
            assert 0.0 <= p <= 1.0, \
                f"Stop prob at {state} = {p}, out of [0,1]"


# ---------------------------------------------------------------------------
# Test 8: Simulate consistency
# ---------------------------------------------------------------------------

class TestSimulateWithAbundance:
    """simulate() should produce valid sequences from abundance-weighted graphs."""

    def test_simulate_produces_valid_sequences(self):
        """All simulated sequences should be reconstructable amino acid strings."""
        seqs = SEQUENCES[:6]
        df = make_aap_df(seqs, abundances=[10, 1, 5, 3, 8, 2])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        simulated = g.simulate(50, seed=42)
        assert len(simulated) == 50

        for s in simulated:
            assert isinstance(s, str)
            assert len(s) > 0
            # Must contain only valid amino acids
            assert all(c in AAPLZGraph.VALID_AMINO_ACIDS for c in s)

    def test_simulate_distribution_biased_toward_abundant(self):
        """
        If one sequence has very high abundance relative to others,
        simulate() should produce sequences that share its subpatterns
        more frequently (statistically).
        """
        s_common = "CASSLGQAYEQYF"
        s_rare = "CASSQAGTYNEQF"

        # Extreme abundance ratio
        df = make_aap_df([s_common, s_rare], abundances=[100, 1])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        simulated = g.simulate(200, seed=42)

        # The common sequence itself should appear far more than the rare one
        count_common = sum(1 for s in simulated if s == s_common)
        count_rare = sum(1 for s in simulated if s == s_rare)
        assert count_common > count_rare, \
            f"Expected common ({count_common}) > rare ({count_rare})"


# ---------------------------------------------------------------------------
# Test 9: LZPgen distribution shift
# ---------------------------------------------------------------------------

class TestLZPgenDistributionShift:
    """
    The LZPgen distribution (Monte Carlo) should shift when abundance changes.

    Specifically: boosting the abundance of sequence S should shift the
    distribution's mean toward the log-probability of S.
    """

    def test_lzpgen_mean_shifts_with_abundance(self):
        """
        Build two graphs from the same sequences but different abundances.
        The graph with higher abundance for a particular sequence should
        produce an LZPgen distribution with a higher (less negative) mean,
        because the dominant sequence's transitions get more probability mass.
        """
        seqs = SEQUENCES[:5]

        # Uniform abundance
        df_uniform = make_aap_df(seqs, abundances=[1, 1, 1, 1, 1])
        g_uniform = AAPLZGraph(df_uniform, verbose=False, min_initial_state_count=0)

        # Boost first sequence to abundance 50
        df_boosted = make_aap_df(seqs, abundances=[50, 1, 1, 1, 1])
        g_boosted = AAPLZGraph(df_boosted, verbose=False, min_initial_state_count=0)

        dist_uniform = g_uniform.lzpgen_distribution(n=500, seed=42)
        dist_boosted = g_boosted.lzpgen_distribution(n=500, seed=42)

        mean_uniform = np.mean(dist_uniform)
        mean_boosted = np.mean(dist_boosted)

        # The boosted graph concentrates probability on one sequence's path,
        # so simulated walks are more likely to follow high-probability paths
        # → higher (less negative) mean log-prob
        assert mean_boosted > mean_uniform, \
            f"Expected boosted mean ({mean_boosted:.3f}) > uniform mean ({mean_uniform:.3f})"


# ---------------------------------------------------------------------------
# Test 10: Exact moments shift (for DAG graphs)
# ---------------------------------------------------------------------------

class TestExactMomentsWithAbundance:
    """
    lzpgen_moments() should reflect abundance weighting.
    """

    def test_moments_shift_with_abundance(self):
        """Mean from exact moments should shift in the same direction as MC."""
        seqs = SEQUENCES[:5]

        df_uniform = make_aap_df(seqs, abundances=[1, 1, 1, 1, 1])
        g_uniform = AAPLZGraph(df_uniform, verbose=False, min_initial_state_count=0)

        df_boosted = make_aap_df(seqs, abundances=[50, 1, 1, 1, 1])
        g_boosted = AAPLZGraph(df_boosted, verbose=False, min_initial_state_count=0)

        m_uniform = g_uniform.lzpgen_moments()
        m_boosted = g_boosted.lzpgen_moments()

        assert m_boosted['mean'] > m_uniform['mean'], \
            f"Expected boosted mean ({m_boosted['mean']:.3f}) > uniform mean ({m_uniform['mean']:.3f})"

    def test_dominant_sequence_pgen_near_mean(self):
        """
        When one sequence has extremely high abundance, the exact mean of the
        LZPgen distribution should converge toward that sequence's walk_log_probability.

        This is because the graph's transitions concentrate on the dominant path,
        making simulated walks overwhelmingly follow that path.
        """
        seqs = SEQUENCES[:5]
        dominant_seq = seqs[0]

        df_dominant = make_aap_df(seqs, abundances=[10000, 1, 1, 1, 1])
        g = AAPLZGraph(df_dominant, verbose=False, min_initial_state_count=0)

        walk = AAPLZGraph.encode_sequence(dominant_seq)
        lp_dominant = g.walk_log_probability(walk, verbose=False)

        m = g.lzpgen_moments()

        # The mean should be close to the dominant sequence's log-prob.
        # With 10000:1 ratio, both should be near zero; we use absolute tolerance.
        assert abs(m['mean'] - lp_dominant) < 0.5, \
            f"Mean ({m['mean']:.4f}) not close to dominant seq log-prob ({lp_dominant:.4f})"


# ---------------------------------------------------------------------------
# Test 11: Recalculate consistency after abundance construction
# ---------------------------------------------------------------------------

class TestRecalculateWithAbundance:
    """
    recalculate() should reproduce the same derived state from raw counts
    that were set during abundance-weighted construction.
    """

    def test_recalculate_preserves_state(self):
        """After recalculate(), all derived quantities should remain unchanged."""
        seqs = SEQUENCES[:5]
        df = make_aap_df(seqs, abundances=[3, 7, 2, 5, 1])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        # Save state before
        weights_before = {
            (a, b): g.graph[a][b]['data'].weight
            for a, b in g.graph.edges()
        }
        stop_before = dict(g._stop_probability_cache)

        # Recalculate
        g.recalculate()

        # Verify
        for (a, b), w in weights_before.items():
            assert abs(g.graph[a][b]['data'].weight - w) < 1e-12

        for state, p in stop_before.items():
            assert abs(g._stop_probability_cache[state] - p) < 1e-12


# ---------------------------------------------------------------------------
# Test 12: NaiveLZGraph abundance support
# ---------------------------------------------------------------------------

class TestNaiveLZGraphAbundance:
    """NaiveLZGraph should also support abundance via the abundances parameter."""

    def test_naive_abundance_equals_repetition(self):
        """abundance=[k] should equal repeating the sequence k times."""
        seqs = ["ATGCATGC", "ATGCGATC", "GATCATGC"]
        dictionary = generate_kmer_dictionary(6)
        k = 5

        # With repetition
        expanded = []
        for seq in seqs:
            expanded.extend([seq] * k)
        g_repeat = NaiveLZGraph(expanded, dictionary, verbose=False)

        # With abundance
        g_abund = NaiveLZGraph(seqs, dictionary, verbose=False, abundances=[k, k, k])

        # Same edge counts
        for a, b in g_repeat.graph.edges():
            assert g_repeat.graph[a][b]['data'].count == g_abund.graph[a][b]['data'].count

        # Same transition probabilities
        for a, b in g_repeat.graph.edges():
            assert abs(g_repeat.graph[a][b]['data'].weight - g_abund.graph[a][b]['data'].weight) < 1e-12

    def test_naive_mixed_abundance(self):
        """Different abundance per sequence."""
        seqs = ["ATGCATGC", "ATGCGATC", "GATCATGC"]
        abundances = [3, 1, 7]
        dictionary = generate_kmer_dictionary(6)

        # Expanded
        expanded = []
        for seq, ab in zip(seqs, abundances):
            expanded.extend([seq] * ab)
        g_repeat = NaiveLZGraph(expanded, dictionary, verbose=False)

        # With abundance
        g_abund = NaiveLZGraph(seqs, dictionary, verbose=False, abundances=abundances)

        for a, b in g_repeat.graph.edges():
            assert g_repeat.graph[a][b]['data'].count == g_abund.graph[a][b]['data'].count


# ---------------------------------------------------------------------------
# Test 13: EdgeData.record count parameter
# ---------------------------------------------------------------------------

class TestEdgeDataRecordCount:
    """Direct unit test for EdgeData.record(count=k)."""

    def test_record_with_count(self):
        from LZGraphs.graphs.edge_data import EdgeData

        ed = EdgeData()
        ed.record(v_gene="TRBV1", j_gene="TRBJ1", count=5)

        assert ed.count == 5
        assert ed.v_genes["TRBV1"] == 5
        assert ed.j_genes["TRBJ1"] == 5

    def test_record_multiple_with_count(self):
        from LZGraphs.graphs.edge_data import EdgeData

        ed = EdgeData()
        ed.record(v_gene="TRBV1", count=3)
        ed.record(v_gene="TRBV2", count=7)
        ed.record(count=2)  # no genes

        assert ed.count == 12  # 3 + 7 + 2
        assert ed.v_genes["TRBV1"] == 3
        assert ed.v_genes["TRBV2"] == 7

    def test_record_default_count_is_one(self):
        """Backward compatibility: record() without count still increments by 1."""
        from LZGraphs.graphs.edge_data import EdgeData

        ed = EdgeData()
        ed.record()
        ed.record()
        ed.record()

        assert ed.count == 3


# ---------------------------------------------------------------------------
# Test 14: Analytical distribution with abundance
# ---------------------------------------------------------------------------

class TestAnalyticalDistributionWithAbundance:
    """lzpgen_analytical_distribution() should work with abundance-weighted graphs."""

    def test_analytical_distribution_runs(self):
        """Should not error on abundance-weighted graph."""
        seqs = SEQUENCES[:5]
        df = make_aap_df(seqs, abundances=[10, 1, 5, 3, 8])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        dist = g.lzpgen_analytical_distribution()
        assert dist.n_components > 0
        assert dist.mean() < 0  # log-probabilities are negative

    def test_analytical_vs_mc_consistency(self):
        """Analytical mean should be close to MC mean."""
        seqs = SEQUENCES[:5]
        df = make_aap_df(seqs, abundances=[10, 1, 5, 3, 8])
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        dist_analytical = g.lzpgen_analytical_distribution()
        dist_mc = g.lzpgen_distribution(n=2000, seed=42)

        mean_analytical = dist_analytical.mean()
        mean_mc = np.mean(dist_mc)

        assert abs(mean_analytical - mean_mc) < 1.0, \
            f"Analytical mean ({mean_analytical:.3f}) too far from MC mean ({mean_mc:.3f})"


# ---------------------------------------------------------------------------
# Test 15: Gene data with abundance
# ---------------------------------------------------------------------------

class TestGeneDataWithAbundance:
    """Gene data (V/J) should scale with abundance."""

    def test_gene_counts_scale_with_abundance(self):
        """V/J gene counts on edges should reflect abundance weighting."""
        seqs = ["CASSLGQAYEQYF", "CASSLAGTYEQYF"]
        v_genes = ["TRBV1", "TRBV2"]
        j_genes = ["TRBJ1", "TRBJ2"]
        abundances = [5, 3]

        df = pd.DataFrame({
            "cdr3_amino_acid": seqs,
            "V": v_genes,
            "J": j_genes,
            "abundance": abundances,
        })
        g = AAPLZGraph(df, verbose=False, min_initial_state_count=0)

        # Build expanded version for comparison
        expanded_seqs = seqs[0:1] * 5 + seqs[1:2] * 3
        expanded_v = v_genes[0:1] * 5 + v_genes[1:2] * 3
        expanded_j = j_genes[0:1] * 5 + j_genes[1:2] * 3
        df_expand = pd.DataFrame({
            "cdr3_amino_acid": expanded_seqs,
            "V": expanded_v,
            "J": expanded_j,
        })
        g_expand = AAPLZGraph(df_expand, verbose=False, min_initial_state_count=0)

        # Edge gene counts must match
        for a, b in g.graph.edges():
            ed = g.graph[a][b]['data']
            ed_exp = g_expand.graph[a][b]['data']
            assert ed.v_genes == ed_exp.v_genes, \
                f"V gene mismatch on {a}->{b}: {ed.v_genes} vs {ed_exp.v_genes}"
            assert ed.j_genes == ed_exp.j_genes, \
                f"J gene mismatch on {a}->{b}: {ed.j_genes} vs {ed_exp.j_genes}"
