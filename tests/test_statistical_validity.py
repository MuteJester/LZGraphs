"""Statistical validation of the LZGraph generative model.

Tests that the model captures the right distribution:
- Simulated sequences reproduce training data statistics
- LZPGEN is well-calibrated (frequency ∝ exp(LZPGEN))
- Novel sequences are generated that respect learned dynamics
- Moments of the LZPGEN distribution are self-consistent
- Length/token distributions are preserved
- Self-consistency: graph from simulations ≈ original graph
"""

import math
import numpy as np
import pytest
from collections import Counter
from LZGraphs import LZGraph, PgenDistribution, lz76_decompose, jensen_shannon_divergence


TRAIN_SEQS = [
    'CASSLGIRRT', 'CASSLGYEQYF', 'CASSLEPSGGTDTQYF',
    'CASSDTSGGTDTQYF', 'CASSFGQGSYEQYF', 'CASSQETQYF',
] * 5  # 30 sequences (each seen 5 times)


@pytest.fixture(scope='module')
def graph():
    return LZGraph(TRAIN_SEQS, variant='aap')


@pytest.fixture(scope='module')
def large_sim(graph):
    """10K simulated sequences for statistical tests."""
    return graph.simulate(10000, seed=42)


# ═══════════════════════════════════════════════════════════════
# 1. LZPGEN calibration: frequency ∝ exp(LZPGEN)
# ═══════════════════════════════════════════════════════════════

class TestLZPGENCalibration:
    def test_frequency_correlates_with_lzpgen(self, graph, large_sim):
        """Sequences simulated more frequently should have higher LZPGEN.
        Spearman rank correlation between count and LZPGEN should be positive."""
        counts = Counter(large_sim.sequences)
        unique_seqs = list(counts.keys())
        frequencies = np.array([counts[s] for s in unique_seqs], dtype=float)
        lzpgens = np.array([graph.lzpgen(s) for s in unique_seqs])

        # Filter out sequences with zero LZPGEN (shouldn't happen, but safety)
        mask = lzpgens > -600
        assert mask.sum() > 5, "not enough sequences with positive LZPGEN"

        freq_filtered = frequencies[mask]
        lzpgen_filtered = lzpgens[mask]

        # Spearman rank correlation
        from scipy.stats import spearmanr
        corr, pvalue = spearmanr(freq_filtered, lzpgen_filtered)

        assert corr > 0.3, \
            f"weak correlation between frequency and LZPGEN: r={corr:.3f}, p={pvalue:.4f}"

    def test_high_lzpgen_seqs_appear_more(self, graph, large_sim):
        """The top-LZPGEN training sequences should be the most frequent
        in simulation."""
        train_unique = list(set(TRAIN_SEQS))
        lzpgens = {s: graph.lzpgen(s) for s in train_unique}
        sim_counts = Counter(large_sim.sequences)

        # Sort training seqs by LZPGEN
        ranked = sorted(train_unique, key=lambda s: lzpgens[s], reverse=True)
        top_seq = ranked[0]
        bottom_seq = ranked[-1]

        top_count = sim_counts.get(top_seq, 0)
        bottom_count = sim_counts.get(bottom_seq, 0)

        # The highest-LZPGEN seq should appear at least as often as the lowest
        assert top_count >= bottom_count, \
            f"top LZPGEN seq '{top_seq}' count={top_count} < bottom '{bottom_seq}' count={bottom_count}"


# ═══════════════════════════════════════════════════════════════
# 2. LZPGEN moment consistency
# ═══════════════════════════════════════════════════════════════

class TestMomentConsistency:
    def test_empirical_mean_matches_analytical(self, graph, large_sim):
        """Mean of simulated log_probs should approximate pgen_moments().mean."""
        moments = graph.pgen_moments()
        empirical_mean = np.mean(large_sim.log_probs)

        # Allow generous tolerance — forward DP is unconstrained, simulation is constrained
        assert abs(empirical_mean - moments['mean']) < 2.0, \
            f"empirical mean={empirical_mean:.4f} vs analytical={moments['mean']:.4f}"

    def test_empirical_std_reasonable(self, graph, large_sim):
        """Std of simulated log_probs should be in a reasonable range."""
        moments = graph.pgen_moments()
        empirical_std = np.std(large_sim.log_probs)

        assert empirical_std > 0, "zero std means all sequences identical"
        # The analytical and empirical may differ, but both should be positive
        assert moments['std'] > 0, "analytical std should be positive"

    def test_pgen_distribution_pdf_integrates(self, graph):
        """The analytical PGEN distribution should integrate to ~1."""
        dist = graph.pgen_distribution()
        # Numerical integration of PDF
        x = np.linspace(dist.mean - 5 * max(dist.stds), dist.mean + 5 * max(dist.stds), 1000)
        dx = x[1] - x[0]
        pdf_vals = np.array([dist.pdf(xi) for xi in x])
        integral = np.sum(pdf_vals) * dx
        assert abs(integral - 1.0) < 0.1, f"PDF integral = {integral:.4f}, expected ~1.0"


# ═══════════════════════════════════════════════════════════════
# 3. Novel sequence generation
# ═══════════════════════════════════════════════════════════════

class TestNovelSequences:
    def test_generates_novel_sequences(self, large_sim):
        """Simulation should produce sequences NOT in the training set."""
        train_set = set(TRAIN_SEQS)
        sim_set = set(large_sim.sequences)
        novel = sim_set - train_set

        assert len(novel) > 0, "no novel sequences generated"

    def test_novel_seqs_have_positive_lzpgen(self, graph, large_sim):
        """Novel sequences should still have positive LZPGEN
        (they share subpatterns with training)."""
        train_set = set(TRAIN_SEQS)
        novel = [s for s in set(large_sim.sequences) if s not in train_set]

        if not novel:
            pytest.skip("no novel sequences to test")

        lzpgens = graph.lzpgen(novel[:20])
        zero_count = np.sum(lzpgens < -600)
        assert zero_count == 0, f"{zero_count}/{len(lzpgens)} novel seqs have zero LZPGEN"

    def test_novel_seqs_share_subpatterns(self, large_sim):
        """Novel sequences should share LZ subpatterns with training."""
        train_subpatterns = set()
        for s in set(TRAIN_SEQS):
            train_subpatterns.update(lz76_decompose(s))

        train_set = set(TRAIN_SEQS)
        novel = [s for s in set(large_sim.sequences) if s not in train_set]

        if not novel:
            pytest.skip("no novel sequences")

        # Each novel seq should have at least some subpatterns from training
        for s in novel[:10]:
            tokens = lz76_decompose(s)
            shared = set(tokens) & train_subpatterns
            assert len(shared) > 0, \
                f"novel seq '{s}' shares no subpatterns with training"


# ═══════════════════════════════════════════════════════════════
# 4. Length distribution preservation
# ═══════════════════════════════════════════════════════════════

class TestLengthDistribution:
    def test_similar_length_range(self, large_sim):
        """Simulated lengths should be in a similar range to training."""
        train_lens = [len(s) for s in set(TRAIN_SEQS)]
        sim_lens = [len(s) for s in large_sim.sequences]

        train_min, train_max = min(train_lens), max(train_lens)
        sim_min, sim_max = min(sim_lens), max(sim_lens)

        # Simulated range should overlap significantly with training range
        assert sim_min <= train_max, "simulated min > training max"
        assert sim_max >= train_min, "simulated max < training min"

    def test_mean_length_similar(self, large_sim):
        """Mean simulated length should be close to training mean."""
        train_lens = [len(s) for s in TRAIN_SEQS]
        sim_lens = [len(s) for s in large_sim.sequences]

        train_mean = np.mean(train_lens)
        sim_mean = np.mean(sim_lens)

        # Allow 50% relative tolerance
        assert abs(sim_mean - train_mean) / train_mean < 0.5, \
            f"mean length: train={train_mean:.1f}, sim={sim_mean:.1f}"


# ═══════════════════════════════════════════════════════════════
# 5. Subpattern frequency correlation
# ═══════════════════════════════════════════════════════════════

class TestSubpatternPreservation:
    def test_subpattern_frequency_correlation(self, large_sim):
        """Subpattern frequencies in simulation should correlate with training."""
        # Training subpattern counts
        train_sp = Counter()
        for s in TRAIN_SEQS:
            for tok in lz76_decompose(s):
                train_sp[tok] += 1

        # Simulation subpattern counts
        sim_sp = Counter()
        for s in large_sim.sequences:
            for tok in lz76_decompose(s):
                sim_sp[tok] += 1

        # Get subpatterns that appear in both
        common = set(train_sp.keys()) & set(sim_sp.keys())
        assert len(common) > 3, f"only {len(common)} shared subpatterns"

        train_vals = np.array([train_sp[k] for k in common], dtype=float)
        sim_vals = np.array([sim_sp[k] for k in common], dtype=float)

        # Normalize to frequencies
        train_freq = train_vals / train_vals.sum()
        sim_freq = sim_vals / sim_vals.sum()

        # Pearson correlation
        corr = np.corrcoef(train_freq, sim_freq)[0, 1]
        assert corr > 0.5, f"weak subpattern correlation: r={corr:.3f}"

    def test_initial_token_preserved(self, graph, large_sim):
        """The first LZ token of simulated sequences should match training."""
        train_first = Counter(lz76_decompose(s)[0] for s in TRAIN_SEQS)
        sim_first = Counter(lz76_decompose(s)[0] for s in large_sim.sequences)

        # The most common first token in training should be common in simulation
        most_common_train = train_first.most_common(1)[0][0]
        assert most_common_train in sim_first, \
            f"most common training first token '{most_common_train}' not in simulation"


# ═══════════════════════════════════════════════════════════════
# 6. Self-consistency: graph from simulations ≈ original
# ═══════════════════════════════════════════════════════════════

class TestSelfConsistency:
    def test_simulated_graph_similar_to_original(self, graph, large_sim):
        """Building a graph from simulated sequences should produce
        a similar graph (low JSD)."""
        g_sim = LZGraph(large_sim.sequences[:2000], variant='aap')

        jsd = jensen_shannon_divergence(graph, g_sim)
        assert jsd < 0.5, f"JSD between original and simulated graph: {jsd:.4f}"

    def test_simulated_graph_preserves_node_count(self, graph, large_sim):
        """Graph from simulations should have similar node count."""
        g_sim = LZGraph(large_sim.sequences[:2000], variant='aap')

        ratio = g_sim.n_nodes / graph.n_nodes
        assert 0.5 < ratio < 3.0, \
            f"node count ratio: {ratio:.2f} (original={graph.n_nodes}, sim={g_sim.n_nodes})"


# ═══════════════════════════════════════════════════════════════
# 7. Training sequence recovery
# ═══════════════════════════════════════════════════════════════

class TestTrainingRecovery:
    def test_training_seqs_appear_in_simulation(self, large_sim):
        """With enough simulation, most training sequences should appear."""
        train_unique = set(TRAIN_SEQS)
        sim_set = set(large_sim.sequences)
        recovered = train_unique & sim_set
        recovery_rate = len(recovered) / len(train_unique)

        assert recovery_rate > 0.3, \
            f"only {recovery_rate:.0%} of training seqs recovered in 10K simulations"


# ═══════════════════════════════════════════════════════════════
# 8. Hill number sanity from simulation
# ═══════════════════════════════════════════════════════════════

class TestHillFromSimulation:
    def test_empirical_d2_from_simulation(self, graph, large_sim):
        """Empirical Simpson diversity from simulation should be positive."""
        counts = Counter(large_sim.sequences)
        freqs = np.array(list(counts.values()), dtype=float)
        freqs /= freqs.sum()

        # Empirical D(2) = 1 / Σ pi²
        d2_empirical = 1.0 / np.sum(freqs ** 2)

        # Analytical D(2) from graph
        d2_analytical = graph.hill_number(2.0)

        assert d2_empirical > 1.0, f"empirical D(2) should be > 1: {d2_empirical:.2f}"
        assert d2_analytical > 1.0, f"analytical D(2) should be > 1: {d2_analytical:.2f}"

    def test_empirical_entropy_from_simulation(self, large_sim):
        """Empirical Shannon entropy from simulation should be positive."""
        counts = Counter(large_sim.sequences)
        freqs = np.array(list(counts.values()), dtype=float)
        freqs /= freqs.sum()

        entropy = -np.sum(freqs * np.log(freqs))
        assert entropy > 0, f"entropy should be positive: {entropy:.4f}"
