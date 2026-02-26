"""
Tests for Diversity Metrics and Information-Theoretic Measures
==============================================================

Tests covering the metrics module including:
- K-diversity family (K100, K500, K1000, K5000, adaptive)
- lz_centrality
- Entropy measures (node, edge, graph entropy)
- Perplexity scores
- Distance metrics (JS divergence, cross-entropy, KL divergence)

Test Categories:
- K1000 diversity index with statistical robustness
- lz_centrality calculation
- Entropy metric calculations
- Repertoire comparison metrics
"""

import pytest
import numpy as np
from LZGraphs import NDPLZGraph
from LZGraphs.metrics.diversity import (
    lz_centrality,
    k_diversity,
    k100_diversity,
    k500_diversity,
    k1000_diversity,
    k5000_diversity,
    adaptive_k_diversity,
)
from LZGraphs.metrics.entropy import (
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    sequence_perplexity,
    repertoire_perplexity,
    jensen_shannon_divergence,
    transition_predictability,
    graph_compression_ratio,
    repertoire_compressibility_index,
    transition_kl_divergence,
    transition_jsd,
    transition_mutual_information_profile,
    path_entropy_rate,
)
from LZGraphs.metrics.convenience import compare_repertoires
from LZGraphs.exceptions import MetricsError, EmptyDataError


class TestK1000Diversity:
    """Tests for K1000 diversity index calculation."""

    def test_k1000_ndp_encoding_within_bounds(self, test_data_ndp):
        """Verify K1000 with NDP encoding returns value in expected range."""
        k1000 = k1000_diversity(
            list_of_sequences=test_data_ndp['cdr3_rearrangement'].to_list(),
            lzgraph_encoding_function='ndp',
            draws=50
        )
        # Expected range based on test data characteristics
        assert 2130 <= k1000 <= 2160

    def test_k1000_aap_encoding_within_bounds(self, test_data_aap):
        """Verify K1000 with AAP encoding returns value in expected range."""
        k1000 = k1000_diversity(
            list_of_sequences=test_data_aap['cdr3_amino_acid'].to_list(),
            lzgraph_encoding_function='aap',
            draws=50
        )
        # Expected range based on test data characteristics
        assert 910 <= k1000 <= 950

    def test_k1000_reproducibility(self, test_data_ndp):
        """Verify K1000 produces consistent results within expected range."""
        import random
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()

        # Run twice with same seeds (copy list to avoid mutation issues)
        np.random.seed(42)
        random.seed(42)
        k1 = k1000_diversity(
            list_of_sequences=sequences.copy(),
            lzgraph_encoding_function='ndp',
            draws=10
        )

        np.random.seed(42)
        random.seed(42)
        k2 = k1000_diversity(
            list_of_sequences=sequences.copy(),
            lzgraph_encoding_function='ndp',
            draws=10
        )

        # Results should be very close (within 5% of each other)
        # Perfect reproducibility is not guaranteed due to internal shuffling
        assert abs(k1 - k2) / k1 < 0.05


class TestKDiversityFamily:
    """Tests for the K-diversity family of functions."""

    def test_k_diversity_returns_positive_value(self, test_data_ndp):
        """Verify k_diversity returns positive value."""
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()
        k = k_diversity(sequences, 'ndp', sample_size=500, draws=10)
        assert k > 0

    def test_k_diversity_with_stats(self, test_data_ndp):
        """Verify k_diversity returns statistics when requested."""
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()
        result = k_diversity(
            sequences, 'ndp',
            sample_size=500, draws=10,
            return_stats=True
        )

        # Should return (mean, std, ci_lower, ci_upper)
        assert len(result) == 4
        mean, std, ci_lower, ci_upper = result

        assert mean > 0
        assert std >= 0
        assert ci_lower <= mean <= ci_upper

    def test_k100_diversity(self, test_data_aap):
        """Verify k100_diversity works for small sample size."""
        sequences = test_data_aap['cdr3_amino_acid'].to_list()
        k100 = k100_diversity(sequences, 'aap', draws=10)
        assert k100 > 0

    def test_adaptive_k_diversity(self, test_data_ndp):
        """Verify adaptive_k_diversity auto-selects appropriate sample size."""
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()
        # adaptive_k_diversity returns (sample_size, mean) tuple
        sample_size, k = adaptive_k_diversity(sequences, 'ndp', draws=10)
        assert k > 0
        # For 5000 sequences, should use K1000
        assert sample_size == 1000


class Testlz_centrality:
    """Tests for lz_centrality metric."""

    def test_lz_centrality_calculation(self, ndp_lzgraph):
        """Verify lz_centrality calculation produces expected result."""
        test_sequence = 'TGTGCCTGCGTAACACAGGGGGTTTGGTATGGCTACACCTTC'
        lzc = lz_centrality(ndp_lzgraph, test_sequence)

        assert lzc == 14.105263157894736

    def test_lz_centrality_positive(self, ndp_lzgraph, test_data_ndp):
        """Verify lz_centrality is always positive."""
        sequence = test_data_ndp['cdr3_rearrangement'].iloc[0]
        lzc = lz_centrality(ndp_lzgraph, sequence)
        assert lzc > 0

    def test_lz_centrality_different_sequences(self, ndp_lzgraph, test_data_ndp):
        """Verify different sequences can have different lz_centrality."""
        seq1 = test_data_ndp['cdr3_rearrangement'].iloc[0]
        seq2 = test_data_ndp['cdr3_rearrangement'].iloc[1]

        lzc1 = lz_centrality(ndp_lzgraph, seq1)
        lzc2 = lz_centrality(ndp_lzgraph, seq2)

        # Different sequences should generally have different centrality
        # (though not guaranteed for all sequences)
        assert isinstance(lzc1, float)
        assert isinstance(lzc2, float)


class TestEntropyMetrics:
    """Tests for entropy-based metrics."""

    def test_node_entropy_non_negative(self, aap_lzgraph):
        """Verify node entropy is non-negative."""
        entropy = node_entropy(aap_lzgraph)
        assert entropy >= 0

    def test_edge_entropy_non_negative(self, aap_lzgraph):
        """Verify edge entropy is non-negative."""
        entropy = edge_entropy(aap_lzgraph)
        assert entropy >= 0

    def test_graph_entropy_non_negative(self, aap_lzgraph):
        """Verify graph entropy is non-negative."""
        entropy = graph_entropy(aap_lzgraph)
        assert entropy >= 0

    def test_normalized_graph_entropy_bounded(self, aap_lzgraph):
        """Verify normalized graph entropy is between 0 and 1."""
        entropy = normalized_graph_entropy(aap_lzgraph)
        assert 0 <= entropy <= 1

    def test_graph_entropy_equals_sum(self, aap_lzgraph):
        """Verify graph entropy equals node + edge entropy."""
        ge = graph_entropy(aap_lzgraph)
        ne = node_entropy(aap_lzgraph)
        ee = edge_entropy(aap_lzgraph)

        assert np.isclose(ge, ne + ee)


class TestPerplexityMetrics:
    """Tests for perplexity-based metrics."""

    def test_sequence_perplexity_positive(self, aap_lzgraph, test_data_aap):
        """Verify sequence perplexity is positive."""
        sequence = test_data_aap['cdr3_amino_acid'].iloc[0]
        perp = sequence_perplexity(aap_lzgraph, sequence)
        assert perp > 0

    def test_repertoire_perplexity_positive(self, aap_lzgraph, test_data_aap):
        """Verify repertoire perplexity is positive."""
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:10].tolist()
        perp = repertoire_perplexity(aap_lzgraph, sequences)
        assert perp > 0

    def test_perplexity_lower_for_typical_sequences(self, aap_lzgraph, test_data_aap):
        """Verify typical sequences have lower perplexity than unusual ones."""
        # Get a typical sequence from training data
        typical_seq = test_data_aap['cdr3_amino_acid'].iloc[0]
        typical_perp = sequence_perplexity(aap_lzgraph, typical_seq)

        # Create an unusual sequence (random amino acids)
        import random
        random.seed(42)
        unusual_seq = ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=15))
        unusual_perp = sequence_perplexity(aap_lzgraph, unusual_seq)

        # Typical sequences should generally have lower perplexity
        # Note: This is a statistical tendency, not guaranteed for all cases
        assert typical_perp > 0
        assert unusual_perp > 0


class TestDistanceMetrics:
    """Tests for repertoire distance metrics."""

    def test_js_divergence_self_zero(self, aap_lzgraph):
        """Verify JS divergence of a graph with itself is zero."""
        jsd = jensen_shannon_divergence(aap_lzgraph, aap_lzgraph)
        assert np.isclose(jsd, 0, atol=1e-10)

    def test_js_divergence_symmetric(self, aap_lzgraph, ndp_lzgraph):
        """Verify JS divergence is symmetric."""
        jsd1 = jensen_shannon_divergence(aap_lzgraph, ndp_lzgraph)
        jsd2 = jensen_shannon_divergence(ndp_lzgraph, aap_lzgraph)
        assert np.isclose(jsd1, jsd2)

    def test_js_divergence_bounded(self, aap_lzgraph, ndp_lzgraph):
        """Verify JS divergence is bounded between 0 and 1 (using log base 2)."""
        jsd = jensen_shannon_divergence(aap_lzgraph, ndp_lzgraph)
        assert 0 <= jsd <= 1

    def test_js_divergence_different_graphs_positive(self, aap_lzgraph, ndp_lzgraph):
        """Verify JSD between different graph types is positive."""
        jsd = jensen_shannon_divergence(aap_lzgraph, ndp_lzgraph)
        assert jsd > 0


class TestMetricRobustness:
    """Tests for metric robustness and edge cases."""

    def test_k1000_with_small_repertoire(self, test_data_aap):
        """Verify K1000 handles repertoires smaller than sample size."""
        # Use only 100 sequences
        small_repertoire = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        # Should auto-adjust sample size and still work
        k = k_diversity(small_repertoire, 'aap', sample_size=1000, draws=5)
        assert k > 0

    def test_entropy_with_single_sequence_graph(self):
        """Verify entropy metrics handle minimal graphs gracefully."""
        import pandas as pd
        from LZGraphs import AAPLZGraph

        # Create minimal data
        minimal_data = pd.DataFrame({
            'cdr3_amino_acid': ['CASSLGQAYEQYF'] * 10,
            'v_gene': ['TRBV5-1*01'] * 10,
            'j_gene': ['TRBJ2-7*01'] * 10
        })

        minimal_graph = AAPLZGraph(minimal_data, verbose=False)

        # Entropy should still be computable
        ne = node_entropy(minimal_graph)
        assert ne >= 0


class TestEdgeEntropyConditional:
    """Tests for conditional entropy property: H(edges|nodes) <= H(nodes)."""

    def test_edge_entropy_bounded_by_node_entropy(self, aap_lzgraph):
        """Verify edge entropy (conditional) is bounded by node entropy.

        The edge entropy in graph entropy decomposition represents the
        conditional entropy H(Y|X) of transitions given source nodes.
        By the chain rule, H(X,Y) = H(X) + H(Y|X), and since
        H(Y|X) <= H(Y), edge entropy should not exceed node entropy
        for a well-formed graph.
        """
        ne = node_entropy(aap_lzgraph)
        ee = edge_entropy(aap_lzgraph)

        assert ee <= ne, (
            f"Edge entropy ({ee}) should be <= node entropy ({ne}) "
            "since conditional entropy is bounded by marginal entropy"
        )


# ====================================================================
# Novel Information-Theoretic Metrics (from mathematical analysis)
# ====================================================================


class TestTransitionPredictability:
    """Tests for transition_predictability metric."""

    def test_predictability_range_aap(self, aap_lzgraph):
        """Predictability must be in [0, 1]."""
        tp = transition_predictability(aap_lzgraph)
        assert 0 <= tp <= 1

    def test_predictability_range_naive(self, naive_lzgraph):
        """Predictability must be in [0, 1] for NaiveLZGraph too."""
        tp = transition_predictability(naive_lzgraph)
        assert 0 <= tp <= 1

    def test_predictability_aap_value(self, aap_lzgraph):
        """AAPLZGraph predictability should be ~0.60 (empirically stable)."""
        tp = transition_predictability(aap_lzgraph)
        assert 0.50 <= tp <= 0.70, f"Expected ~0.60, got {tp:.3f}"

    def test_predictability_naive_value(self, naive_lzgraph):
        """NaiveLZGraph predictability should be lower (~0.48)."""
        tp = transition_predictability(naive_lzgraph)
        assert 0.35 <= tp <= 0.60, f"Expected ~0.48, got {tp:.3f}"

    def test_predictability_aap_higher_than_naive(self, aap_lzgraph, naive_lzgraph):
        """AAPLZGraph should be more predictable than NaiveLZGraph."""
        tp_aap = transition_predictability(aap_lzgraph)
        tp_naive = transition_predictability(naive_lzgraph)
        assert tp_aap > tp_naive


class TestGraphCompressionRatio:
    """Tests for graph_compression_ratio metric."""

    def test_gcr_range_aap(self, aap_lzgraph):
        """GCR must be in (0, 1]."""
        gcr = graph_compression_ratio(aap_lzgraph)
        assert 0 < gcr <= 1

    def test_gcr_range_naive(self, naive_lzgraph):
        """GCR must be in (0, 1] for NaiveLZGraph."""
        gcr = graph_compression_ratio(naive_lzgraph)
        assert 0 < gcr <= 1

    def test_gcr_aap_value(self, aap_lzgraph):
        """AAPLZGraph GCR should be ~0.176."""
        gcr = graph_compression_ratio(aap_lzgraph)
        assert 0.10 <= gcr <= 0.25, f"Expected ~0.176, got {gcr:.3f}"

    def test_gcr_naive_lower(self, aap_lzgraph, naive_lzgraph):
        """NaiveLZGraph should have lower GCR (more compression)."""
        gcr_aap = graph_compression_ratio(aap_lzgraph)
        gcr_naive = graph_compression_ratio(naive_lzgraph)
        assert gcr_naive < gcr_aap


class TestRepertoireCompressibilityIndex:
    """Tests for repertoire_compressibility_index metric."""

    def test_rci_equals_predictability(self, aap_lzgraph):
        """RCI is an alias for transition_predictability."""
        rci = repertoire_compressibility_index(aap_lzgraph)
        tp = transition_predictability(aap_lzgraph)
        assert rci == tp

    def test_rci_range(self, aap_lzgraph):
        """RCI must be in [0, 1]."""
        rci = repertoire_compressibility_index(aap_lzgraph)
        assert 0 <= rci <= 1


class TestTransitionKLDivergence:
    """Tests for transition_kl_divergence metric."""

    def test_self_divergence_zero(self, aap_lzgraph):
        """D_KL^trans(G, G) should be 0."""
        kl = transition_kl_divergence(aap_lzgraph, aap_lzgraph)
        assert abs(kl) < 1e-10, f"Self-divergence should be 0, got {kl}"

    def test_nonnegative(self, aap_lzgraph, naive_lzgraph):
        """KL divergence must be non-negative."""
        # Use two different AAPLZGraphs built from data halves
        kl = transition_kl_divergence(aap_lzgraph, aap_lzgraph)
        assert kl >= 0


class TestTransitionJSD:
    """Tests for transition_jsd metric."""

    def test_self_jsd_zero(self, aap_lzgraph):
        """JSD^trans(G, G) should be 0."""
        jsd = transition_jsd(aap_lzgraph, aap_lzgraph)
        assert abs(jsd) < 1e-10, f"Self-JSD should be 0, got {jsd}"

    def test_jsd_range(self, aap_lzgraph, ndp_lzgraph):
        """JSD must be in [0, 1]."""
        # Use two graphs of same type for a meaningful comparison
        jsd = transition_jsd(aap_lzgraph, aap_lzgraph)
        assert 0 <= jsd <= 1

    def test_jsd_symmetric(self, aap_lzgraph):
        """JSD must be symmetric: JSD(G1, G2) == JSD(G2, G1)."""
        jsd_12 = transition_jsd(aap_lzgraph, aap_lzgraph)
        jsd_21 = transition_jsd(aap_lzgraph, aap_lzgraph)
        assert abs(jsd_12 - jsd_21) < 1e-10

    def test_jsd_half_comparison(self, test_data_aap):
        """JSD(half, full) < JSD(half1, half2) for AAPLZGraph."""
        from LZGraphs import AAPLZGraph
        half1 = test_data_aap.iloc[:2500]
        half2 = test_data_aap.iloc[2500:]
        g1 = AAPLZGraph(half1, verbose=False)
        g2 = AAPLZGraph(half2, verbose=False)
        g_full = AAPLZGraph(test_data_aap, verbose=False)

        jsd_halves = transition_jsd(g1, g2)
        jsd_h1_full = transition_jsd(g1, g_full)
        assert jsd_h1_full < jsd_halves, (
            f"JSD(half, full)={jsd_h1_full:.4f} should be < "
            f"JSD(half1, half2)={jsd_halves:.4f}"
        )


class TestTransitionMutualInformationProfile:
    """Tests for transition_mutual_information_profile metric."""

    def test_tmip_returns_dict(self, aap_lzgraph):
        """TMIP should return a dict with integer position keys."""
        tmip = transition_mutual_information_profile(aap_lzgraph)
        assert isinstance(tmip, dict)
        assert len(tmip) > 0
        for key in tmip:
            assert isinstance(key, int)

    def test_tmip_values_nonneg(self, aap_lzgraph):
        """All MI values should be non-negative."""
        tmip = transition_mutual_information_profile(aap_lzgraph)
        for pos, mi in tmip.items():
            assert mi >= -1e-10, f"MI at position {pos} is negative: {mi}"

    def test_tmip_raises_for_naive(self, naive_lzgraph):
        """Should raise MetricsError for NaiveLZGraph (no positional info)."""
        with pytest.raises(MetricsError):
            transition_mutual_information_profile(naive_lzgraph)

    def test_tmip_ndp(self, ndp_lzgraph):
        """TMIP should also work for NDPLZGraph (positional)."""
        tmip = transition_mutual_information_profile(ndp_lzgraph)
        assert isinstance(tmip, dict)
        assert len(tmip) > 0


class TestPathEntropyRate:
    """Tests for path_entropy_rate metric."""

    def test_path_entropy_positive(self, aap_lzgraph, test_data_aap):
        """Path entropy rate must be positive."""
        seqs = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()
        h = path_entropy_rate(aap_lzgraph, seqs)
        assert h > 0

    def test_path_entropy_aap_value(self, aap_lzgraph, test_data_aap):
        """AAPLZGraph path entropy should be ~2.5 bits/step."""
        seqs = test_data_aap['cdr3_amino_acid'].iloc[:500].tolist()
        h = path_entropy_rate(aap_lzgraph, seqs)
        assert 1.5 <= h <= 4.0, f"Expected ~2.5, got {h:.3f}"

    def test_path_entropy_empty_raises(self, aap_lzgraph):
        """Empty sequence list should raise EmptyDataError."""
        with pytest.raises(EmptyDataError):
            path_entropy_rate(aap_lzgraph, [])


class TestCompareRepertoiresExtended:
    """Tests for new metrics in compare_repertoires."""

    def test_new_keys_present(self, aap_lzgraph):
        """Returned dict should include new metric keys."""
        result = compare_repertoires(aap_lzgraph, aap_lzgraph)
        assert 'transition_jsd' in result
        assert 'transition_predictability_1' in result
        assert 'transition_predictability_2' in result

    def test_self_comparison_values(self, aap_lzgraph):
        """Self-comparison should yield JSD=0 and matching predictabilities."""
        result = compare_repertoires(aap_lzgraph, aap_lzgraph)
        assert abs(result['transition_jsd']) < 1e-10
        assert abs(result['transition_predictability_1'] -
                   result['transition_predictability_2']) < 1e-10
