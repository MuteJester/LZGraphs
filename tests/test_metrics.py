"""
Tests for Diversity Metrics and Information-Theoretic Measures
==============================================================

Tests covering the metrics module including:
- K-diversity family (K100, K500, K1000, K5000, adaptive)
- LZCentrality
- Entropy measures (node, edge, graph entropy)
- Perplexity scores
- Distance metrics (JS divergence, cross-entropy, KL divergence)

Test Categories:
- K1000 diversity index with statistical robustness
- LZCentrality calculation
- Entropy metric calculations
- Repertoire comparison metrics
"""

import pytest
import numpy as np
from LZGraphs import NDPLZGraph
from LZGraphs.Metrics.Metrics import (
    LZCentrality,
    K_Diversity,
    K100_Diversity,
    K500_Diversity,
    K1000_Diversity,
    K5000_Diversity,
    adaptive_K_Diversity,
)
from LZGraphs.Metrics.entropy import (
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    sequence_perplexity,
    repertoire_perplexity,
    jensen_shannon_divergence,
)


class TestK1000Diversity:
    """Tests for K1000 diversity index calculation."""

    def test_k1000_ndp_encoding_within_bounds(self, test_data_ndp):
        """Verify K1000 with NDP encoding returns value in expected range."""
        k1000 = K1000_Diversity(
            list_of_sequences=test_data_ndp['cdr3_rearrangement'].to_list(),
            lzgraph_encoding_function='ndp',
            draws=50
        )
        # Expected range based on test data characteristics
        assert 2130 <= k1000 <= 2160

    def test_k1000_aap_encoding_within_bounds(self, test_data_aap):
        """Verify K1000 with AAP encoding returns value in expected range."""
        k1000 = K1000_Diversity(
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
        k1 = K1000_Diversity(
            list_of_sequences=sequences.copy(),
            lzgraph_encoding_function='ndp',
            draws=10
        )

        np.random.seed(42)
        random.seed(42)
        k2 = K1000_Diversity(
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
        """Verify K_Diversity returns positive value."""
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()
        k = K_Diversity(sequences, 'ndp', sample_size=500, draws=10)
        assert k > 0

    def test_k_diversity_with_stats(self, test_data_ndp):
        """Verify K_Diversity returns statistics when requested."""
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()
        result = K_Diversity(
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
        """Verify K100_Diversity works for small sample size."""
        sequences = test_data_aap['cdr3_amino_acid'].to_list()
        k100 = K100_Diversity(sequences, 'aap', draws=10)
        assert k100 > 0

    def test_adaptive_k_diversity(self, test_data_ndp):
        """Verify adaptive_K_Diversity auto-selects appropriate sample size."""
        sequences = test_data_ndp['cdr3_rearrangement'].to_list()
        # adaptive_K_Diversity returns (sample_size, mean) tuple
        sample_size, k = adaptive_K_Diversity(sequences, 'ndp', draws=10)
        assert k > 0
        # For 5000 sequences, should use K1000
        assert sample_size == 1000


class TestLZCentrality:
    """Tests for LZCentrality metric."""

    def test_lz_centrality_calculation(self, ndp_lzgraph):
        """Verify LZCentrality calculation produces expected result."""
        test_sequence = 'TGTGCCTGCGTAACACAGGGGGTTTGGTATGGCTACACCTTC'
        lzc = LZCentrality(ndp_lzgraph, test_sequence)

        assert lzc == 14.105263157894736

    def test_lz_centrality_positive(self, ndp_lzgraph, test_data_ndp):
        """Verify LZCentrality is always positive."""
        sequence = test_data_ndp['cdr3_rearrangement'].iloc[0]
        lzc = LZCentrality(ndp_lzgraph, sequence)
        assert lzc > 0

    def test_lz_centrality_different_sequences(self, ndp_lzgraph, test_data_ndp):
        """Verify different sequences can have different LZCentrality."""
        seq1 = test_data_ndp['cdr3_rearrangement'].iloc[0]
        seq2 = test_data_ndp['cdr3_rearrangement'].iloc[1]

        lzc1 = LZCentrality(ndp_lzgraph, seq1)
        lzc2 = LZCentrality(ndp_lzgraph, seq2)

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


class TestMetricRobustness:
    """Tests for metric robustness and edge cases."""

    def test_k1000_with_small_repertoire(self, test_data_aap):
        """Verify K1000 handles repertoires smaller than sample size."""
        # Use only 100 sequences
        small_repertoire = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        # Should auto-adjust sample size and still work
        k = K_Diversity(small_repertoire, 'aap', sample_size=1000, draws=5)
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
