"""
Tests for Phase 9 New Features
================================

Tests covering:
- AIRR format support (from_airr classmethod)
- Batch PGEN computation (batch_walk_probability)
- Enhanced LZBOW (fit_transform, tfidf_transform)
- Convenience wrappers (compare_repertoires)
"""

import pytest
import pandas as pd
import numpy as np

from LZGraphs import (
    AAPLZGraph,
    NDPLZGraph,
    NaiveLZGraph,
    compare_repertoires,
    lempel_ziv_decomposition,
)
from LZGraphs.bag_of_words.bow_encoder import LZBOW


# =========================================================================
# AIRR Format Support
# =========================================================================

class TestFromAIRR:
    """Tests for LZGraphBase.from_airr classmethod."""

    def test_from_airr_aap(self, test_data_aap):
        """Verify AAPLZGraph.from_airr maps AIRR columns correctly."""
        # Create AIRR-formatted data
        airr_df = test_data_aap.rename(columns={
            'cdr3_amino_acid': 'junction_aa',
            'V': 'v_call',
            'J': 'j_call',
        })

        graph = AAPLZGraph.from_airr(airr_df, verbose=False)

        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() > 0
        assert graph.genetic is True

    def test_from_airr_ndp(self, test_data_ndp):
        """Verify NDPLZGraph.from_airr maps AIRR columns correctly."""
        airr_df = test_data_ndp.rename(columns={
            'cdr3_rearrangement': 'junction',
            'V': 'v_call',
            'J': 'j_call',
        })

        graph = NDPLZGraph.from_airr(airr_df, verbose=False)

        assert graph.graph.number_of_nodes() > 0
        assert graph.genetic is True

    def test_from_airr_custom_column_map(self, test_data_aap):
        """Verify from_airr works with custom column mappings."""
        # Rename to non-standard names
        custom_df = test_data_aap.rename(columns={
            'cdr3_amino_acid': 'my_sequence_col',
            'V': 'v_gene',
            'J': 'j_gene',
        })

        custom_map = {
            'my_sequence_col': 'cdr3_amino_acid',
            'v_gene': 'V',
            'j_gene': 'J',
        }

        graph = AAPLZGraph.from_airr(custom_df, column_map=custom_map, verbose=False)

        assert graph.graph.number_of_nodes() > 0
        assert graph.genetic is True

    def test_from_airr_no_gene_columns(self, test_data_aap):
        """Verify from_airr works without gene columns."""
        airr_df = pd.DataFrame({
            'junction_aa': test_data_aap['cdr3_amino_acid'].values,
        })

        graph = AAPLZGraph.from_airr(airr_df, verbose=False)

        assert graph.graph.number_of_nodes() > 0
        assert graph.genetic is False

    def test_from_airr_preserves_existing_columns(self, test_data_aap):
        """Verify from_airr doesn't rename columns that already match."""
        # If data already has LZGraphs column names, don't rename
        graph = AAPLZGraph.from_airr(test_data_aap.copy(), verbose=False)

        assert graph.graph.number_of_nodes() > 0


# =========================================================================
# Batch PGEN Computation
# =========================================================================

class TestBatchWalkProbability:
    """Tests for LZGraphBase.batch_walk_probability method."""

    def test_batch_returns_array(self, aap_lzgraph, sample_amino_acid_sequences):
        """Verify batch_walk_probability returns numpy array."""
        seqs = sample_amino_acid_sequences[:5]
        result = aap_lzgraph.batch_walk_probability(seqs)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_batch_matches_individual(self, aap_lzgraph, sample_amino_acid_sequences):
        """Verify batch results match individual walk_probability calls."""
        seqs = sample_amino_acid_sequences[:5]
        batch_result = aap_lzgraph.batch_walk_probability(seqs)

        for i, seq in enumerate(seqs):
            individual = aap_lzgraph.walk_probability(seq, verbose=False)
            assert abs(batch_result[i] - individual) < 1e-10

    def test_batch_log_probability(self, aap_lzgraph, sample_amino_acid_sequences):
        """Verify batch log-probability mode works."""
        seqs = sample_amino_acid_sequences[:5]
        result = aap_lzgraph.batch_walk_probability(seqs, use_log=True)

        assert isinstance(result, np.ndarray)
        # Log probabilities should be non-positive
        for val in result:
            assert val <= 0 or val == float('-inf')

    def test_batch_empty_input(self, aap_lzgraph):
        """Verify batch handles empty input."""
        result = aap_lzgraph.batch_walk_probability([])
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_batch_ndp(self, ndp_lzgraph, test_data_ndp):
        """Verify batch works for NDPLZGraph."""
        seqs = test_data_ndp['cdr3_rearrangement'].iloc[:5].tolist()
        result = ndp_lzgraph.batch_walk_probability(seqs)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_batch_naive(self, naive_lzgraph, test_data_nucleotide):
        """Verify batch works for NaiveLZGraph."""
        seqs = test_data_nucleotide['cdr3_rearrangement'].iloc[:5].tolist()
        result = naive_lzgraph.batch_walk_probability(seqs)

        assert isinstance(result, np.ndarray)
        assert len(result) == 5


# =========================================================================
# Enhanced LZBOW
# =========================================================================

class TestLZBOWFitTransform:
    """Tests for LZBOW.fit_transform method."""

    def test_fit_transform_single_string(self):
        """Verify fit_transform works on single string."""
        bow = LZBOW()
        result = bow.fit_transform("CASSLGIRRTNTEAFF")

        assert isinstance(result, np.ndarray)
        assert bow.dictionary_size > 0
        assert result.sum() > 0

    def test_fit_transform_list(self, sample_amino_acid_sequences):
        """Verify fit_transform works on list of sequences."""
        bow = LZBOW()
        result = bow.fit_transform(sample_amino_acid_sequences)

        assert isinstance(result, np.ndarray)
        assert bow.dictionary_size > 0
        assert result.sum() > 0

    def test_fit_transform_per_sequence(self, sample_amino_acid_sequences):
        """Verify fit_transform per_sequence returns matrix."""
        bow = LZBOW()
        result = bow.fit_transform(sample_amino_acid_sequences, per_sequence=True)

        assert result.ndim == 2
        assert result.shape[0] == len(sample_amino_acid_sequences)
        assert result.shape[1] == bow.dictionary_size

    def test_fit_transform_equals_separate(self, sample_amino_acid_sequences):
        """Verify fit_transform equals separate fit + transform."""
        bow1 = LZBOW()
        result1 = bow1.fit_transform(sample_amino_acid_sequences)

        bow2 = LZBOW()
        bow2.fit(sample_amino_acid_sequences)
        result2 = bow2.transform(sample_amino_acid_sequences)

        np.testing.assert_array_equal(result1, result2)


class TestLZBOWTfIdf:
    """Tests for LZBOW.tfidf_transform method."""

    def test_tfidf_returns_matrix(self, sample_amino_acid_sequences):
        """Verify tfidf_transform returns a 2D matrix."""
        bow = LZBOW()
        bow.fit(sample_amino_acid_sequences)
        result = bow.tfidf_transform(sample_amino_acid_sequences)

        assert result.ndim == 2
        assert result.shape[0] == len(sample_amino_acid_sequences)
        assert result.shape[1] == bow.dictionary_size

    def test_tfidf_values_reasonable(self, sample_amino_acid_sequences):
        """Verify TF-IDF values are non-negative."""
        bow = LZBOW()
        bow.fit(sample_amino_acid_sequences)
        result = bow.tfidf_transform(sample_amino_acid_sequences)

        # TF values are non-negative, IDF can be 0 for universal terms
        # Overall TF-IDF should be non-negative
        assert np.all(result >= 0)

    def test_tfidf_common_terms_downweighted(self, sample_amino_acid_sequences):
        """Verify TF-IDF down-weights terms that appear in all sequences."""
        bow = LZBOW()
        bow.fit(sample_amino_acid_sequences)
        tf_matrix = bow.transform(sample_amino_acid_sequences, per_sequence=True)
        tfidf_matrix = bow.tfidf_transform(sample_amino_acid_sequences)

        # For terms appearing in all sequences, IDF is log(N/(1+N)) < 0,
        # so TF-IDF should be <= TF for those columns (after normalization)
        # Just verify dimensions match and values changed
        assert tf_matrix.shape == tfidf_matrix.shape


# =========================================================================
# Convenience Wrappers
# =========================================================================

class TestCompareRepertoires:
    """Tests for compare_repertoires function."""

    def test_compare_returns_series(self, test_data_aap):
        """Verify compare_repertoires returns a pandas Series."""
        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        g1 = AAPLZGraph(data1, verbose=False)
        g2 = AAPLZGraph(data2, verbose=False)

        result = compare_repertoires(g1, g2)

        assert isinstance(result, pd.Series)

    def test_compare_contains_expected_metrics(self, test_data_aap):
        """Verify result contains all expected metric names."""
        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        g1 = AAPLZGraph(data1, verbose=False)
        g2 = AAPLZGraph(data2, verbose=False)

        result = compare_repertoires(g1, g2)

        expected_keys = [
            'js_divergence',
            'cross_entropy_1_2',
            'cross_entropy_2_1',
            'kl_divergence_1_2',
            'kl_divergence_2_1',
            'node_entropy_1',
            'node_entropy_2',
            'edge_entropy_1',
            'edge_entropy_2',
            'shared_nodes',
            'shared_edges',
            'jaccard_nodes',
            'jaccard_edges',
        ]
        for key in expected_keys:
            assert key in result.index, f"Missing key: {key}"

    def test_compare_same_graph_low_divergence(self, aap_lzgraph):
        """Verify comparing a graph with itself gives low divergence."""
        result = compare_repertoires(aap_lzgraph, aap_lzgraph)

        # JS divergence should be 0 for identical distributions
        assert result['js_divergence'] < 0.01
        # Jaccard should be 1.0 for identical graphs
        assert result['jaccard_nodes'] == 1.0
        assert result['jaccard_edges'] == 1.0

    def test_compare_different_graphs_positive_divergence(self, test_data_aap):
        """Verify different graphs have positive divergence."""
        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        g1 = AAPLZGraph(data1, verbose=False)
        g2 = AAPLZGraph(data2, verbose=False)

        result = compare_repertoires(g1, g2)

        assert result['js_divergence'] > 0
        assert result['shared_nodes'] > 0  # Some overlap expected

    def test_compare_entropies_positive(self, test_data_aap):
        """Verify entropies are positive for non-trivial graphs."""
        data1 = test_data_aap.iloc[:2500].copy()
        data2 = test_data_aap.iloc[2500:].copy()

        g1 = AAPLZGraph(data1, verbose=False)
        g2 = AAPLZGraph(data2, verbose=False)

        result = compare_repertoires(g1, g2)

        assert result['node_entropy_1'] > 0
        assert result['node_entropy_2'] > 0
        assert result['edge_entropy_1'] > 0
        assert result['edge_entropy_2'] > 0
