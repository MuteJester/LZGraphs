"""
Tests for LZBOW (Bag of Words Encoder)
======================================

Tests covering the LZBOW class for encoding repertoires as bag-of-words vectors:
- Dictionary building (fit)
- Sequence transformation (transform)
- BOW object combination (__add__)
- Dictionary loading (load_from)
"""

import pytest
import numpy as np

from LZGraphs.BagOfWords.BOWEncoder import LZBOW
from LZGraphs.Utilities.decomposition import lempel_ziv_decomposition
from LZGraphs import EncodingFunctionMismatchError


class TestLZBOWFit:
    """Tests for LZBOW fitting (dictionary building)."""

    def test_fit_single_sequence(self):
        """Verify fit works with a single sequence."""
        bow = LZBOW()
        bow.fit("CASSLGQAYEQYF")

        assert bow.dictionary_size > 0
        assert len(bow.dictionary) > 0
        assert len(bow.dictionary_index_map) == bow.dictionary_size

    def test_fit_list_of_sequences(self, test_data_aap):
        """Verify fit works with a list of sequences."""
        bow = LZBOW()
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()
        bow.fit(sequences)

        assert bow.dictionary_size > 0
        assert bow.observed_sequences == 100

    def test_fit_increases_dictionary(self):
        """Verify fitting on more data grows the dictionary."""
        bow = LZBOW()
        bow.fit("CASSLGQAYEQYF")
        size_after_one = bow.dictionary_size

        bow.fit("CASSLDRGTEAFF")
        size_after_two = bow.dictionary_size

        # Should have grown (or stayed same if patterns overlap)
        assert size_after_two >= size_after_one

    def test_fit_index_maps_consistent(self):
        """Verify forward and inverse index maps are consistent."""
        bow = LZBOW()
        bow.fit(["CASSLGQAYEQYF", "CASSLDRGTEAFF", "CASSPDRGSYEQYF"])

        # Every pattern in dictionary should be in index map
        for pattern in bow.dictionary:
            assert pattern in bow.dictionary_index_map

        # Inverse map should be symmetric
        for idx, pattern in bow.dictionary_index_inverse_map.items():
            assert bow.dictionary_index_map[pattern] == idx


class TestLZBOWTransform:
    """Tests for LZBOW transformation."""

    @pytest.fixture
    def fitted_bow(self, test_data_aap):
        """Create a fitted BOW model."""
        bow = LZBOW()
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:200].tolist()
        bow.fit(sequences)
        return bow

    def test_transform_single_sequence(self, fitted_bow):
        """Verify transform returns correct shape for single sequence."""
        result = fitted_bow.transform("CASSLGQAYEQYF")

        assert isinstance(result, np.ndarray)
        assert result.shape == (fitted_bow.dictionary_size,)

    def test_transform_list_of_sequences(self, fitted_bow, test_data_aap):
        """Verify transform works with list of sequences."""
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:50].tolist()
        result = fitted_bow.transform(sequences)

        assert isinstance(result, np.ndarray)
        assert result.shape == (fitted_bow.dictionary_size,)
        assert result.sum() > 0  # Should have counted some patterns

    def test_transform_normalized(self, fitted_bow, test_data_aap):
        """Verify normalized transform sums to 1."""
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:50].tolist()
        result = fitted_bow.transform(sequences, normalize=True)

        assert np.isclose(result.sum(), 1.0, rtol=1e-10)

    def test_transform_counts_patterns(self, fitted_bow):
        """Verify transform correctly counts pattern occurrences."""
        # Transform a sequence we know
        seq = "CASSLGQAYEQYF"
        result = fitted_bow.transform(seq)

        # Get expected patterns from decomposition
        patterns = lempel_ziv_decomposition(seq)
        patterns_in_dict = [p for p in patterns if p in fitted_bow.dictionary]

        # Sum of counts should equal number of patterns found
        assert result.sum() == len(patterns_in_dict)

    def test_transform_unseen_patterns_ignored(self):
        """Verify unseen patterns don't cause errors."""
        # Fit on minimal data
        bow = LZBOW()
        bow.fit("AAA")

        # Transform sequence with patterns not in dictionary
        result = bow.transform("CASSLGQAYEQYF")

        # Should return array (possibly with zeros for unseen patterns)
        assert isinstance(result, np.ndarray)
        assert result.shape == (bow.dictionary_size,)


class TestLZBOWCombination:
    """Tests for combining BOW objects."""

    def test_add_bow_objects(self):
        """Verify __add__ combines dictionaries."""
        bow1 = LZBOW()
        bow1.fit("CASSLGQAYEQYF")

        bow2 = LZBOW()
        bow2.fit("CASSLDRGTEAFF")

        combined = bow1 + bow2

        # Combined should have patterns from both
        assert combined.dictionary >= bow1.dictionary
        assert combined.dictionary >= bow2.dictionary

    def test_add_preserves_observed_count(self):
        """Verify __add__ sums observed sequences."""
        bow1 = LZBOW()
        bow1.fit(["CASSLGQAYEQYF", "CASSAB"])

        bow2 = LZBOW()
        bow2.fit(["CASSLDRGTEAFF"])

        combined = bow1 + bow2

        expected_count = bow1.observed_sequences + bow2.observed_sequences
        assert combined.observed_sequences == expected_count

    def test_add_different_encodings_raises(self):
        """Verify adding BOWs with different encodings raises error."""
        bow1 = LZBOW(encoding_function=lempel_ziv_decomposition)
        bow1.fit("CASSLGQAYEQYF")

        bow2 = LZBOW(encoding_function=lambda x: list(x))  # Different function
        bow2.fit("CASSLGQAYEQYF")

        with pytest.raises(EncodingFunctionMismatchError):
            _ = bow1 + bow2


class TestLZBOWLoadFrom:
    """Tests for loading BOW from another instance."""

    def test_load_from_copies_dictionary(self):
        """Verify load_from copies dictionary."""
        source = LZBOW()
        source.fit(["CASSLGQAYEQYF", "CASSLDRGTEAFF"])

        target = LZBOW()
        target.load_from(source)

        assert target.dictionary == source.dictionary
        assert target.dictionary_size == source.dictionary_size
        assert target.dictionary_index_map == source.dictionary_index_map

    def test_load_from_enables_transform(self):
        """Verify loaded BOW can transform sequences."""
        source = LZBOW()
        source.fit(["CASSLGQAYEQYF", "CASSLDRGTEAFF"])

        target = LZBOW()
        target.load_from(source)

        # Should be able to transform
        result = target.transform("CASSLGQAYEQYF")
        assert isinstance(result, np.ndarray)


class TestLZBOWEncodingFunctions:
    """Tests for different encoding functions."""

    def test_default_encoding(self):
        """Verify default encoding uses LZ decomposition."""
        bow = LZBOW()
        bow.fit("CASSLGQAYEQYF")

        # Default should use lempel_ziv_decomposition
        expected_patterns = set(lempel_ziv_decomposition("CASSLGQAYEQYF"))
        assert bow.dictionary == expected_patterns

    def test_custom_encoding_function(self):
        """Verify custom encoding function is used."""
        # Simple k-mer encoding
        def kmer_encode(seq, k=3):
            return [seq[i:i+k] for i in range(len(seq) - k + 1)]

        bow = LZBOW(encoding_function=kmer_encode)
        bow.fit("CASSLGQAYEQYF")

        # Should contain 3-mers
        assert "CAS" in bow.dictionary
        assert "ASS" in bow.dictionary
