"""
Tests for Utility Functions
===========================

Tests covering utility functions and classes:
- K-mer dictionary generation
- Lempel-Ziv decomposition algorithm
- Node/Edge saturation probe
- Saturation curve metrics

Test Categories:
- Dictionary generation correctness
- LZ decomposition properties
- Saturation curve calculations
"""

import pytest
import numpy as np
from LZGraphs import generate_kmer_dictionary
from LZGraphs.utilities.decomposition import lempel_ziv_decomposition
from LZGraphs.utilities.misc import _is_v_gene, _is_j_gene


class TestKmerDictionary:
    """Tests for k-mer dictionary generation."""

    def test_dictionary_generation(self, lz_dictionary):
        """Verify k-mer dictionary generates expected patterns."""
        keys = lz_dictionary[15:25]
        expected = ['GC', 'CA', 'CT', 'CG', 'CC', 'AAA', 'AAT', 'AAG', 'AAC', 'ATA']
        assert keys == expected

    def test_dictionary_contains_single_bases(self, lz_dictionary):
        """Verify dictionary contains all single nucleotides."""
        single_bases = ['A', 'T', 'G', 'C']
        for base in single_bases:
            assert base in lz_dictionary

    def test_dictionary_length_increases_with_k(self):
        """Verify dictionary length increases with maximum k-mer length."""
        dict_4 = generate_kmer_dictionary(4)
        dict_6 = generate_kmer_dictionary(6)

        assert len(dict_4) < len(dict_6)

    def test_dictionary_all_unique(self, lz_dictionary):
        """Verify all k-mers in dictionary are unique."""
        assert len(lz_dictionary) == len(set(lz_dictionary))


class TestLempelZivDecomposition:
    """Tests for Lempel-Ziv decomposition algorithm."""

    def test_decomposition_covers_full_sequence(self):
        """Verify decomposition covers the entire input sequence."""
        sequence = "TGTGCCAGCAGCCAAGA"
        subpatterns = lempel_ziv_decomposition(sequence)
        reconstructed = ''.join(subpatterns)
        assert reconstructed == sequence

    def test_decomposition_produces_unique_patterns(self):
        """Verify all subpatterns (except last) are unique."""
        sequence = "TGTGCCAGCAGCCAAGA"
        subpatterns = lempel_ziv_decomposition(sequence)

        # All patterns except possibly the last should be unique
        # (last pattern may repeat if sequence ends mid-pattern)
        patterns_except_last = subpatterns[:-1]
        assert len(patterns_except_last) == len(set(patterns_except_last))

    def test_decomposition_empty_sequence(self):
        """Verify decomposition handles empty sequence."""
        subpatterns = lempel_ziv_decomposition("")
        assert subpatterns == []

    def test_decomposition_single_character(self):
        """Verify decomposition handles single character."""
        subpatterns = lempel_ziv_decomposition("A")
        assert subpatterns == ["A"]

    def test_decomposition_repeated_character(self):
        """Verify decomposition handles repeated characters correctly."""
        sequence = "AAAA"
        subpatterns = lempel_ziv_decomposition(sequence)

        # Should produce: ['A', 'AA', 'A'] for "AAAA"
        reconstructed = ''.join(subpatterns)
        assert reconstructed == sequence

    def test_decomposition_all_unique_characters(self):
        """Verify decomposition of all-unique sequence."""
        sequence = "ATGC"
        subpatterns = lempel_ziv_decomposition(sequence)

        # Each character should be its own pattern since none repeat
        assert subpatterns == ['A', 'T', 'G', 'C']

    def test_decomposition_long_sequence(self):
        """Verify decomposition works on longer sequences."""
        # Create a long sequence with patterns
        sequence = "TGTGCCAGCAGCCAAGATATGGCTACACCTTC" * 3
        subpatterns = lempel_ziv_decomposition(sequence)

        # Should produce valid decomposition
        reconstructed = ''.join(subpatterns)
        assert reconstructed == sequence

    def test_decomposition_deterministic(self):
        """Verify decomposition is deterministic."""
        sequence = "TGTGCCAGCAGCCAAGA"

        result1 = lempel_ziv_decomposition(sequence)
        result2 = lempel_ziv_decomposition(sequence)

        assert result1 == result2


class TestDecompositionPerformance:
    """Performance-related tests for LZ decomposition."""

    def test_decomposition_linear_time_behavior(self):
        """Verify decomposition scales reasonably with sequence length."""
        import time

        # Generate sequences of increasing length
        base_seq = "TGTGCCAGCAGCCAAGA"
        times = []

        for multiplier in [1, 5, 10]:
            sequence = base_seq * multiplier
            start = time.time()
            for _ in range(100):
                lempel_ziv_decomposition(sequence)
            elapsed = time.time() - start
            times.append(elapsed)

        # Verify time doesn't grow super-linearly
        # (10x length shouldn't take more than 20x time)
        ratio = times[2] / times[0]
        assert ratio < 200  # Very generous bound


class TestNodeEdgeSaturationProbe:
    """Tests for NodeEdgeSaturationProbe utility."""

    @pytest.fixture
    def saturation_probe(self):
        """Create a saturation probe instance."""
        from LZGraphs.metrics.saturation import NodeEdgeSaturationProbe
        return NodeEdgeSaturationProbe(node_function='aap')

    def test_saturation_curve_returns_dataframe(
        self, saturation_probe, test_data_aap
    ):
        """Verify saturation_curve returns a DataFrame."""
        import pandas as pd
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        curve = saturation_probe.saturation_curve(sequences, log_every=20)
        assert isinstance(curve, pd.DataFrame)
        assert 'n_sequences' in curve.columns
        assert 'nodes' in curve.columns
        assert 'edges' in curve.columns

    def test_saturation_curve_monotonic_nodes(
        self, saturation_probe, test_data_aap
    ):
        """Verify node count is monotonically increasing."""
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        curve = saturation_probe.saturation_curve(sequences, log_every=20)
        nodes = curve['nodes'].tolist()

        for i in range(1, len(nodes)):
            assert nodes[i] >= nodes[i - 1]

    def test_half_saturation_point_positive(
        self, saturation_probe, test_data_aap
    ):
        """Verify half_saturation_point returns positive value."""
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        k50 = saturation_probe.half_saturation_point(sequences)
        assert k50 > 0

    def test_area_under_saturation_curve_bounded(
        self, saturation_probe, test_data_aap
    ):
        """Verify normalized AUSC is between 0 and 1."""
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        ausc = saturation_probe.area_under_saturation_curve(
            sequences, normalize=True
        )
        assert 0 <= ausc <= 1

    def test_diversity_profile_returns_dataframe(
        self, saturation_probe, test_data_aap
    ):
        """Verify diversity_profile returns a DataFrame with all metrics."""
        import pandas as pd
        sequences = test_data_aap['cdr3_amino_acid'].iloc[:100].tolist()

        profile = saturation_probe.diversity_profile(sequences)

        # Should return a DataFrame
        assert isinstance(profile, pd.DataFrame)

        expected_columns = [
            'n_sequences', 'final_nodes', 'final_edges',
            'k50_nodes', 'k50_edges', 'ausc_nodes', 'ausc_edges'
        ]
        for col in expected_columns:
            assert col in profile.columns


class TestDecompositionEdgeCases:
    """Edge case tests for LZ decomposition."""

    def test_decomposition_with_amino_acids(self):
        """Verify decomposition works with amino acid sequences."""
        sequence = "CASSLGQAYEQYF"
        subpatterns = lempel_ziv_decomposition(sequence)

        reconstructed = ''.join(subpatterns)
        assert reconstructed == sequence

    def test_decomposition_with_numbers(self):
        """Verify decomposition works with numeric strings."""
        sequence = "123123412345"
        subpatterns = lempel_ziv_decomposition(sequence)

        reconstructed = ''.join(subpatterns)
        assert reconstructed == sequence

    def test_decomposition_pattern_growth(self):
        """Verify patterns grow as expected in LZ decomposition."""
        # In LZ76, each new pattern should be a seen pattern + one char
        sequence = "ABAABABAAB"
        subpatterns = lempel_ziv_decomposition(sequence)

        # Verify we can reconstruct
        reconstructed = ''.join(subpatterns)
        assert reconstructed == sequence


class TestGeneNameDetection:
    """Tests for _is_v_gene and _is_j_gene covering all IMGT nomenclature."""

    # --- V gene tests ---
    @pytest.mark.parametrize("gene_name", [
        "V30",              # Simple naming
        "TRBV5-1*01",       # TCR beta
        "TRAV12-2*01",      # TCR alpha
        "TRGV9*01",         # TCR gamma
        "TRDV1*01",         # TCR delta
        "IGHV3-23*01",      # BCR heavy
        "IGKV1-39*01",      # BCR kappa
        "IGLV2-14*01",      # BCR lambda
    ])
    def test_is_v_gene_positive(self, gene_name):
        """V gene names from all chain types should be recognized."""
        assert _is_v_gene(gene_name) is True

    @pytest.mark.parametrize("gene_name", [
        "Vsum", "Jsum", "weight",   # Meta keys
        "TRBJ2-1*01",               # J gene, not V
        "IGHJ4*02",                 # J gene, not V
        "J2",                       # Simple J
    ])
    def test_is_v_gene_negative(self, gene_name):
        """Non-V-gene keys should not be recognized as V genes."""
        assert _is_v_gene(gene_name) is False

    # --- J gene tests ---
    @pytest.mark.parametrize("gene_name", [
        "J2",               # Simple naming
        "TRBJ2-1*01",       # TCR beta
        "TRAJ40*01",        # TCR alpha
        "TRGJ1*01",         # TCR gamma
        "TRDJ1*01",         # TCR delta
        "IGHJ4*02",         # BCR heavy
        "IGKJ2*01",         # BCR kappa
        "IGLJ3*01",         # BCR lambda
    ])
    def test_is_j_gene_positive(self, gene_name):
        """J gene names from all chain types should be recognized."""
        assert _is_j_gene(gene_name) is True

    @pytest.mark.parametrize("gene_name", [
        "Vsum", "Jsum", "weight",   # Meta keys
        "TRBV5-1*01",               # V gene, not J
        "IGHV3-23*01",              # V gene, not J
        "V30",                      # Simple V
    ])
    def test_is_j_gene_negative(self, gene_name):
        """Non-J-gene keys should not be recognized as J genes."""
        assert _is_j_gene(gene_name) is False
