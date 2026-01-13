"""
Tests for AAPLZGraph Class
==========================

Tests covering the AAPLZGraph (Amino Acid Positional) implementation
which uses positional LZ decomposition for amino acid sequences.

Test Categories:
- Graph construction and structure
- Node and edge properties
- Positional encoding consistency
- Probability calculations (including log-probability mode)
- Gene-aware random walks
- Terminal states and length distributions
- Input validation
"""

import pytest
import numpy as np
from LZGraphs import AAPLZGraph, NDPLZGraph, InvalidSequenceError, MissingColumnError


class TestAAPLZGraphConstruction:
    """Tests for AAPLZGraph construction and basic properties."""

    def test_graph_has_expected_edge_count(self, aap_lzgraph):
        """Verify the graph has the expected number of edges."""
        assert len(aap_lzgraph.edges) == 9528

    def test_initial_states_populated(self, aap_lzgraph):
        """Verify initial states are correctly counted with position."""
        # C at position 1 is the canonical start for CDR3 sequences
        assert aap_lzgraph.initial_states['C_1'] == 4996

    def test_node_frequency_tracking(self, aap_lzgraph):
        """Verify per-node frequency is correctly tracked."""
        assert aap_lzgraph.per_node_observed_frequency['SG_5'] == 89


class TestAAPLZGraphProbabilities:
    """Tests for probability calculations in AAPLZGraph."""

    def test_subpattern_probability_common_position(self, aap_lzgraph):
        """Verify probability for a common positional pattern."""
        expected = 0.0031496933193346966
        actual = aap_lzgraph.subpattern_individual_probability.loc['Y_9', 'proba']
        assert actual == expected

    def test_subpattern_probability_rare_position(self, aap_lzgraph):
        """Verify probability for a rare positional pattern."""
        expected = 0.00014735407341916708
        actual = aap_lzgraph.subpattern_individual_probability.loc['Y_5', 'proba']
        assert actual == expected

    def test_walk_probability_calculation(self, aap_lzgraph, test_data_aap):
        """Verify walk probability calculation produces expected results."""
        lzpgens = []
        for sequence in test_data_aap['cdr3_amino_acid'].iloc[:15]:
            # Note: Using NDPLZGraph.encode_sequence as per original test
            walk = NDPLZGraph.encode_sequence(sequence)
            lzpgen = aap_lzgraph.walk_probability(walk, verbose=False)
            lzpgens.append(lzpgen)

        # Verify first sequence's log-probability
        assert np.round(np.log(lzpgens[0]), 2) == -72.09


class TestAAPLZGraphTerminalStates:
    """Tests for terminal state mapping in AAPLZGraph."""

    def test_terminal_state_map_late_position(self, aap_lzgraph):
        """Verify terminal state mapping for late sequence position."""
        expected_terminals = ['F_19', 'F_21', 'F_20']
        actual = aap_lzgraph.terminal_state_map['F_19']
        assert set(actual) == set(expected_terminals)

    def test_terminal_state_map_mid_position(self, aap_lzgraph):
        """Verify terminal state mapping for mid sequence position."""
        expected_terminals = ['F_19', 'F_21', 'F_18', 'F_17', 'F_20']
        actual = aap_lzgraph.terminal_state_map['F_17']
        assert set(actual) == set(expected_terminals)

    def test_terminal_state_map_complex(self, aap_lzgraph):
        """Verify terminal state mapping with many terminal options."""
        expected_terminals = [
            'YF_18', 'TF_12', 'YF_17', 'F_16', 'FF_16', 'TF_17', 'TF_18',
            'FF_15', 'TF_20', 'HF_19', 'FF_17', 'YF_16', 'F_17', 'F_22',
            'TF_15', 'F_18', 'FF_18', 'HF_17', 'TF_21', 'F_19', 'F_21',
            'YF_20', 'HF_20', 'YF_19', 'FF_19', 'TF_19', 'F_15', 'F_14', 'F_20'
        ]
        actual = aap_lzgraph.terminal_state_map['TF_12']
        assert set(actual) == set(expected_terminals)

    def test_terminal_states_count(self, aap_lzgraph):
        """Verify terminal state counts."""
        assert aap_lzgraph.terminal_states['F_17'] == 308


class TestAAPLZGraphRandomWalks:
    """Tests for random walk generation in AAPLZGraph."""

    def test_unsupervised_random_walk_returns_result(self, aap_lzgraph):
        """Verify unsupervised random walk generates a sequence."""
        result = aap_lzgraph.unsupervised_random_walk()
        assert result is not None

    def test_genomic_random_walk_returns_result(self, aap_lzgraph):
        """Verify genomic random walk generates a sequence."""
        result = aap_lzgraph.genomic_random_walk()
        assert result is not None

    def test_random_walk_with_initial_state(self, aap_lzgraph):
        """Verify random walk starts with specified initial state."""
        # RandomWalkMixin.random_walk returns just the walk list
        walk = aap_lzgraph.random_walk(initial_state='C_1')
        assert walk[0] == 'C_1'


class TestAAPLZGraphFeatures:
    """Tests for feature extraction from AAPLZGraph."""

    def test_eigenvector_centrality(self, aap_lzgraph):
        """Verify eigenvector centrality calculation."""
        feature_vector = aap_lzgraph.eigenvector_centrality()
        assert np.round(np.log(feature_vector['Q_15']), 5) == -16.29151


class TestAAPLZGraphGeneData:
    """Tests for V/J gene data handling in AAPLZGraph."""

    def test_marginal_vgene_probability(self, aap_lzgraph):
        """Verify marginal V gene probability."""
        assert aap_lzgraph.marginal_vgenes['TRBV19-1*01'] == 0.0774

    def test_length_distribution_common(self, aap_lzgraph):
        """Verify length distribution for common length."""
        assert aap_lzgraph.lengths[14] == 1161

    def test_length_distribution_rare(self, aap_lzgraph):
        """Verify length distribution for rare length."""
        assert aap_lzgraph.lengths[21] == 5


class TestAAPLZGraphEncoding:
    """Tests for sequence encoding in AAPLZGraph."""

    def test_encode_sequence_includes_position(self):
        """Verify encoded sequence includes positional information."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        # All nodes should contain underscore and position number
        for node in walk:
            assert '_' in node, f"Node {node} missing position separator"
            parts = node.rsplit('_', 1)
            assert parts[1].isdigit(), f"Node {node} missing valid position"

    def test_encode_sequence_starts_at_position_one(self):
        """Verify encoding starts at position 1."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        # First node should be at position 1
        first_position = int(walk[0].rsplit('_', 1)[1])
        assert first_position == 1

    def test_encoding_consistency_with_walk_probability(self, aap_lzgraph):
        """
        Verify that encode_sequence output matches walk_probability expectations.

        This tests the critical bug fix where the underscore was missing
        in the walk_probability encoding format.
        """
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        # Should be able to compute probability without KeyError
        prob = aap_lzgraph.walk_probability(walk, verbose=False)
        assert prob > 0


class TestAAPLZGraphLogProbability:
    """Tests for log-probability mode in AAPLZGraph."""

    def test_log_probability_returns_negative_value(self, aap_lzgraph):
        """Verify log-probability returns a negative value."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        log_prob = aap_lzgraph.walk_probability(walk, verbose=False, use_log=True)
        assert log_prob < 0

    def test_log_probability_consistency(self, aap_lzgraph):
        """Verify log-probability is consistent with regular probability."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        regular_prob = aap_lzgraph.walk_probability(walk, verbose=False)
        log_prob = aap_lzgraph.walk_probability(walk, verbose=False, use_log=True)

        # log(prob) should equal log_prob (within floating point tolerance)
        assert np.isclose(np.log(regular_prob), log_prob, rtol=1e-5)

    def test_walk_log_probability_convenience_method(self, aap_lzgraph):
        """Verify walk_log_probability convenience method works."""
        sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(sequence)

        log_prob = aap_lzgraph.walk_log_probability(walk, verbose=False)
        assert log_prob < 0


class TestAAPLZGraphValidation:
    """Tests for input validation in AAPLZGraph."""

    def test_invalid_amino_acid_raises_error(self):
        """Verify that invalid amino acids raise ValueError."""
        import pandas as pd

        # Create small dataset with invalid amino acid 'X'
        # Using small dataset ensures the invalid sequence is in the validation sample
        bad_data = pd.DataFrame({
            'cdr3_amino_acid': [
                'CASSLGQAYEQYF',  # valid
                'CASSXLGQAYEQYF', # invalid - contains X
                'CASSLKDYEQYF',   # valid
            ],
            'V': ['TRBV5-1*01'] * 3,
            'J': ['TRBJ2-7*01'] * 3
        })

        with pytest.raises(InvalidSequenceError, match="invalid amino acid"):
            AAPLZGraph(bad_data)

    def test_missing_column_raises_error(self, test_data_aap):
        """Verify that missing required column raises MissingColumnError."""
        # Remove required column
        bad_data = test_data_aap.drop(columns=['cdr3_amino_acid'])

        with pytest.raises(MissingColumnError, match="cdr3_amino_acid"):
            AAPLZGraph(bad_data)

    def test_non_dataframe_raises_error(self):
        """Verify that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="DataFrame"):
            AAPLZGraph(['sequence1', 'sequence2'])
