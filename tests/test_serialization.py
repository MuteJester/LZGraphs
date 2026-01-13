"""
Tests for Graph Serialization
=============================

Tests covering the save/load functionality for LZGraph persistence:
- Pickle format (binary, fast, complete)
- JSON format (human-readable, interoperable)
- Compressed pickle format
- Cross-class loading
- Data integrity after round-trip

Test Categories:
- Basic save/load operations
- Format-specific behavior
- Data integrity verification
- Error handling
"""

import pytest
import tempfile
import os
from pathlib import Path
import numpy as np
import networkx as nx

from LZGraphs import AAPLZGraph, NDPLZGraph, UnsupportedFormatError


class TestPickleSerialization:
    """Tests for pickle-based serialization."""

    def test_save_and_load_pickle_aap(self, aap_lzgraph):
        """Verify AAPLZGraph can be saved and loaded with pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'

            # Save
            aap_lzgraph.save(filepath, format='pickle')
            assert filepath.exists()

            # Load
            loaded = AAPLZGraph.load(filepath)

            # Verify type preserved
            assert isinstance(loaded, AAPLZGraph)

    def test_save_and_load_pickle_ndp(self, ndp_lzgraph):
        """Verify NDPLZGraph can be saved and loaded with pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'

            # Save
            ndp_lzgraph.save(filepath, format='pickle')
            assert filepath.exists()

            # Load
            loaded = NDPLZGraph.load(filepath)

            # Verify type preserved
            assert isinstance(loaded, NDPLZGraph)

    def test_pickle_preserves_edge_count(self, aap_lzgraph):
        """Verify edge count is preserved after save/load."""
        original_edge_count = len(aap_lzgraph.edges)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        assert len(loaded.edges) == original_edge_count

    def test_pickle_preserves_node_count(self, aap_lzgraph):
        """Verify node count is preserved after save/load."""
        original_node_count = len(aap_lzgraph.nodes)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        assert len(loaded.nodes) == original_node_count

    def test_pickle_preserves_initial_states(self, aap_lzgraph):
        """Verify initial states are preserved after save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        # Check a specific initial state
        assert loaded.initial_states['C_1'] == aap_lzgraph.initial_states['C_1']

    def test_pickle_preserves_probabilities(self, aap_lzgraph):
        """Verify subpattern probabilities are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        # Check probability data is the same
        original_prob = aap_lzgraph.subpattern_individual_probability
        loaded_prob = loaded.subpattern_individual_probability

        assert original_prob.shape == loaded_prob.shape
        assert np.allclose(
            original_prob['proba'].values,
            loaded_prob['proba'].values,
            rtol=1e-10
        )

    def test_pickle_auto_extension(self, aap_lzgraph):
        """Verify .pkl extension is added automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph'  # No extension

            aap_lzgraph.save(filepath, format='pickle')

            # Should have added .pkl
            expected_path = Path(tmpdir) / 'test_graph.pkl'
            assert expected_path.exists()


class TestCompressedPickle:
    """Tests for compressed pickle serialization."""

    def test_save_compressed_pickle(self, aap_lzgraph):
        """Verify compressed pickle saves correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'

            aap_lzgraph.save(filepath, format='pickle', compress=True)

            # Should have .gz extension
            expected_path = Path(tmpdir) / 'test_graph.pkl.gz'
            assert expected_path.exists()

    def test_load_compressed_pickle(self, aap_lzgraph):
        """Verify compressed pickle loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.pkl'

            aap_lzgraph.save(filepath, format='pickle', compress=True)
            expected_path = Path(tmpdir) / 'test_graph.pkl.gz'

            loaded = AAPLZGraph.load(expected_path)

            assert isinstance(loaded, AAPLZGraph)
            assert len(loaded.edges) == len(aap_lzgraph.edges)

    def test_compressed_smaller_than_uncompressed(self, aap_lzgraph):
        """Verify compressed file is smaller than uncompressed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            uncompressed_path = Path(tmpdir) / 'uncompressed.pkl'
            compressed_path = Path(tmpdir) / 'compressed.pkl'

            aap_lzgraph.save(uncompressed_path, format='pickle', compress=False)
            aap_lzgraph.save(compressed_path, format='pickle', compress=True)

            uncompressed_size = uncompressed_path.stat().st_size
            compressed_size = (Path(tmpdir) / 'compressed.pkl.gz').stat().st_size

            assert compressed_size < uncompressed_size


class TestJSONSerialization:
    """Tests for JSON-based serialization."""

    def test_save_and_load_json(self, aap_lzgraph):
        """Verify graph can be saved and loaded with JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.json'

            aap_lzgraph.save(filepath, format='json')
            assert filepath.exists()

            loaded = AAPLZGraph.load(filepath)
            assert isinstance(loaded, AAPLZGraph)

    def test_json_preserves_structure(self, aap_lzgraph):
        """Verify JSON preserves basic graph structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.json'

            aap_lzgraph.save(filepath, format='json')
            loaded = AAPLZGraph.load(filepath)

        # Check basic structure preserved
        assert len(loaded.nodes) == len(aap_lzgraph.nodes)
        assert len(loaded.edges) == len(aap_lzgraph.edges)

    def test_json_auto_extension(self, aap_lzgraph):
        """Verify .json extension is added automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph'  # No extension

            aap_lzgraph.save(filepath, format='json')

            expected_path = Path(tmpdir) / 'test_graph.json'
            assert expected_path.exists()

    def test_json_is_readable(self, aap_lzgraph):
        """Verify JSON output is human-readable."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_graph.json'

            aap_lzgraph.save(filepath, format='json')

            # Should be valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)

            assert '_class' in data
            assert data['_class'] == 'AAPLZGraph'
            assert 'graph' in data


class TestFormatDetection:
    """Tests for automatic format detection."""

    def test_auto_detect_pickle(self, aap_lzgraph):
        """Verify pickle format is auto-detected from .pkl extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            aap_lzgraph.save(filepath, format='pickle')

            # Load without specifying format
            loaded = AAPLZGraph.load(filepath)
            assert isinstance(loaded, AAPLZGraph)

    def test_auto_detect_json(self, aap_lzgraph):
        """Verify JSON format is auto-detected from .json extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.json'
            aap_lzgraph.save(filepath, format='json')

            # Load without specifying format
            loaded = AAPLZGraph.load(filepath)
            assert isinstance(loaded, AAPLZGraph)

    def test_auto_detect_gz(self, aap_lzgraph):
        """Verify compressed format is auto-detected from .gz extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            aap_lzgraph.save(filepath, format='pickle', compress=True)

            # Load without specifying format
            gz_path = Path(tmpdir) / 'test.pkl.gz'
            loaded = AAPLZGraph.load(gz_path)
            assert isinstance(loaded, AAPLZGraph)


class TestErrorHandling:
    """Tests for error handling in serialization."""

    def test_load_nonexistent_file_raises_error(self):
        """Verify loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AAPLZGraph.load('/nonexistent/path/graph.pkl')

    def test_unsupported_save_format_raises_error(self, aap_lzgraph):
        """Verify unsupported format raises UnsupportedFormatError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.xyz'

            with pytest.raises(UnsupportedFormatError):
                aap_lzgraph.save(filepath, format='xyz')

    def test_unknown_extension_without_format_raises_error(self, aap_lzgraph):
        """Verify unknown extension without explicit format raises UnsupportedFormatError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with unknown extension
            filepath = Path(tmpdir) / 'test.xyz'
            filepath.touch()

            with pytest.raises(UnsupportedFormatError):
                AAPLZGraph.load(filepath)


class TestNetworkXExport:
    """Tests for NetworkX export functionality."""

    def test_to_networkx_returns_digraph(self, aap_lzgraph):
        """Verify to_networkx returns a NetworkX DiGraph."""
        G = aap_lzgraph.to_networkx()
        assert isinstance(G, nx.DiGraph)

    def test_to_networkx_preserves_nodes(self, aap_lzgraph):
        """Verify exported graph has same nodes."""
        G = aap_lzgraph.to_networkx()
        assert set(G.nodes()) == set(aap_lzgraph.nodes)

    def test_to_networkx_preserves_edges(self, aap_lzgraph):
        """Verify exported graph has same edges."""
        G = aap_lzgraph.to_networkx()
        assert set(G.edges()) == set(aap_lzgraph.edges)

    def test_to_networkx_returns_copy(self, aap_lzgraph):
        """Verify to_networkx returns a copy, not the original."""
        G = aap_lzgraph.to_networkx()

        # Modify the copy
        G.add_node('TEST_NODE')

        # Original should be unchanged
        assert 'TEST_NODE' not in aap_lzgraph.nodes

    def test_to_networkx_preserves_edge_weights(self, aap_lzgraph):
        """Verify edge weights are preserved in export."""
        G = aap_lzgraph.to_networkx()

        # Check a few edge weights
        for u, v in list(aap_lzgraph.edges)[:5]:
            original_weight = aap_lzgraph.graph[u][v]['weight']
            exported_weight = G[u][v]['weight']
            assert np.isclose(original_weight, exported_weight)


class TestGraphMetadata:
    """Tests for graph metadata extraction."""

    def test_get_graph_metadata_returns_dict(self, aap_lzgraph):
        """Verify get_graph_metadata returns a dictionary."""
        metadata = aap_lzgraph.get_graph_metadata()
        assert isinstance(metadata, dict)

    def test_metadata_contains_expected_keys(self, aap_lzgraph):
        """Verify metadata contains expected keys."""
        metadata = aap_lzgraph.get_graph_metadata()

        expected_keys = [
            'class', 'n_nodes', 'n_edges', 'genetic',
            'n_subpatterns', 'n_transitions',
            'n_initial_states', 'n_terminal_states'
        ]

        for key in expected_keys:
            assert key in metadata, f"Missing key: {key}"

    def test_metadata_class_name_correct(self, aap_lzgraph, ndp_lzgraph):
        """Verify class name is reported correctly."""
        aap_metadata = aap_lzgraph.get_graph_metadata()
        ndp_metadata = ndp_lzgraph.get_graph_metadata()

        assert aap_metadata['class'] == 'AAPLZGraph'
        assert ndp_metadata['class'] == 'NDPLZGraph'

    def test_metadata_counts_match_graph(self, aap_lzgraph):
        """Verify metadata counts match actual graph."""
        metadata = aap_lzgraph.get_graph_metadata()

        assert metadata['n_nodes'] == aap_lzgraph.graph.number_of_nodes()
        assert metadata['n_edges'] == aap_lzgraph.graph.number_of_edges()


class TestDataIntegrityAfterRoundTrip:
    """Tests for complete data integrity after save/load cycle."""

    def test_walk_probability_unchanged_after_pickle(self, aap_lzgraph):
        """Verify walk_probability gives same result after pickle round-trip."""
        test_sequence = "CASSLGQAYEQYF"
        walk = AAPLZGraph.encode_sequence(test_sequence)

        original_prob = aap_lzgraph.walk_probability(walk, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        loaded_prob = loaded.walk_probability(walk, verbose=False)

        assert np.isclose(original_prob, loaded_prob, rtol=1e-10)

    def test_random_walk_works_after_load(self, aap_lzgraph):
        """Verify random walks work on loaded graph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        # Should be able to generate random walk
        walk = loaded.unsupervised_random_walk()
        assert walk is not None

    def test_gene_data_preserved_after_pickle(self, aap_lzgraph):
        """Verify gene-related data is preserved after pickle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        # Check marginal V gene probabilities
        assert loaded.marginal_vgenes.equals(aap_lzgraph.marginal_vgenes)

        # Check length distribution
        assert loaded.lengths == aap_lzgraph.lengths

    def test_terminal_states_preserved(self, aap_lzgraph):
        """Verify terminal states are preserved after round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.pkl'
            aap_lzgraph.save(filepath)
            loaded = AAPLZGraph.load(filepath)

        # Check terminal states match
        assert loaded.terminal_states.equals(aap_lzgraph.terminal_states)
