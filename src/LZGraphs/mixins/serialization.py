import json
import logging
import pickle
from pathlib import Path
from typing import Union, Optional

import networkx as nx
from networkx.readwrite import json_graph

from ..graphs.edge_data import EdgeData
from ..exceptions import UnsupportedFormatError

logger = logging.getLogger(__name__)


class SerializationMixin:
    """Mixin providing save/load, JSON serialization, AIRR import, and NetworkX export.

    Requirements:
        - self.graph (networkx.DiGraph)
        - self.has_gene_data (bool)
        - self.__dict__ (for __setstate__)
        - All public state attributes (for full serialization)
    """

    # Mapping from old attribute names to new names (for loading old pickles)
    _ATTR_MIGRATION = {
        'genetic': 'has_gene_data',
        'initial_states': 'initial_state_counts',
        'terminal_states': 'terminal_state_counts',
        'initial_states_probability': 'initial_state_probabilities',
        'subpattern_individual_probability': 'node_probability',
        'per_node_observed_frequency': 'node_outgoing_counts',
        'length_distribution_proba': 'length_probabilities',
        'length_distribution': 'length_counts',
        'n_subpatterns': 'num_subpatterns',
        'n_transitions': 'num_transitions',
        'marginal_vgenes': 'marginal_v_genes',
        'marginal_jgenes': 'marginal_j_genes',
        'genetic_walks_black_list': '_walk_exclusions',
        '_terminal_stop_dict': '_stop_probability_cache',
        'initial_state_threshold': 'min_initial_state_count',
        'cac_graphs': 'vj_combination_graphs',
        'edges_list': '_edges_cache',
        'n_neighbours': 'num_neighbours',
        'observed_vgenes': 'observed_v_genes',
        'observed_jgenes': 'observed_j_genes',
    }

    # Default column mapping from AIRR standard to LZGraphs internal names
    _AIRR_COLUMN_MAP = {
        'junction_aa': 'cdr3_amino_acid',
        'junction': 'cdr3_rearrangement',
        'v_call': 'V',
        'j_call': 'J',
    }

    def __setstate__(self, state):
        """Restore instance from pickle, migrating old attribute names and pandas types."""
        # Migrate old attribute names to new names
        for old_name, new_name in self._ATTR_MIGRATION.items():
            if old_name in state and new_name not in state:
                state[new_name] = state.pop(old_name)

        # Migrate 'wsif/sep' key inside terminal_state_data dicts
        tsd = state.get('terminal_state_data')
        if tsd is not None and isinstance(tsd, dict):
            for node_data in tsd.values():
                if isinstance(node_data, dict) and 'wsif/sep' in node_data:
                    node_data['stop_probability'] = node_data.pop('wsif/sep')

        self.__dict__.update(state)
        self._migrate_from_pandas()

    def _migrate_from_pandas(self):
        """Convert any old Series/DataFrame attributes to plain dicts.

        Called by ``__setstate__`` when loading old pickled graphs.
        """
        for attr in ('initial_state_counts', 'terminal_state_counts',
                      'initial_state_probabilities', 'length_probabilities',
                      'marginal_v_genes', 'marginal_j_genes', 'vj_probabilities',
                      'length_counts'):
            val = getattr(self, attr, None)
            if val is not None and hasattr(val, 'to_dict'):
                setattr(self, attr, val.to_dict())

        # node_probability was a DataFrame with 'proba' column
        sip = getattr(self, 'node_probability', None)
        if sip is not None and hasattr(sip, 'columns'):
            # Old DataFrame format
            if 'proba' in sip.columns:
                self.node_probability = sip['proba'].to_dict()
            else:
                self.node_probability = {}
        elif sip is not None and hasattr(sip, 'to_dict') and not isinstance(sip, dict):
            self.node_probability = sip.to_dict()

        # terminal_state_data: DataFrame -> dict-of-dicts
        tsd = getattr(self, 'terminal_state_data', None)
        if tsd is not None and hasattr(tsd, 'columns'):
            # Old DataFrame format — convert to dict-of-dicts
            # Handle old 'wsif/sep' column name from pre-v2.0.0 pickles
            stop_col = None
            if 'stop_probability' in tsd.columns:
                stop_col = 'stop_probability'
            elif 'wsif/sep' in tsd.columns:
                stop_col = 'wsif/sep'

            new_tsd = {}
            if stop_col is not None:
                for idx in tsd.index:
                    row = {col: tsd.loc[idx, col] for col in tsd.columns}
                    # Normalize key name to 'stop_probability'
                    if stop_col == 'wsif/sep' and 'wsif/sep' in row:
                        row['stop_probability'] = row.pop('wsif/sep')
                    new_tsd[idx] = row
                self._stop_probability_cache = tsd[stop_col].to_dict()
            self.terminal_state_data = new_tsd

        # Ensure caches exist
        if not hasattr(self, '_walk_cache'):
            self._walk_cache = None
        if not hasattr(self, '_stop_probability_cache'):
            self._stop_probability_cache = {}
        if not hasattr(self, '_topo_order'):
            self._topo_order = None

    def save(self, filepath: Union[str, Path], format: str = 'pickle',
             compress: bool = False) -> None:
        """
        Save the LZGraph to a file for later use.

        This method serializes the entire LZGraph object, including the underlying
        NetworkX graph and all computed attributes (probabilities, gene data, etc.).
        Saving avoids expensive re-computation when working with large repertoires.

        Args:
            filepath: Path where the graph will be saved. File extension is
                automatically added based on format if not present.
            format: Serialization format to use:
                - 'pickle' (default): Binary format, fastest and most complete.
                  Preserves all Python objects exactly. Recommended for most uses.
                - 'json': Human-readable text format. Useful for interoperability
                  but may not preserve all edge attributes with complex types.
            compress: If True, compress the output using gzip (adds .gz extension).
                Only supported for pickle format.

        Raises:
            ValueError: If an unsupported format is specified.
            IOError: If the file cannot be written.

        Example:
            >>> graph = AAPLZGraph(data)
            >>> graph.save('my_repertoire.pkl')
            >>> # Later...
            >>> loaded_graph = AAPLZGraph.load('my_repertoire.pkl')

        Note:
            The pickle format preserves the exact class type (AAPLZGraph, NDPLZGraph, etc.)
            so the loaded object will be the same subclass as the original.
        """
        filepath = Path(filepath)

        if format == 'pickle':
            # Add extension if not present
            if filepath.suffix not in ('.pkl', '.pickle', '.gz'):
                filepath = filepath.with_suffix('.pkl')

            if compress:
                import gzip
                if not filepath.suffix == '.gz':
                    filepath = Path(str(filepath) + '.gz')
                with gzip.open(filepath, 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == 'json':
            if filepath.suffix != '.json':
                filepath = filepath.with_suffix('.json')

            # Prepare serializable data
            data = self._to_json_dict()

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        else:
            raise UnsupportedFormatError(format=format)

        logger.info(f"LZGraph saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path], format: Optional[str] = None):
        """
        Load an LZGraph from a file.

        This class method reconstructs an LZGraph object from a previously saved file.
        The loaded graph will have all the same attributes and capabilities as the
        original, including probability calculations and random walk functionality.

        Args:
            filepath: Path to the saved graph file.
            format: Serialization format. If None, automatically detected from
                file extension:
                - '.pkl', '.pickle', '.gz' -> pickle format
                - '.json' -> json format

        Returns:
            The loaded LZGraph instance. The exact class type (AAPLZGraph, NDPLZGraph, etc.)
            is preserved when using pickle format.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the format cannot be determined or is unsupported.
            pickle.UnpicklingError: If the pickle file is corrupted.

        Example:
            >>> # Load a previously saved graph
            >>> graph = AAPLZGraph.load('my_repertoire.pkl')
            >>> # Use it immediately
            >>> prob = graph.walk_probability(sequence)

        Note:
            For pickle format, the actual class of the returned object may differ
            from the class used to call load(). For example, calling
            `LZGraphBase.load('aap_graph.pkl')` will return an AAPLZGraph instance
            if that's what was originally saved.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Auto-detect format from extension
        if format is None:
            if filepath.suffix in ('.pkl', '.pickle'):
                format = 'pickle'
            elif filepath.suffix == '.gz':
                format = 'pickle'  # Compressed pickle
            elif filepath.suffix == '.json':
                format = 'json'
            else:
                raise UnsupportedFormatError(
                    format=filepath.suffix,
                    message=f"Cannot determine format from extension '{filepath.suffix}'. "
                    "Please specify format='pickle' or format='json'."
                )

        if format == 'pickle':
            if filepath.suffix == '.gz':
                import gzip
                with gzip.open(filepath, 'rb') as f:
                    obj = pickle.load(f)
            else:
                with open(filepath, 'rb') as f:
                    obj = pickle.load(f)
            return obj

        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            return cls._from_json_dict(data)

        else:
            raise UnsupportedFormatError(format=format)

    def _to_json_dict(self) -> dict:
        """
        Convert the LZGraph to a JSON-serializable dictionary.

        This internal method prepares all graph data for JSON serialization.
        Complex types (DataFrames, Series) are converted to dictionaries.

        Returns:
            dict: A dictionary containing all graph data in JSON-compatible format.
        """
        # Temporarily replace EdgeData objects with legacy dicts for JSON serialization
        edge_data_backup = {}
        for a, b in self.graph.edges:
            ed = self.graph[a][b]['data']
            edge_data_backup[(a, b)] = ed
            self.graph[a][b]['data'] = ed.to_legacy_dict()

        # Convert NetworkX graph to node-link format
        graph_data = json_graph.node_link_data(self.graph)

        # Restore EdgeData objects
        for (a, b), ed in edge_data_backup.items():
            self.graph[a][b]['data'] = ed

        # Helper to convert any remaining pandas objects (backward compat)
        def _to_dict(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return obj

        data = {
            '_class': self.__class__.__name__,
            '_module': self.__class__.__module__,
            'graph': graph_data,
            'has_gene_data': self.has_gene_data,
            'num_subpatterns': self.num_subpatterns,
            'num_transitions': self.num_transitions,
            'initial_state_counts': _to_dict(self.initial_state_counts),
            'terminal_state_counts': _to_dict(self.terminal_state_counts),
            'lengths': self.lengths,
            'node_outgoing_counts': self.node_outgoing_counts,
            'initial_state_probabilities': _to_dict(self.initial_state_probabilities),
            'length_probabilities': _to_dict(self.length_probabilities),
            'node_probability': _to_dict(self.node_probability),
            'impute_missing_edges': self.impute_missing_edges,
            'smoothing_alpha': self.smoothing_alpha,
            'min_initial_state_count': self.min_initial_state_count,
        }

        # Add gene-related attributes if genetic
        if self.has_gene_data:
            if hasattr(self, 'marginal_v_genes'):
                data['marginal_v_genes'] = _to_dict(self.marginal_v_genes)
            if hasattr(self, 'marginal_j_genes'):
                data['marginal_j_genes'] = _to_dict(self.marginal_j_genes)
            if hasattr(self, 'vj_probabilities'):
                data['vj_probabilities'] = _to_dict(self.vj_probabilities)
            if hasattr(self, 'length_counts'):
                data['length_counts'] = _to_dict(self.length_counts)
            if hasattr(self, 'observed_v_genes'):
                data['observed_v_genes'] = list(self.observed_v_genes)
            if hasattr(self, 'observed_j_genes'):
                data['observed_j_genes'] = list(self.observed_j_genes)

        # Terminal state data
        if hasattr(self, 'terminal_state_data'):
            data['terminal_state_data'] = _to_dict(self.terminal_state_data)

        return data

    @classmethod
    def _from_json_dict(cls, data: dict):
        """
        Reconstruct an LZGraph from a JSON dictionary.

        This internal method rebuilds the graph from JSON-serialized data.

        Args:
            data: Dictionary containing serialized graph data.

        Returns:
            Reconstructed LZGraph instance.
        """
        # Helper to convert old pandas-serialized objects to plain dicts
        def _to_plain_dict(obj, default=None):
            if default is None:
                default = {}
            if obj is None:
                return default
            if isinstance(obj, dict):
                if '_type' in obj:
                    # Old pandas-serialized format
                    return obj.get('data', default)
                return obj
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return default

        # Import the correct class
        class_name = data.get('_class', 'LZGraphBase')
        module_name = data.get('_module', 'LZGraphs.graphs.lz_graph_base')

        # Try to get the actual class, fall back to current class
        try:
            import importlib
            module = importlib.import_module(module_name)
            graph_class = getattr(module, class_name)
        except (ImportError, AttributeError):
            graph_class = cls

        # Create instance without calling __init__
        instance = object.__new__(graph_class)

        # Restore NetworkX graph
        instance.graph = json_graph.node_link_graph(data['graph'])

        # Convert edge dicts to EdgeData objects
        per_node_freq = data.get('node_outgoing_counts', {})
        for a, b in list(instance.graph.edges):
            edge_attrs = dict(instance.graph[a][b])
            existing = edge_attrs.get('data')
            if isinstance(existing, EdgeData):
                continue  # Already an EdgeData (shouldn't happen from JSON, but be safe)
            # Determine the legacy dict: either nested under 'data' key or at top level
            if isinstance(existing, dict):
                legacy = existing
            else:
                legacy = edge_attrs
            node_freq = per_node_freq.get(a, 0)
            ed = EdgeData.from_legacy_dict(legacy, node_freq)
            # Clear old attrs and set EdgeData
            for key in list(instance.graph[a][b].keys()):
                del instance.graph[a][b][key]
            instance.graph[a][b]['data'] = ed

        # Restore basic attributes
        instance.has_gene_data = data.get('has_gene_data', data.get('genetic', False))
        instance._walk_exclusions = {}
        instance.num_subpatterns = data.get('num_subpatterns', data.get('n_subpatterns', 0))
        instance.num_transitions = data.get('num_transitions', data.get('n_transitions', 0))

        # Restore PGEN configuration
        instance.impute_missing_edges = data.get('impute_missing_edges', False)
        instance.smoothing_alpha = data.get('smoothing_alpha', 0.0)
        instance.min_initial_state_count = data.get('min_initial_state_count',
                                                     data.get('initial_state_threshold', 0))

        # Restore plain dict attributes (try new key, fall back to old)
        instance.initial_state_counts = _to_plain_dict(
            data.get('initial_state_counts', data.get('initial_states'))
        )
        instance.terminal_state_counts = _to_plain_dict(
            data.get('terminal_state_counts', data.get('terminal_states'))
        )
        instance.lengths = data.get('lengths', {})
        instance.vj_combination_graphs = {}
        instance.num_neighbours = {}
        instance.node_outgoing_counts = data.get('node_outgoing_counts',
                                                   data.get('per_node_observed_frequency', {}))

        # Restore probability dicts (try new key, fall back to old)
        instance.initial_state_probabilities = _to_plain_dict(
            data.get('initial_state_probabilities',
                      data.get('initial_states_probability'))
        )
        instance.length_probabilities = _to_plain_dict(
            data.get('length_probabilities',
                      data.get('length_distribution_proba'))
        )
        instance.node_probability = _to_plain_dict(
            data.get('node_probability',
                      data.get('subpattern_individual_probability'))
        )

        # Restore gene-related attributes if present
        if instance.has_gene_data:
            mg_v = data.get('marginal_v_genes', data.get('marginal_vgenes'))
            if mg_v is not None:
                instance.marginal_v_genes = _to_plain_dict(mg_v)
            mg_j = data.get('marginal_j_genes', data.get('marginal_jgenes'))
            if mg_j is not None:
                instance.marginal_j_genes = _to_plain_dict(mg_j)
            if 'vj_probabilities' in data:
                instance.vj_probabilities = _to_plain_dict(data['vj_probabilities'])
            lc = data.get('length_counts', data.get('length_distribution'))
            if lc is not None:
                instance.length_counts = _to_plain_dict(lc)
            ov = data.get('observed_v_genes', data.get('observed_vgenes'))
            if ov is not None:
                instance.observed_v_genes = set(ov)
            oj = data.get('observed_j_genes', data.get('observed_jgenes'))
            if oj is not None:
                instance.observed_j_genes = set(oj)

        # Restore terminal state data
        if 'terminal_state_data' in data:
            tsd_raw = data['terminal_state_data']
            instance.terminal_state_data = _to_plain_dict(tsd_raw)

        # Build _stop_probability_cache
        tsd = getattr(instance, 'terminal_state_data', None)
        if isinstance(tsd, dict):
            first_val = next(iter(tsd.values()), None) if tsd else None
            if isinstance(first_val, dict) and 'stop_probability' in first_val:
                # New format: dict of dicts
                instance._stop_probability_cache = {k: v['stop_probability'] for k, v in tsd.items()}
            else:
                # Flat dict — recompute
                instance._stop_probability_cache = {}
        else:
            instance._stop_probability_cache = {}

        # Walk cache (rebuilt lazily, not serialized)
        instance._walk_cache = None

        # Convert JSON string keys to int for lengths and terminal_states if needed
        if instance.initial_state_counts:
            instance.initial_state_counts = {
                k: int(v) if isinstance(v, float) and v == int(v) else v
                for k, v in instance.initial_state_counts.items()
            }
        if instance.terminal_state_counts:
            instance.terminal_state_counts = {
                k: int(v) if isinstance(v, float) and v == int(v) else v
                for k, v in instance.terminal_state_counts.items()
            }

        return instance

    def to_networkx(self) -> nx.DiGraph:
        """
        Export the underlying graph as a pure NetworkX DiGraph.

        This method returns a copy of the internal graph structure with
        EdgeData objects converted to flat dictionaries (weight, count,
        gene probabilities) for compatibility with standard NetworkX tools.

        Returns:
            nx.DiGraph: A directed graph with flat edge attribute dicts.

        Example:
            >>> lzgraph = AAPLZGraph(data)
            >>> G = lzgraph.to_networkx()
            >>> # Use standard NetworkX functions
            >>> nx.draw(G)
            >>> components = list(nx.weakly_connected_components(G))
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.graph.nodes(data=True))
        for a, b in self.graph.edges:
            G.add_edge(a, b, **self.graph[a][b]['data'].to_legacy_dict())
        return G

    def get_graph_metadata(self) -> dict:
        """
        Get a summary of graph metadata for inspection.

        Returns a dictionary containing key statistics about the graph,
        useful for debugging or logging.

        Returns:
            dict: Dictionary with graph statistics including node count,
                edge count, genetic status, and size information.
        """
        return {
            'class': self.__class__.__name__,
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'genetic': self.has_gene_data,
            'n_subpatterns': self.num_subpatterns,
            'n_transitions': self.num_transitions,
            'n_initial_states': len(self.initial_state_counts) if hasattr(self, 'initial_state_counts') else 0,
            'n_terminal_states': len(self.terminal_state_counts) if hasattr(self, 'terminal_state_counts') else 0,
        }

    @classmethod
    def from_airr(cls, data, column_map=None, **kwargs):
        """
        Create an LZGraph from AIRR-formatted data.

        The Adaptive Immune Receptor Repertoire (AIRR) standard uses different
        column names than LZGraphs. This classmethod automatically maps AIRR
        column names to the LZGraphs convention.

        Default mapping:
            - junction_aa -> cdr3_amino_acid
            - junction -> cdr3_rearrangement
            - v_call -> V
            - j_call -> J

        Args:
            data: Either a DataFrame-like object with AIRR column names,
                or a file path to an AIRR-format TSV file.
            column_map (dict, optional): Custom column name mapping from AIRR to
                LZGraphs format. Merged with defaults (custom takes precedence).
            **kwargs: Additional keyword arguments passed to the graph constructor
                (e.g., verbose, calculate_trainset_pgen).

        Returns:
            An instance of the graph class (AAPLZGraph, NDPLZGraph, etc.)

        Example:
            >>> # From AIRR DataFrame
            >>> graph = AAPLZGraph.from_airr(airr_df)
            >>>
            >>> # From AIRR TSV file
            >>> graph = NDPLZGraph.from_airr("repertoire.tsv", verbose=False)
            >>>
            >>> # With custom column mapping
            >>> graph = AAPLZGraph.from_airr(df, column_map={'amino_acid': 'cdr3_amino_acid'})
        """
        import csv

        # Load from file if path provided
        if isinstance(data, (str, Path)):
            with open(data, newline='') as fh:
                reader = csv.DictReader(fh, delimiter='\t')
                rows = list(reader)
            # Convert list-of-dicts → dict-of-lists
            if rows:
                data = {col: [row[col] for row in rows] for col in rows[0]}
            else:
                data = {}

        # Build effective column map
        effective_map = dict(cls._AIRR_COLUMN_MAP)
        if column_map:
            effective_map.update(column_map)

        # Rename columns (works for both DataFrame and dict-of-lists)
        if hasattr(data, 'columns'):
            # DataFrame-like
            rename_map = {
                airr_col: lzg_col
                for airr_col, lzg_col in effective_map.items()
                if airr_col in data.columns and lzg_col not in data.columns
            }
            if rename_map:
                data = data.rename(columns=rename_map)
        elif isinstance(data, dict):
            # dict-of-lists
            for airr_col, lzg_col in effective_map.items():
                if airr_col in data and lzg_col not in data:
                    data[lzg_col] = data.pop(airr_col)

        return cls(data, **kwargs)
