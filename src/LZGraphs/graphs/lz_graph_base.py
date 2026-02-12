import logging
import json
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Union, Optional

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import pandas as pd

# Utility functions
from ..utilities.misc import choice, window
from ..utilities.decomposition import lempel_ziv_decomposition

# EdgeData
from .edge_data import EdgeData

# The three mixins
from ..mixins import GeneLogicMixin
from ..mixins import RandomWalkMixin
from ..mixins import GenePredictionMixin

# Custom exceptions
from ..exceptions import UnsupportedFormatError, GeneAnnotationError, NoGeneDataError

# Create a logger for this module
logger = logging.getLogger(__name__)

# Machine epsilon — cached once at module level (avoids repeated np.finfo calls)
_EPS = np.finfo(np.float64).eps
_LOG_EPS = np.log(_EPS)


def _dicts_close(a, b, decimals=3):
    """Compare two dicts with rounding tolerance.

    Works on both plain dicts and pd.Series (for backward compat).
    """
    # Convert pd.Series to dict if needed (backward compat with old pickles)
    if hasattr(a, 'to_dict'):
        a = a.to_dict()
    if hasattr(b, 'to_dict'):
        b = b.to_dict()
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        if round(float(a[k]), decimals) != round(float(b[k]), decimals):
            return False
    return True


class LZGraphBase(ABC, GeneLogicMixin, RandomWalkMixin, GenePredictionMixin):
    """
    This abstract class provides the base functionality and attributes
    shared between different LZGraph implementations (excluding the Naive LZGraph).
    It inherits from:
      - GeneLogicMixin: For V/J gene loading & edge annotation logic.
      - RandomWalkMixin: For genomic & non-genomic random walks with blacklisting.
      - GenePredictionMixin: For various gene-prediction heuristics.
    """

    def __init__(self):
        # start time of constructor
        self.constructor_start_time = time()
        # create graph
        self.graph = nx.DiGraph()

        # genetics
        self.has_gene_data = False
        self._walk_exclusions = {}

        # sub-pattern count, transitions, etc.
        self.num_subpatterns = 0
        self.num_transitions = 0

        self.initial_state_counts, self.terminal_state_counts = dict(), dict()
        self.initial_state_probabilities = {}
        self.lengths = dict()
        self.vj_combination_graphs = dict()
        self.num_neighbours = dict()
        self.node_outgoing_counts = dict()

        self.length_probabilities = {}
        self.node_probability = {}

        # PGEN configuration (overridden by subclasses)
        self.impute_missing_edges = False
        self.smoothing_alpha = 0.0
        self.min_initial_state_count = 0

        # Fast stop probability lookup (populated by _derive_stop_probability_data)
        self._stop_probability_cache = {}
        # Walk cache for simulate() (built lazily)
        self._walk_cache = None
        # Topological order cache (built lazily, invalidated on structural changes)
        self._topo_order = None

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

    @staticmethod
    def _normalize_input(data, seq_column, abundances=None, v_genes=None, j_genes=None):
        """Convert flexible input to a DataFrame for the build pipeline.

        Accepts a DataFrame (returned as-is), a list of strings, or a
        pandas Series.  Optional *abundances*, *v_genes*, and *j_genes*
        lists are only allowed when *data* is **not** already a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            if abundances is not None or v_genes is not None or j_genes is not None:
                raise TypeError(
                    "Cannot pass abundances/v_genes/j_genes when data is "
                    "already a DataFrame — use DataFrame columns instead"
                )
            return data

        # Convert list / tuple / Series → plain list
        if isinstance(data, pd.Series):
            sequences = data.tolist()
        elif isinstance(data, (list, tuple)):
            sequences = list(data)
        else:
            raise TypeError(
                f"data must be a DataFrame, list, or Series, "
                f"got {type(data).__name__}"
            )

        df_dict = {seq_column: sequences}
        n = len(sequences)

        if abundances is not None:
            if len(abundances) != n:
                raise ValueError(
                    f"abundances length ({len(abundances)}) != "
                    f"sequences length ({n})"
                )
            df_dict["abundance"] = abundances

        if v_genes is not None or j_genes is not None:
            if v_genes is None or j_genes is None:
                raise ValueError(
                    "v_genes and j_genes must both be provided or both omitted"
                )
            if len(v_genes) != n or len(j_genes) != n:
                raise ValueError(
                    f"v_genes/j_genes length must match sequences length ({n})"
                )
            df_dict["V"] = v_genes
            df_dict["J"] = j_genes

        return pd.DataFrame(df_dict)

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
        """Convert any old pd.Series/DataFrame attributes to plain dicts.

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

    def __getattr__(self, name):
        """Backward compatibility for attributes added after v2.0.0.

        Handles old pickled graphs that lack ``_stop_probability_cache`` or
        ``_walk_cache``.
        """
        if name == '_stop_probability_cache':
            tsd = self.__dict__.get('terminal_state_data')
            if tsd is not None:
                if isinstance(tsd, dict):
                    # Dict of dicts — handle both 'stop_probability' and old 'wsif/sep' keys
                    cache = {}
                    for k, v in tsd.items():
                        if isinstance(v, dict):
                            cache[k] = v.get('stop_probability', v.get('wsif/sep', 0.0))
                    self._stop_probability_cache = cache
                elif hasattr(tsd, 'columns'):
                    # Old format: pandas DataFrame
                    if 'stop_probability' in tsd.columns:
                        self._stop_probability_cache = tsd['stop_probability'].to_dict()
                    elif 'wsif/sep' in tsd.columns:
                        self._stop_probability_cache = tsd['wsif/sep'].to_dict()
                    else:
                        self._stop_probability_cache = {}
                else:
                    self._stop_probability_cache = {}
            else:
                self._stop_probability_cache = {}
            return self._stop_probability_cache
        if name == '_walk_cache':
            self._walk_cache = None
            return None
        if name == '_topo_order':
            self._topo_order = None
            return None
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __eq__(self, other):
        """
        This method tests whether two LZGraphs are equal, i.e. have the same node, edges,
        and metadata on the edges.
        """
        # Check if graphs have same structure
        if not nx.utils.graphs_equal(self.graph, other.graph):
            return False

        # Check if both have same genetic status
        if self.has_gene_data != other.has_gene_data:
            return False

        aux = 0
        aux += self._walk_exclusions != other._walk_exclusions
        aux += self.num_subpatterns != other.num_subpatterns

        # Compare initial_states, terminal_states, etc.
        aux += not _dicts_close(self.initial_state_counts, other.initial_state_counts, decimals=3)
        aux += not _dicts_close(self.terminal_state_counts, other.terminal_state_counts, decimals=3)
        aux += not _dicts_close(self.length_probabilities, other.length_probabilities, decimals=3)

        # Compare gene-related distributions only if both are genetic
        if self.has_gene_data and other.has_gene_data:
            aux += not _dicts_close(self.marginal_v_genes, other.marginal_v_genes, decimals=3)
            aux += not _dicts_close(self.vj_probabilities, other.vj_probabilities, decimals=3)
            aux += not _dicts_close(self.length_counts, other.length_counts, decimals=3)

        return (aux == 0)

    def __repr__(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        genetic_str = "genetic" if self.has_gene_data else "non-genetic"
        return f"{self.__class__.__name__}(nodes={n_nodes}, edges={n_edges}, {genetic_str})"

    @staticmethod
    @abstractmethod
    def encode_sequence(sequence):
        """
        Abstract method: should be overridden by subclasses.
        Processes a sequence of symbols (nucleotides/amino acids) into a list of subpatterns (nodes).
        """
        ...

    @staticmethod
    def extract_subpattern(base):
        """
        Extract only nucleotides from a node string containing frame/position info.
        """
        match = re.search(r'[ATGC]+', base)
        return match.group(0) if match else ""

    @abstractmethod
    def _decomposed_sequence_generator(self, data):
        """
        Abstract method: should be overridden by subclasses.
        Yields (steps, locations[, v, j]) per row for building the graph.
        """
        ...

    def _simultaneous_graph_construction(self, data: pd.DataFrame):
        """
        Leverages the generator implemented in `_decomposed_sequence_generator`
        to insert node/edge data into the networkx DiGraph. If `self.has_gene_data` is True,
        we also embed gene information.

        The generator yields tuples that may include a ``count`` field (abundance weight).
        When present, each edge/node/state is incremented by ``count`` rather than 1.
        """
        processing_stream = self._decomposed_sequence_generator(data)
        if self.has_gene_data:
            for output in processing_stream:
                steps, locations, v, j = output[:4]
                count = output[4] if len(output) > 4 else 1
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.node_outgoing_counts[A_] = self.node_outgoing_counts.get(A_, 0) + count
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information(A_, B_, v, j, count=count)
                # ensure final node exists in frequency dict
                self.node_outgoing_counts[B_] = self.node_outgoing_counts.get(B_, 0)
        else:
            for output in processing_stream:
                steps, locations = output[:2]
                count = output[2] if len(output) > 2 else 1
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.node_outgoing_counts[A_] = self.node_outgoing_counts.get(A_, 0) + count
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information_no_genes(A_, B_, count=count)
                # ensure final node exists in frequency dict
                self.node_outgoing_counts[B_] = self.node_outgoing_counts.get(B_, 0)

    def _insert_edge_and_information_no_genes(self, node_a, node_b, count=1):
        """
        Insert or update an edge (node_a -> node_b) with no gene info.

        Args:
            node_a: Source node.
            node_b: Target node.
            count (int): Number of traversals (abundance weight). Default 1.
        """
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]['data'].record(count=count)
        else:
            ed = EdgeData()
            ed.record(count=count)
            self.graph.add_edge(node_a, node_b, data=ed)
        self.num_transitions += count

    def _normalize_edge_weights(self):
        """
        Normalize each outgoing edge weight by the total count for its source node.
        The sum of outgoing edges from any node then becomes 1.

        If ``self.smoothing_alpha > 0``, Laplace smoothing is applied:
        P(B|A) = (count(A→B) + α) / (freq(A) + α × K)
        where K is the number of observed successors of A.
        """
        alpha = self.smoothing_alpha
        if alpha == 0.0:
            for edge_a, edge_b in self.graph.edges:
                ed = self.graph[edge_a][edge_b]['data']
                total = self.node_outgoing_counts.get(edge_a, 0)
                ed.normalize(total)
        else:
            for node in self.graph.nodes():
                successors = list(self.graph.successors(node))
                if not successors:
                    continue
                k = len(successors)
                raw_total = self.node_outgoing_counts.get(node, 0)
                for succ in successors:
                    self.graph[node][succ]['data'].normalize(raw_total, alpha, k)

    def _get_node_info_df(self, node_a, V=None, J=None, condition='and'):
        """
        Returns a DataFrame containing metadata for edges from node_a,
        optionally filtered by presence of V/J genes (with 'and'/'or' logic).
        """
        node_data = self.graph[node_a]
        if V is None or J is None:
            return pd.DataFrame({
                nb: self.graph[node_a][nb]['data'].to_legacy_dict()
                for nb in node_data
            })
        else:
            if condition == 'and':
                partial_dict = {
                    nb: self.graph[node_a][nb]['data'].to_legacy_dict()
                    for nb in node_data
                    if self.graph[node_a][nb]['data'].has_gene(V) and self.graph[node_a][nb]['data'].has_gene(J)
                }
            else:
                partial_dict = {
                    nb: self.graph[node_a][nb]['data'].to_legacy_dict()
                    for nb in node_data
                    if self.graph[node_a][nb]['data'].has_gene(V) or self.graph[node_a][nb]['data'].has_gene(J)
                }
            return pd.DataFrame(partial_dict)

    def _get_node_feature_info_df(self, node_a, feature, V=None, J=None, asdict=False):
        """
        Similar to `_get_node_info_df` but returns only a specific `feature`.
        If asdict=True, returns a dict for easier key-based manipulations.
        """
        node_data = self.graph[node_a]
        if V is None or J is None:
            # Return everything if V/J not specified
            result = {
                nb: self.graph[node_a][nb]['data'].to_legacy_dict()
                for nb in node_data
            }
            if asdict:
                return result
            return pd.DataFrame(result)

        # Filter edges to those containing V and J
        partial_dict = {}
        for nb in node_data:
            ed = self.graph[node_a][nb]['data']
            if ed.has_gene(V) and ed.has_gene(J):
                partial_dict[nb] = {feature: ed.weight if feature == 'weight' else ed.to_legacy_dict().get(feature)}
        if asdict:
            return partial_dict
        else:
            return pd.DataFrame(partial_dict)

    def is_stop_condition(self, state, selected_v=None, selected_j=None):
        """
        Decide if a walk should stop at 'state'.
        If `self.has_gene_data` is True, we check V/J constraints plus
        a random stop probability.
        """
        stop_prob = self._stop_probability_cache.get(state)
        if stop_prob is None:
            return False

        if self.has_gene_data:
            if selected_j is not None:
                # Check whether we have neighbors that contain both selected_v and selected_j
                has_compatible = False
                for nb in self.graph[state]:
                    ed = self.graph[state][nb]['data']
                    if ed.has_gene(selected_v) and ed.has_gene(selected_j):
                        has_compatible = True
                        break
            else:
                has_compatible = self.graph.out_degree(state) > 0

            if not has_compatible:
                return True
            else:
                return np.random.random() < stop_prob
        else:
            return np.random.random() < stop_prob

    def _derive_node_probability(self):
        """
        Summation of edge counts by source node -> yields empirical probabilities for each node.
        Uses raw counts (not normalized weights) so this can be called before or after normalization.
        """
        node_counts = {}
        for a, b in self.graph.edges:
            count = self.graph[a][b]['data'].count
            node_counts[a] = node_counts.get(a, 0) + count
        total = sum(node_counts.values())
        if total > 0:
            self.node_probability = {
                node: c / total for node, c in node_counts.items()
            }
        else:
            self.node_probability = {}

    def verbose_driver(self, message_number, verbose):
        """
        Replaces the print-based driver with logging calls for more professional output.
        If verbose == False, no logs will be emitted.
        """
        if not verbose:
            return

        elapsed = round(time() - self.constructor_start_time, 2)

        # You could store your messages in a dict to keep them organized,
        # but here's the direct if/elif approach:
        if message_number == -2:
            logger.info("======================================\n")
        elif message_number == 0:
            logger.info(f"[{elapsed:.2f}s] Gene Information Loaded.")
        elif message_number == 1:
            logger.info(f"[{elapsed:.2f}s] Graph Constructed.")
        elif message_number == 2:
            logger.info(f"[{elapsed:.2f}s] Graph Metadata Derived.")
        elif message_number == 3:
            logger.info(f"[{elapsed:.2f}s] Graph Edge Weights Normalized.")
        elif message_number == 6:
            # We assume there's a self.constructor_end_time set somewhere
            total_time = round(self.constructor_end_time - self.constructor_start_time, 2)
            logger.info(f"[{total_time:.2f}s] LZGraph Created Successfully.")
        elif message_number == 8:
            logger.info(f"[{elapsed:.2f}s] Individual Subpattern Empirical Probability Derived.")
        elif message_number == 9:
            logger.info(f"[{elapsed:.2f}s] Terminal State Conditional Probabilities Map Derived.")
        else:
            logger.info(f"[{elapsed:.2f}s] Unrecognized Message ID: {message_number}")

    def random_step(self, state):
        """
        Given the current state, pick and take a random step based on the translation probabilities.
        """
        states, probabilities = self._get_state_weights(state)
        return choice(states, probabilities)

    def _random_initial_state(self):
        """
        Select a random initial state based on the marginal distribution of initial states.
        """
        keys = list(self.initial_state_probabilities.keys())
        vals = list(self.initial_state_probabilities.values())
        return choice(keys, vals)

    def _get_state_weights(self, node, v=None, j=None):
        """
        Return all possible next-states (edges) from `node` and their respective weights.
        """
        if v is None and j is None:
            node_data = self.graph[node]
            states = list(node_data.keys())
            probabilities = [node_data[s]['data'].weight for s in states]
            return states, probabilities
        else:
            result = {
                nb: self.graph[node][nb]['data'].to_legacy_dict()
                for nb in self.graph[node]
            }
            df = pd.DataFrame(result).T
            return df

    def _update_terminal_states(self, terminal_state, count=1):
        self.terminal_state_counts[terminal_state] = self.terminal_state_counts.get(terminal_state, 0) + count

    def _update_initial_states(self, initial_state, count=1):
        self.initial_state_counts[initial_state] = self.initial_state_counts.get(initial_state, 0) + count

    def edge_data(self, a, b):
        """Shortcut to access EdgeData for edge a->b."""
        return self.graph[a][b]['data']

    def outgoing_edges(self, node):
        """Return {neighbor: EdgeData} for all outgoing edges from node."""
        return {nb: self.graph[node][nb]['data'] for nb in self.graph[node]}

    def recalculate(self):
        """Recompute all derived attributes from raw counts.

        Call this after modifying raw counts (e.g., after graph_union
        or remove_sequence) to update all cached probabilities.
        """
        # Recompute node_outgoing_counts from actual edge counts.
        # freq[node] = sum of outgoing edge counts (matching construction convention).
        self.node_outgoing_counts = {}
        for node in self.graph.nodes():
            outgoing_sum = sum(
                self.graph[node][succ]['data'].count
                for succ in self.graph.successors(node)
            )
            if outgoing_sum > 0:
                self.node_outgoing_counts[node] = outgoing_sum

        # Initial state probabilities
        if self.initial_state_counts:
            total = sum(self.initial_state_counts.values())
            if total > 0:
                self.initial_state_probabilities = {
                    k: v / total for k, v in self.initial_state_counts.items()
                }

        # Length distribution probabilities
        if self.terminal_state_counts:
            total = sum(self.terminal_state_counts.values())
            if total > 0:
                self.length_probabilities = {
                    k: v / total for k, v in self.terminal_state_counts.items()
                }

        # Normalize edge weights
        self._normalize_edge_weights()

        # Derived probability tables
        self._derive_node_probability()
        self._derive_stop_probability_data()

        # Invalidate caches
        self._walk_cache = None
        self._topo_order = None

    def remove_sequence(self, sequence, v_gene=None, j_gene=None):
        """Remove a single sequence's contribution from the graph.

        Decrements edge counts along the sequence's path, updates node
        frequencies, initial/terminal state counts, and recalculates all
        derived probabilities.

        Args:
            sequence (str): The raw sequence to remove (amino acid or nucleotide).
            v_gene (str, optional): V gene associated with this sequence.
            j_gene (str, optional): J gene associated with this sequence.
        """
        walk = self.encode_sequence(sequence)
        if len(walk) == 0:
            return

        # Decrement initial/terminal state counts
        first_node, last_node = walk[0], walk[-1]
        if first_node in self.initial_state_counts:
            self.initial_state_counts[first_node] = max(0, self.initial_state_counts[first_node] - 1)
            if self.initial_state_counts[first_node] <= 0:
                del self.initial_state_counts[first_node]
        if last_node in self.terminal_state_counts:
            self.terminal_state_counts[last_node] = max(0, self.terminal_state_counts[last_node] - 1)
            if self.terminal_state_counts[last_node] <= 0:
                del self.terminal_state_counts[last_node]

        # Decrement edge counts
        for a, b in window(walk, 2):
            if self.graph.has_edge(a, b):
                self.graph[a][b]['data'].unrecord(v_gene=v_gene, j_gene=j_gene)
                if self.graph[a][b]['data'].count <= 0:
                    self.graph.remove_edge(a, b)

        # Remove isolated nodes (no edges left)
        for node in walk:
            if (node in self.graph
                    and self.graph.in_degree(node) == 0
                    and self.graph.out_degree(node) == 0):
                self.graph.remove_node(node)

        # Update length tracking
        seq_len = len(sequence)
        if seq_len in self.lengths:
            self.lengths[seq_len] -= 1
            if self.lengths[seq_len] <= 0:
                del self.lengths[seq_len]

        self.num_subpatterns = max(0, self.num_subpatterns - len(walk))
        self.num_transitions = max(0, self.num_transitions - max(0, len(walk) - 1))

        # Clear cached edges list
        self._edges_cache = None

        # Recalculate derived state
        self.recalculate()

    def _derive_stop_probability_data(self):
        """Compute MLE stop probabilities at terminal states.

        For each terminal state *t*, the stop probability is the Maximum
        Likelihood Estimator of a Bernoulli process:

            P(stop | t) = T(t) / (T(t) + f(t))

        where T(t) is the number of sequences that ended at *t*
        (``terminal_states[t]``) and f(t) is the number of outgoing edge
        traversals from *t* (``node_outgoing_counts[t]``).

        Properties:
        - Always in [0, 1] by construction (no clamping needed).
        - Preserves the Markov property (depends only on local node counts).
        - O(|T|) computation cost.
        """
        stop_probs = {}
        terminal_state_data = {}
        for state, t_count in self.terminal_state_counts.items():
            f_count = self.node_outgoing_counts.get(state, 0)
            denominator = t_count + f_count
            sp = t_count / denominator if denominator > 0 else 1.0
            stop_probs[state] = sp
            terminal_state_data[state] = {
                'terminal_count': t_count,
                'outgoing_count': f_count,
                'stop_probability': sp,
            }

        self.terminal_state_data = terminal_state_data
        self._stop_probability_cache = stop_probs

    def _length_specific_terminal_state(self, length):
        """
        Return all terminal states whose suffix (split by '_') equals the given `length`.
        """
        return [
            state for state in self.terminal_state_counts
            if int(state.rsplit('_', 1)[-1]) == length
        ]

    def eigenvector_centrality(self, max_iter=500):
        # Build a weight-attribute view for NetworkX algorithms
        for a, b in self.graph.edges:
            self.graph[a][b]['weight'] = self.graph[a][b]['data'].weight
        result = nx.algorithms.eigenvector_centrality(self.graph, weight='weight', max_iter=max_iter)
        # Clean up temporary attribute
        for a, b in self.graph.edges:
            if 'weight' in self.graph[a][b]:
                del self.graph[a][b]['weight']
        return result

    @property
    def isolates(self):
        """
        Return a list of isolate nodes (nodes with zero edges).
        """
        return list(nx.isolates(self.graph))

    def drop_isolates(self):
        """
        Remove isolates (nodes with zero edges) from the graph.
        """
        self.graph.remove_nodes_from(self.isolates)
        self._walk_cache = None
        self._topo_order = None

    @property
    def is_dag(self):
        """
        Check whether the graph is a Directed Acyclic Graph (DAG).
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def _get_topo_order(self):
        """Return cached topological order, rebuilding if needed.

        Raises:
            RuntimeError: If the graph contains cycles.
        """
        if self._topo_order is None:
            try:
                self._topo_order = list(nx.topological_sort(self.graph))
            except nx.NetworkXUnfeasible:
                raise RuntimeError(
                    "This operation requires a DAG-structured graph. "
                    "Use lzpgen_distribution() for graphs with cycles."
                )
        return self._topo_order

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def graph_summary(self):
        """
        Return a quick summary of the graph: Chromatic Number, Number of Isolates,
        Max In Deg, Max Out Deg, and Number of Edges.
        """
        return pd.Series({
            'Chromatic Number': max(nx.greedy_color(self.graph).values()) + 1,
            'Number of Isolates': nx.number_of_isolates(self.graph),
            'Max In Deg': max(dict(self.graph.in_degree).values(), default=0),
            'Max Out Deg': max(dict(self.graph.out_degree).values(), default=0),
            'Number of Edges': self.graph.number_of_edges(),
        })

    def voterank(self, n_nodes=25):
        """
        Use the VoteRank algorithm to return the top N most influential nodes in the graph.
        """
        return nx.algorithms.voterank(self.graph, number_of_nodes=n_nodes)

    # =========================================================================
    # Shared Analysis Methods
    # =========================================================================

    def walk_probability(self, walk, verbose=True, use_log=False):
        """
        Compute the probability (PGEN) of a walk on the graph.

        The model computes:
            P(seq) = P(init) × ∏ P(edge_i) × P(stop | last_node)

        where:
        - P(init) comes from ``initial_state_probabilities``
        - P(edge_i) are the conditional transition probabilities on each edge
        - P(stop | last_node) comes from ``terminal_state_data['stop_probability']``

        If ``self.impute_missing_edges`` is True and some (but not all) edges
        are missing, the missing edges are imputed using the geometric mean of
        the observed edge weights. If all edges are missing, returns 0 / -inf.

        Args:
            walk (list or str): A list of node identifiers or a raw sequence string.
            verbose (bool): Whether to log missing-edge warnings.
            use_log (bool): If True, return log-probability.

        Returns:
            float: Probability or log-probability of the walk.
        """
        if isinstance(walk, str):
            walk_ = self.encode_sequence(walk)
        else:
            walk_ = walk

        if len(walk_) == 0:
            return float('-inf') if use_log else 0.0

        # 1. Initial state probability
        first_node = walk_[0]
        if first_node not in self.initial_state_probabilities:
            if verbose:
                logger.debug(f"First node {first_node} not in initial_state_probabilities")
            return float('-inf') if use_log else 0.0

        initial_prob = self.initial_state_probabilities[first_node]

        # 2. Edge transition probabilities
        missing_count = 0
        observed_count = 0

        if use_log:
            log_initial = np.log(initial_prob)
            log_edge_sum = 0.0

            graph = self.graph  # local ref for speed
            for step1, step2 in window(walk_, 2):
                try:
                    log_edge_sum += np.log(graph[step1][step2]['data'].weight)
                    observed_count += 1
                except KeyError:
                    if verbose:
                        logger.debug(f"No edge connecting: {step1} --> {step2}")
                    missing_count += 1

            # Handle missing edges
            if missing_count > 0:
                if not self.impute_missing_edges or observed_count == 0:
                    return float('-inf')
                # Geometric mean imputation
                log_gmean = log_edge_sum / observed_count
                log_edge_sum += log_gmean * missing_count

            log_result = log_initial + log_edge_sum

            # 3. Terminal state factor
            last_node = walk_[-1]
            stop_prob = self._stop_probability_cache.get(last_node)
            if stop_prob is not None:
                log_result += np.log(max(stop_prob, _EPS))
            else:
                log_result += _LOG_EPS

            return log_result

        else:
            edge_product = 1.0

            graph = self.graph  # local ref for speed
            for step1, step2 in window(walk_, 2):
                try:
                    edge_product *= graph[step1][step2]['data'].weight
                    observed_count += 1
                except KeyError:
                    if verbose:
                        logger.debug(f"No edge connecting: {step1} --> {step2}")
                    missing_count += 1

            # Handle missing edges
            if missing_count > 0:
                if not self.impute_missing_edges or observed_count == 0:
                    return 0.0
                gmean = np.power(edge_product, 1.0 / observed_count)
                edge_product *= (gmean ** missing_count)

            result = initial_prob * edge_product

            # 3. Terminal state factor
            last_node = walk_[-1]
            stop_prob = self._stop_probability_cache.get(last_node)
            if stop_prob is not None:
                result *= max(stop_prob, _EPS)
            else:
                result *= _EPS

            return result

    def walk_log_probability(self, walk, verbose=True):
        """
        Convenience method to compute log-probability of a walk.
        Equivalent to walk_probability(walk, use_log=True).

        Recommended for long sequences to prevent numerical underflow.

        Args:
            walk (list or str): A list of node identifiers or a raw sequence string.
            verbose (bool): Whether to log missing-edge warnings.

        Returns:
            float: Log-probability of generating the walk.
        """
        return self.walk_probability(walk, verbose=verbose, use_log=True)

    def unsupervised_random_walk(self):
        """
        Perform a random walk from a random initial state until a stop condition
        is met. Returns both the walk (list of nodes) and the reconstructed
        sequence (cleaned node labels concatenated).

        Returns:
            tuple: (walk, sequence) where walk is a list of node names and
                sequence is the reconstructed string.
        """
        random_init = self._random_initial_state()
        current_state = random_init
        walk = [random_init]
        parts = [self.extract_subpattern(random_init)]

        while not self.is_stop_condition(current_state):
            current_state = self.random_step(current_state)
            walk.append(current_state)
            parts.append(self.extract_subpattern(current_state))

        return walk, ''.join(parts)

    def sequence_variation_curve(self, cdr3_sample):
        """
        Given a sequence, return the encoded subpatterns and the out-degree
        (number of possible transitions) at each position.

        Args:
            cdr3_sample (str): A sequence to analyze.

        Returns:
            tuple: (encoded_subpatterns, out_degrees) where both are lists.
        """
        encoded = self.encode_sequence(cdr3_sample)
        curve = [self.graph.out_degree(node) for node in encoded]
        return encoded, curve

    def walk_genes(self, walk, dropna=True, raise_error=True):
        """
        Given a walk (list of nodes), return a DataFrame of gene usage at each edge.

        Args:
            walk (list): The node path.
            dropna (bool): If True, drop genes with all-NaN across edges.
            raise_error (bool): If True and result is empty, raise an error.

        Returns:
            pd.DataFrame: Rows are gene names, columns are edges, with a 'type'
                and 'sum' column appended.
        """
        trans_genes = {}
        for i in range(len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                edge_attrs = self.graph[walk[i]][walk[i + 1]]['data'].gene_dict()
                trans_genes[f"{walk[i]}->{walk[i + 1]}"] = edge_attrs

        df = pd.DataFrame(trans_genes)
        if dropna:
            df.dropna(how="all", inplace=True)

        if df.empty and raise_error:
            raise GeneAnnotationError("No gene data found in the edges for the given walk.")

        if not df.empty:
            df["type"] = df.index.to_series().apply(
                lambda x: "V" if "v" in x.lower() else ("J" if "j" in x.lower() else "Unknown")
            )
            df["sum"] = df.sum(axis=1, numeric_only=True)

        return df

    def path_gene_table(self, cdr3_sample, threshold=None):
        """
        Return two tables (V genes, J genes) representing which genes could
        generate the given sequence. Genes missing from more than 'threshold'
        edges are dropped.

        Args:
            cdr3_sample (str): The sequence to examine.
            threshold (float, optional): NaN threshold. Defaults to length/4
                for V genes and length/2 for J genes.

        Returns:
            tuple: (vgene_table, jgene_table) as DataFrames.
        """
        encoded = self.encode_sequence(cdr3_sample)
        length = len(encoded)

        if threshold is None:
            threshold_v = length * 0.25
            threshold_j = length * 0.5
        else:
            threshold_v = threshold
            threshold_j = threshold

        gene_table = self.walk_genes(encoded, dropna=False, raise_error=False)
        na_counts = gene_table.isna().sum(axis=1)

        mask_v = na_counts < threshold_v
        vgene_table = gene_table[mask_v & gene_table.index.str.contains("V", case=False)]

        mask_j = na_counts < threshold_j
        jgene_table = gene_table[mask_j & gene_table.index.str.contains("J", case=False)]

        jgene_table = jgene_table.loc[jgene_table.isna().sum(axis=1).sort_values().index]
        vgene_table = vgene_table.loc[vgene_table.isna().sum(axis=1).sort_values().index]

        return vgene_table, jgene_table

    def gene_variation(self, cdr3):
        """
        Return a DataFrame showing how many V and J genes are possible at
        each subpattern position in the given sequence.

        Args:
            cdr3 (str): The sequence to analyze.

        Returns:
            pd.DataFrame: With columns 'genes', 'type', and 'sp'.

        Raises:
            NoGeneDataError: If the graph has no gene data.
        """
        if not self.has_gene_data:
            raise NoGeneDataError(
                operation="gene_variation",
                message="Cannot compute gene variation: this LZGraph has no gene data (genetic=False)."
            )

        encoded = self.encode_sequence(cdr3)

        n_v = [len(self.marginal_v_genes)]
        n_j = [len(self.marginal_j_genes)]

        for node in encoded[1:]:
            in_edges = self.graph.in_edges(node)
            v_candidates = set()
            j_candidates = set()
            for ea, eb in in_edges:
                ed = self.graph[ea][eb]['data']
                v_candidates |= set(ed.v_genes.keys())
                j_candidates |= set(ed.j_genes.keys())

            n_v.append(len(v_candidates))
            n_j.append(len(j_candidates))

        lz_subpatterns = lempel_ziv_decomposition(cdr3)
        df = pd.DataFrame({
            "genes": n_v + n_j,
            "type": (["V"] * len(n_v)) + (["J"] * len(n_j)),
            "sp": lz_subpatterns + lz_subpatterns,
        })
        return df

    # =========================================================================
    # Fast Simulation
    # =========================================================================

    def _build_walk_cache(self, seed=None):
        """Build pre-computed numpy arrays for fast random walks.

        Returns a dict with the cache data, stored as ``self._walk_cache``.
        """
        graph = self.graph
        nodes = list(graph.nodes())
        n = len(nodes)

        node_to_id = {name: i for i, name in enumerate(nodes)}
        id_to_node = np.array(nodes, dtype=object)

        # Pre-compute clean labels for all nodes
        clean_labels = np.array([self.extract_subpattern(name) for name in nodes], dtype=object)

        # Per-node neighbor IDs and weights
        neighbor_ids = [None] * n
        neighbor_weights = [None] * n
        for i, name in enumerate(nodes):
            succs = list(graph.successors(name))
            if succs:
                ids = np.array([node_to_id[s] for s in succs], dtype=np.intp)
                wts = np.array([graph[name][s]['data'].weight for s in succs], dtype=np.float64)
                wts /= wts.sum()  # ensure normalization
                neighbor_ids[i] = ids
                neighbor_weights[i] = wts

        # Stop probabilities: NaN for non-terminal nodes
        stop_probs = np.full(n, np.nan, dtype=np.float64)
        for state, prob in self._stop_probability_cache.items():
            if state in node_to_id:
                stop_probs[node_to_id[state]] = prob

        # Initial state arrays
        init_states = list(self.initial_state_probabilities.keys())
        init_probs = np.array(
            [self.initial_state_probabilities[s] for s in init_states],
            dtype=np.float64,
        )
        init_probs = init_probs / init_probs.sum()  # ensure normalization
        initial_ids = np.array([node_to_id[s] for s in init_states], dtype=np.intp)

        rng = np.random.default_rng(seed)

        self._walk_cache = {
            'node_to_id': node_to_id,
            'id_to_node': id_to_node,
            'clean_labels': clean_labels,
            'neighbor_ids': neighbor_ids,
            'neighbor_weights': neighbor_weights,
            'stop_probs': stop_probs,
            'initial_ids': initial_ids,
            'initial_probs': init_probs,
            'rng': rng,
        }
        return self._walk_cache

    def simulate(self, n, seed=None, return_walks=False):
        """Generate *n* sequences via optimized random walks.

        Uses a pre-computed walk cache for maximum throughput.
        The cache is built lazily on first call and invalidated when the
        graph is modified (``recalculate()``, ``drop_isolates()``).

        Args:
            n (int): Number of sequences to generate.
            seed (int, optional): RNG seed for reproducibility. If given,
                the cache is rebuilt with this seed.
            return_walks (bool): If True, return ``(walk, sequence)`` tuples
                instead of plain strings.

        Returns:
            list[str] or list[tuple[list[str], str]]: Generated sequences,
                or ``(walk, sequence)`` pairs if *return_walks* is True.
        """
        # Build or rebuild cache
        if self._walk_cache is None or seed is not None:
            self._build_walk_cache(seed)

        cache = self._walk_cache
        rng = cache['rng']
        initial_ids = cache['initial_ids']
        initial_probs = cache['initial_probs']
        stop_probs = cache['stop_probs']
        neighbor_ids = cache['neighbor_ids']
        neighbor_weights = cache['neighbor_weights']
        clean_labels = cache['clean_labels']
        id_to_node = cache['id_to_node']

        results = []
        for _ in range(n):
            # Pick initial state
            current = rng.choice(initial_ids, p=initial_probs)
            parts = [clean_labels[current]]
            walk_ids = [current] if return_walks else None

            while True:
                # Check stop condition
                stop_p = stop_probs[current]
                if not np.isnan(stop_p):
                    if rng.random() < stop_p:
                        break

                # Check for dead-end (no outgoing edges)
                nb_ids = neighbor_ids[current]
                if nb_ids is None:
                    break

                # Take a step
                current = rng.choice(nb_ids, p=neighbor_weights[current])
                parts.append(clean_labels[current])
                if return_walks:
                    walk_ids.append(current)

            sequence = ''.join(parts)
            if return_walks:
                walk = [id_to_node[wid] for wid in walk_ids]
                results.append((walk, sequence))
            else:
                results.append(sequence)

        return results

    # =========================================================================
    # LZPgen Distribution Methods
    # =========================================================================

    def lzpgen_distribution(self, n=100_000, seed=None):
        """Compute the empirical LZPgen distribution via Monte Carlo.

        Generates *n* random walks from the graph's generative model and
        computes the log-probability (LZPgen) for each walk simultaneously.
        This gives the probability-weighted distribution of generation
        probabilities *directly from the graph structure*, without needing
        the original input sequences.

        The log-probability for each walk is computed as::

            log P(init) + sum(log P(edge_i)) + log P(stop | last_node)

        consistent with :meth:`walk_log_probability`.

        Args:
            n (int): Number of sequences to sample.  Default 100,000.
            seed (int, optional): RNG seed for reproducibility.

        Returns:
            numpy.ndarray: Array of shape ``(n,)`` with log-probability values.

        Example::

            log_probs = graph.lzpgen_distribution(n=50_000, seed=42)
            plt.hist(log_probs, bins=100, density=True)
            plt.xlabel('log P(seq)')
            plt.ylabel('Density')
        """
        if n <= 0:
            return np.empty(0, dtype=np.float64)

        # Build / rebuild cache
        if self._walk_cache is None or seed is not None:
            self._build_walk_cache(seed)

        cache = self._walk_cache
        rng = cache['rng']
        initial_ids = cache['initial_ids']
        initial_probs = cache['initial_probs']
        stop_probs = cache['stop_probs']
        neighbor_ids = cache['neighbor_ids']
        neighbor_weights = cache['neighbor_weights']

        # Pre-compute log values for zero per-step overhead
        eps = _EPS
        initial_log_probs = np.log(np.maximum(initial_probs, eps))

        n_nodes = len(cache['id_to_node'])
        stop_log_probs = np.where(
            np.isnan(stop_probs), np.nan,
            np.log(np.maximum(stop_probs, eps))
        )

        neighbor_log_weights = [None] * n_nodes
        for i in range(n_nodes):
            if neighbor_weights[i] is not None:
                neighbor_log_weights[i] = np.log(
                    np.maximum(neighbor_weights[i], eps)
                )

        log_probs = np.empty(n, dtype=np.float64)
        n_initial = len(initial_ids)

        for seq_idx in range(n):
            # Pick initial state
            init_idx = rng.choice(n_initial, p=initial_probs)
            current = initial_ids[init_idx]
            log_p = initial_log_probs[init_idx]

            while True:
                # Check stop condition
                stop_p = stop_probs[current]
                if not np.isnan(stop_p):
                    if rng.random() < stop_p:
                        log_p += stop_log_probs[current]
                        break

                # Dead-end check
                nb_ids = neighbor_ids[current]
                if nb_ids is None:
                    log_p += np.log(eps)
                    break

                # Take a step
                n_nb = len(nb_ids)
                step_idx = rng.choice(n_nb, p=neighbor_weights[current])
                log_p += neighbor_log_weights[current][step_idx]
                current = nb_ids[step_idx]

            log_probs[seq_idx] = log_p

        return log_probs

    def lzpgen_moments(self):
        """Compute exact moments of the LZPgen distribution via forward propagation.

        For DAG-structured graphs (AAPLZGraph, NDPLZGraph), this computes the
        *exact* mean and variance of the log-probability distribution by
        propagating probability mass and moment accumulators through the graph
        in topological order.  This runs in O(|E|) time and gives deterministic
        results with no sampling required.

        The computed moments correspond to the distribution of
        :meth:`walk_log_probability` values over all possible walks, weighted
        by their generative probability.

        The forward propagation tracks five quantities per node:

        - **m0**: total probability mass arriving at the node
        - **m1**: first moment (mass-weighted sum of log-probabilities)
        - **m2**: second moment (mass-weighted sum of squared log-probabilities)
        - **m3**: third moment (for skewness)
        - **m4**: fourth moment (for kurtosis)

        At each terminal node, the stopping mass contributes to the overall
        distribution moments.  Continue-mass propagates forward via edge
        transitions.

        Returns:
            dict: Keys ``'mean'``, ``'variance'``, ``'std'``, ``'skewness'``,
                ``'kurtosis'``, ``'total_mass'``.
                ``total_mass`` should be close to 1.0 for a well-formed model.

        Raises:
            RuntimeError: If the graph contains cycles (use
                :meth:`lzpgen_distribution` for cyclic graphs instead).

        Example::

            moments = graph.lzpgen_moments()
            print(f"E[log P] = {moments['mean']:.4f}")
            print(f"Std[log P] = {moments['std']:.4f}")
        """
        topo_order = self._get_topo_order()
        eps = _EPS
        n_nodes = len(topo_order)

        # Moment accumulators per node (indexed by topological position)
        m0 = np.zeros(n_nodes)
        m1 = np.zeros(n_nodes)
        m2 = np.zeros(n_nodes)
        m3 = np.zeros(n_nodes)
        m4 = np.zeros(n_nodes)

        node_to_idx = {node: i for i, node in enumerate(topo_order)}

        # Seed initial states
        for state, p in self.initial_state_probabilities.items():
            if state in node_to_idx:
                idx = node_to_idx[state]
                lp = np.log(max(p, eps))
                lp2 = lp * lp
                m0[idx] += p
                m1[idx] += p * lp
                m2[idx] += p * lp2
                m3[idx] += p * lp2 * lp
                m4[idx] += p * lp2 * lp2

        # Terminal accumulators
        T0 = T1 = T2 = T3 = T4 = 0.0

        graph = self.graph

        for u in topo_order:
            ui = node_to_idx[u]
            if m0[ui] == 0:
                continue

            # Terminal contribution: mass that stops here
            stop_prob = self._stop_probability_cache.get(u)
            if stop_prob is not None and stop_prob > 0:
                ls = np.log(max(stop_prob, eps))
                ls2 = ls * ls
                ls3 = ls2 * ls
                ls4 = ls3 * ls

                T0 += stop_prob * m0[ui]
                T1 += stop_prob * (m1[ui] + ls * m0[ui])
                T2 += stop_prob * (m2[ui] + 2 * ls * m1[ui] + ls2 * m0[ui])
                T3 += stop_prob * (
                    m3[ui] + 3 * ls * m2[ui]
                    + 3 * ls2 * m1[ui] + ls3 * m0[ui]
                )
                T4 += stop_prob * (
                    m4[ui] + 4 * ls * m3[ui] + 6 * ls2 * m2[ui]
                    + 4 * ls3 * m1[ui] + ls4 * m0[ui]
                )
                continue_factor = 1.0 - stop_prob
            else:
                continue_factor = 1.0

            if continue_factor < eps:
                continue

            # Propagate to successors (binomial moment expansion)
            for v in graph.successors(u):
                vi = node_to_idx[v]
                w = graph[u][v]['data'].weight
                lw = np.log(max(w, eps))
                lw2 = lw * lw
                lw3 = lw2 * lw
                lw4 = lw3 * lw
                f = continue_factor * w

                m0[vi] += f * m0[ui]
                m1[vi] += f * (m1[ui] + lw * m0[ui])
                m2[vi] += f * (m2[ui] + 2 * lw * m1[ui] + lw2 * m0[ui])
                m3[vi] += f * (
                    m3[ui] + 3 * lw * m2[ui]
                    + 3 * lw2 * m1[ui] + lw3 * m0[ui]
                )
                m4[vi] += f * (
                    m4[ui] + 4 * lw * m3[ui] + 6 * lw2 * m2[ui]
                    + 4 * lw3 * m1[ui] + lw4 * m0[ui]
                )

        if T0 < eps:
            raise RuntimeError("No probability mass reached terminal states")

        # Raw moments about origin
        mu1 = T1 / T0
        mu2 = T2 / T0
        mu3 = T3 / T0
        mu4 = T4 / T0

        # Cumulants
        k1 = mu1
        k2 = mu2 - mu1 ** 2
        k3 = mu3 - 3 * mu2 * mu1 + 2 * mu1 ** 3
        k4 = (mu4 - 4 * mu3 * mu1 - 3 * mu2 ** 2
              + 12 * mu2 * mu1 ** 2 - 6 * mu1 ** 4)

        variance = max(k2, 0.0)
        std = np.sqrt(variance)

        return {
            'mean': float(k1),
            'variance': float(variance),
            'std': float(std),
            'skewness': float(k3 / k2 ** 1.5) if k2 > 0 else 0.0,
            'kurtosis': float(k4 / k2 ** 2) if k2 > 0 else 0.0,
            'total_mass': float(T0),
        }

    def lzpgen_analytical_distribution(self):
        """Derive a full analytical LZPgen distribution from graph structure.

        Combines two complementary techniques:

        1. **Length-conditional Gaussian mixture** — one Normal component per
           walk length, with exact per-length (weight, mean, variance) computed
           by length-stratified forward propagation.
        2. **Saddlepoint approximation** — exact kappa_1 … kappa_4 computed by
           extending the moment propagation to m3 and m4, enabling the
           Lugannani-Rice PDF/CDF.

        All parameters are derived deterministically from the graph's adjacency
        matrix, edge weights, and stop probabilities.  No Monte Carlo sampling.

        Time complexity: O(|E| * K_max) where K_max is the maximum walk length
        (typically 5–25).

        Returns:
            LZPgenDistribution: A scipy-like distribution object with
                ``.pdf()``, ``.cdf()``, ``.ppf()``, ``.rvs()``,
                ``.confidence_interval()``, ``.saddlepoint_pdf()``,
                ``.saddlepoint_cdf()``, ``.mean()``, ``.var()``,
                ``.skewness()``, ``.kurtosis()``, and ``.summary()``.

        Raises:
            RuntimeError: If the graph contains cycles.

        Example::

            dist = graph.lzpgen_analytical_distribution()
            print(dist.summary())

            x = np.linspace(-35, -10, 500)
            plt.plot(x, dist.pdf(x), label='Gaussian mixture')
            plt.plot(x, dist.saddlepoint_pdf(x), '--', label='Saddlepoint')
        """
        from ..metrics.pgen_distribution import LZPgenDistribution

        topo_order = self._get_topo_order()
        eps = _EPS
        n_nodes = len(topo_order)
        node_to_idx = {node: i for i, node in enumerate(topo_order)}

        # ── Accumulators: m0…m4 per node ──
        m0 = np.zeros(n_nodes)
        m1 = np.zeros(n_nodes)
        m2 = np.zeros(n_nodes)
        m3 = np.zeros(n_nodes)
        m4 = np.zeros(n_nodes)

        # ── Depth per node (= walk edge-count from any initial state) ──
        depth = np.full(n_nodes, -1, dtype=np.int32)

        # Seed initial states
        for state, p in self.initial_state_probabilities.items():
            if state in node_to_idx:
                idx = node_to_idx[state]
                lp = np.log(max(p, eps))
                lp2 = lp * lp
                lp3 = lp2 * lp
                lp4 = lp3 * lp
                m0[idx] += p
                m1[idx] += p * lp
                m2[idx] += p * lp2
                m3[idx] += p * lp3
                m4[idx] += p * lp4
                depth[idx] = 0

        # ── Per-length terminal accumulators ──
        # {walk_length: [tm0, tm1, tm2]}
        per_length = {}
        # Global terminal accumulators for cumulants
        T0 = T1 = T2 = T3 = T4 = 0.0

        graph = self.graph

        for u in topo_order:
            ui = node_to_idx[u]
            if m0[ui] == 0:
                continue

            # Terminal contribution
            stop_prob = self._stop_probability_cache.get(u)
            if stop_prob is not None and stop_prob > 0:
                ls = np.log(max(stop_prob, eps))
                ls2 = ls * ls
                ls3 = ls2 * ls
                ls4 = ls3 * ls

                sm0 = stop_prob * m0[ui]
                sm1 = stop_prob * (m1[ui] + ls * m0[ui])
                sm2 = stop_prob * (m2[ui] + 2 * ls * m1[ui] + ls2 * m0[ui])
                sm3 = stop_prob * (
                    m3[ui] + 3 * ls * m2[ui]
                    + 3 * ls2 * m1[ui] + ls3 * m0[ui]
                )
                sm4 = stop_prob * (
                    m4[ui] + 4 * ls * m3[ui] + 6 * ls2 * m2[ui]
                    + 4 * ls3 * m1[ui] + ls4 * m0[ui]
                )

                T0 += sm0; T1 += sm1; T2 += sm2; T3 += sm3; T4 += sm4

                # Per-length accumulation
                d = int(depth[ui])
                if d not in per_length:
                    per_length[d] = [0.0, 0.0, 0.0]
                per_length[d][0] += sm0
                per_length[d][1] += sm1
                per_length[d][2] += sm2

                continue_factor = 1.0 - stop_prob
            else:
                continue_factor = 1.0

            if continue_factor < eps:
                continue

            # Propagate to successors
            for v in graph.successors(u):
                vi = node_to_idx[v]
                w = graph[u][v]['data'].weight
                lw = np.log(max(w, eps))
                lw2 = lw * lw
                lw3 = lw2 * lw
                lw4 = lw3 * lw
                f = continue_factor * w

                m0[vi] += f * m0[ui]
                m1[vi] += f * (m1[ui] + lw * m0[ui])
                m2[vi] += f * (m2[ui] + 2 * lw * m1[ui] + lw2 * m0[ui])
                m3[vi] += f * (
                    m3[ui] + 3 * lw * m2[ui]
                    + 3 * lw2 * m1[ui] + lw3 * m0[ui]
                )
                m4[vi] += f * (
                    m4[ui] + 4 * lw * m3[ui] + 6 * lw2 * m2[ui]
                    + 4 * lw3 * m1[ui] + lw4 * m0[ui]
                )

                # Propagate depth
                if depth[vi] == -1:
                    depth[vi] = depth[ui] + 1

        if T0 < eps:
            raise RuntimeError("No probability mass reached terminal states")

        # ── Build cumulants from global moments ──
        mu1 = T1 / T0
        mu2 = T2 / T0
        mu3 = T3 / T0
        mu4 = T4 / T0

        k1 = mu1
        k2 = mu2 - mu1 ** 2
        k3 = mu3 - 3 * mu2 * mu1 + 2 * mu1 ** 3
        k4 = mu4 - 4 * mu3 * mu1 - 3 * mu2 ** 2 + 12 * mu2 * mu1 ** 2 - 6 * mu1 ** 4

        cumulants = {
            'kappa_1': float(k1),
            'kappa_2': float(max(k2, 0.0)),
            'kappa_3': float(k3),
            'kappa_4': float(k4),
            'skewness': float(k3 / k2 ** 1.5) if k2 > 0 else 0.0,
            'kurtosis': float(k4 / k2 ** 2) if k2 > 0 else 0.0,
            'total_mass': float(T0),
        }

        # ── Build Gaussian mixture from per-length terminal data ──
        sorted_lengths = sorted(per_length.keys())
        weights = []
        means = []
        stds = []
        walk_lengths = []

        for d in sorted_lengths:
            pm0, pm1, pm2 = per_length[d]
            if pm0 < 1e-12:
                continue
            wt = pm0 / T0
            mu_d = pm1 / pm0
            var_d = pm2 / pm0 - mu_d ** 2
            weights.append(wt)
            means.append(mu_d)
            stds.append(np.sqrt(max(var_d, 0.0)))
            walk_lengths.append(d)

        return LZPgenDistribution(
            weights=weights,
            means=means,
            stds=stds,
            walk_lengths=walk_lengths,
            cumulants=cumulants,
        )

    # =========================================================================
    # Serialization Methods
    # =========================================================================

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

    # =========================================================================
    # AIRR Format Support
    # =========================================================================

    # Default column mapping from AIRR standard to LZGraphs internal names
    _AIRR_COLUMN_MAP = {
        'junction_aa': 'cdr3_amino_acid',
        'junction': 'cdr3_rearrangement',
        'v_call': 'V',
        'j_call': 'J',
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
            data (pd.DataFrame or str): Either a DataFrame with AIRR column names,
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
        # Load from file if path provided
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data, sep='\t')

        # Build effective column map
        effective_map = dict(cls._AIRR_COLUMN_MAP)
        if column_map:
            effective_map.update(column_map)

        # Rename columns that exist in the DataFrame
        rename_map = {
            airr_col: lzg_col
            for airr_col, lzg_col in effective_map.items()
            if airr_col in data.columns and lzg_col not in data.columns
        }
        if rename_map:
            data = data.rename(columns=rename_map)

        return cls(data, **kwargs)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def batch_walk_probability(self, sequences, use_log=False, verbose=False):
        """
        Compute walk probability for a batch of sequences.

        This is more efficient than calling walk_probability in a loop
        as it collects all results into a NumPy array.

        Args:
            sequences (list): List of sequences (strings or pre-encoded walks).
            use_log (bool): If True, return log-probabilities. Recommended for
                long sequences to prevent numerical underflow.
            verbose (bool): If True, show a progress bar.

        Returns:
            np.ndarray: Array of (log-)probabilities, one per sequence.

        Example:
            >>> graph = AAPLZGraph(data)
            >>> seqs = ["CASSLGIRRTNTEAFF", "CASSLEGKYEQYF"]
            >>> probs = graph.batch_walk_probability(seqs)
            >>> log_probs = graph.batch_walk_probability(seqs, use_log=True)
        """
        from tqdm.auto import tqdm as tqdm_auto

        results = np.empty(len(sequences), dtype=np.float64)
        iterator = enumerate(sequences)
        if verbose:
            iterator = tqdm_auto(iterator, total=len(sequences),
                                 desc="Computing walk probabilities", leave=False)

        for i, seq in iterator:
            results[i] = self.walk_probability(seq, verbose=False, use_log=use_log)

        return results
