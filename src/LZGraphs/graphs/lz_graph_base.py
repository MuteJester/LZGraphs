import logging
import re
from abc import ABC, abstractmethod
from time import time

import networkx as nx
import numpy as np

# Utility functions
from ..utilities.misc import choice, window

# Shared constants
from ..constants import _EPS, _LOG_EPS

# EdgeData
from .edge_data import EdgeData

# Mixins
from ..mixins import (
    GeneLogicMixin,
    RandomWalkMixin,
    GenePredictionMixin,
    GraphTopologyMixin,
    LZPgenDistributionMixin,
    WalkAnalysisMixin,
    BayesianPosteriorMixin,
    SerializationMixin,
)

# Custom exceptions
from ..exceptions import MissingColumnError

# Create a logger for this module
logger = logging.getLogger(__name__)


def _dicts_close(a, b, decimals=3):
    """Compare two dicts with rounding tolerance.

    Works on both plain dicts and Series-like objects (for backward compat).
    """
    # Convert Series-like to dict if needed (backward compat with old pickles)
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


class LZGraphBase(
    ABC,
    GeneLogicMixin,
    RandomWalkMixin,
    GenePredictionMixin,
    GraphTopologyMixin,
    LZPgenDistributionMixin,
    WalkAnalysisMixin,
    BayesianPosteriorMixin,
    SerializationMixin,
):
    """
    Abstract base class for LZGraph implementations.

    Concrete subclasses: AAPLZGraph, NDPLZGraph.
    Functionality is organised into focused mixins — see ``src/LZGraphs/mixins/``.
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

    @staticmethod
    def _normalize_input(data, seq_column, abundances=None, v_genes=None, j_genes=None):
        """Convert flexible input to a standardised dict-of-lists.

        Accepts a DataFrame-like object (anything with a ``.columns``
        attribute), a list of strings, or an iterable with a ``.tolist()``
        method (e.g. pandas Series).

        Returns a dict with keys:
            ``'sequences'``, ``'abundances'`` (list or None),
            ``'v_genes'`` (list or None), ``'j_genes'`` (list or None).
        """
        # ── DataFrame-like (has .columns) ──
        if hasattr(data, 'columns'):
            if abundances is not None or v_genes is not None or j_genes is not None:
                raise TypeError(
                    "Cannot pass abundances/v_genes/j_genes when data is "
                    "already a DataFrame — use DataFrame columns instead"
                )
            if seq_column not in data.columns:
                raise MissingColumnError(
                    column_name=seq_column,
                    available_columns=list(data.columns),
                )
            result = {'sequences': list(data[seq_column])}
            result['abundances'] = (
                list(data['abundance']) if 'abundance' in data.columns else None
            )
            has_genes = 'V' in data.columns and 'J' in data.columns
            result['v_genes'] = list(data['V']) if has_genes else None
            result['j_genes'] = list(data['J']) if has_genes else None
            return result

        # ── Series-like (has .tolist but no .columns) ──
        if hasattr(data, 'tolist') and not hasattr(data, 'columns'):
            sequences = data.tolist()
        elif isinstance(data, (list, tuple)):
            sequences = list(data)
        else:
            raise TypeError(
                f"data must be a DataFrame, list, or Series, "
                f"got {type(data).__name__}"
            )

        n = len(sequences)

        if abundances is not None:
            abundances = list(abundances)
            if len(abundances) != n:
                raise ValueError(
                    f"abundances length ({len(abundances)}) != "
                    f"sequences length ({n})"
                )

        if v_genes is not None or j_genes is not None:
            if v_genes is None or j_genes is None:
                raise ValueError(
                    "v_genes and j_genes must both be provided or both omitted"
                )
            v_genes = list(v_genes)
            j_genes = list(j_genes)
            if len(v_genes) != n or len(j_genes) != n:
                raise ValueError(
                    f"v_genes/j_genes length must match sequences length ({n})"
                )

        return {
            'sequences': sequences,
            'abundances': abundances,
            'v_genes': v_genes,
            'j_genes': j_genes,
        }

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

    def _simultaneous_graph_construction(self, data: dict):
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

    def _get_node_feature_info(self, node_a, feature, V=None, J=None, asdict=False):
        """
        Return edge metadata for outgoing edges from *node_a*.

        If *V* and *J* are given, only edges containing both genes are
        included, and only the requested *feature* is returned per edge.

        Always returns a dict (the *asdict* parameter is kept for
        backward compatibility but has no effect).
        """
        node_data = self.graph[node_a]
        if V is None or J is None:
            return {
                nb: self.graph[node_a][nb]['data'].to_legacy_dict()
                for nb in node_data
            }

        # Filter edges to those containing V and J
        partial_dict = {}
        for nb in node_data:
            ed = self.graph[node_a][nb]['data']
            if ed.has_gene(V) and ed.has_gene(J):
                partial_dict[nb] = {feature: ed.weight if feature == 'weight' else ed.to_legacy_dict().get(feature)}
        return partial_dict

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

    def _log_step(self, message, verbose):
        """Log a construction progress message with elapsed time."""
        if verbose:
            elapsed = time() - self.constructor_start_time
            logger.info(f"[{elapsed:.2f}s] {message}")

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

    def _get_state_weights(self, node):
        """
        Return all possible next-states (edges) from `node` and their respective weights.
        """
        node_data = self.graph[node]
        states = list(node_data.keys())
        probabilities = [node_data[s]['data'].weight for s in states]
        return states, probabilities

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
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def graph_summary(self):
        """
        Return a quick summary of the graph: Chromatic Number, Number of Isolates,
        Max In Deg, Max Out Deg, and Number of Edges.

        Returns:
            dict with string keys and integer values.
        """
        return {
            'Chromatic Number': max(nx.greedy_color(self.graph).values()) + 1,
            'Number of Isolates': nx.number_of_isolates(self.graph),
            'Max In Deg': max(dict(self.graph.in_degree).values(), default=0),
            'Max Out Deg': max(dict(self.graph.out_degree).values(), default=0),
            'Number of Edges': self.graph.number_of_edges(),
        }

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
