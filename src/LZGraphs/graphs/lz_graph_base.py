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
        self.genetic = False
        self.genetic_walks_black_list = {}

        # sub-pattern count, transitions, etc.
        self.n_subpatterns = 0
        self.n_transitions = 0

        self.initial_states, self.terminal_states = dict(), dict()
        self.initial_states_probability = pd.Series(dtype=np.float64)
        self.lengths = dict()
        self.cac_graphs = dict()
        self.n_neighbours = dict()
        self.per_node_observed_frequency = dict()

        self.length_distribution_proba = pd.Series(dtype=np.float64)
        self.subpattern_individual_probability = pd.Series(dtype=np.float64)

        # PGEN configuration (overridden by subclasses)
        self.impute_missing_edges = False
        self.smoothing_alpha = 0.0
        self.initial_state_threshold = 0

    def __eq__(self, other):
        """
        This method tests whether two LZGraphs are equal, i.e. have the same node, edges,
        and metadata on the edges.
        """
        # Check if graphs have same structure
        if not nx.utils.graphs_equal(self.graph, other.graph):
            return False

        # Check if both have same genetic status
        if self.genetic != other.genetic:
            return False

        aux = 0
        aux += self.genetic_walks_black_list != other.genetic_walks_black_list
        aux += self.n_subpatterns != other.n_subpatterns

        # Compare initial_states, terminal_states, etc.
        aux += not self.initial_states.round(3).equals(other.initial_states.round(3))
        aux += not self.terminal_states.round(3).equals(other.terminal_states.round(3))
        aux += not other.length_distribution_proba.round(3).equals(self.length_distribution_proba.round(3))

        # Compare gene-related distributions only if both are genetic
        if self.genetic and other.genetic:
            aux += not other.marginal_vgenes.round(3).equals(self.marginal_vgenes.round(3))
            aux += not other.vj_probabilities.round(3).equals(self.vj_probabilities.round(3))
            aux += not other.length_distribution.round(3).equals(self.length_distribution.round(3))

        return (aux == 0)

    def __repr__(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        genetic_str = "genetic" if self.genetic else "non-genetic"
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
    def clean_node(base):
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
        to insert node/edge data into the networkx DiGraph. If `self.genetic` is True,
        we also embed gene information.
        """
        processing_stream = self._decomposed_sequence_generator(data)
        if self.genetic:
            for output in processing_stream:
                steps, locations, v, j = output
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information(A_, B_, v, j)
                # ensure final node exists in frequency dict
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)
        else:
            for output in processing_stream:
                steps, locations = output
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information_no_genes(A_, B_)
                # ensure final node exists in frequency dict
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)

    def _insert_edge_and_information_no_genes(self, node_a, node_b):
        """
        Insert or update an edge (node_a -> node_b) with no gene info, incrementing count by 1.
        """
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]['data'].record()
        else:
            ed = EdgeData()
            ed.record()
            self.graph.add_edge(node_a, node_b, data=ed)
        self.n_transitions += 1

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
                total = self.per_node_observed_frequency.get(edge_a, 0)
                ed.normalize(total)
        else:
            for node in self.graph.nodes():
                successors = list(self.graph.successors(node))
                if not successors:
                    continue
                k = len(successors)
                raw_total = self.per_node_observed_frequency.get(node, 0)
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
        If `self.genetic` is True, we check V/J constraints plus
        a random stop probability.
        """
        if state not in self.terminal_states:
            return False

        if self.genetic:
            if selected_j is not None:
                # Check whether we have neighbors that contain both selected_v and selected_j
                neighbours = 0
                for nb in self.graph[state]:
                    ed = self.graph[state][nb]['data']
                    if ed.has_gene(selected_v) and ed.has_gene(selected_j):
                        neighbours = 2
                        break
            else:
                neighbours = self.graph.out_degree(state)

            if neighbours == 0:
                return True
            else:
                stop_probability = self.terminal_state_data.loc[state, 'wsif/sep']
                return (np.random.binomial(1, stop_probability) == 1)
        else:
            stop_probability = self.terminal_state_data.loc[state, 'wsif/sep']
            return (np.random.binomial(1, stop_probability) == 1)

    def _derive_subpattern_individual_probability(self):
        """
        Summation of edge counts by source node -> yields empirical probabilities for each node.
        Uses raw counts (not normalized weights) so this can be called before or after normalization.
        """
        counts = {(a, b): self.graph[a][b]['data'].count for a, b in self.graph.edges}
        count_df = pd.Series(counts).reset_index()
        self.subpattern_individual_probability = count_df.groupby('level_0').sum(numeric_only=True).rename(columns={0: 'proba'})
        self.subpattern_individual_probability.proba /= self.subpattern_individual_probability.proba.sum()

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
        elif message_number == 4:
            logger.info(f"[{elapsed:.2f}s] Graph Edge Gene Weights Normalized.")
        elif message_number == 5:
            logger.info(f"[{elapsed:.2f}s] Terminal State Map Derived.")
        elif message_number == 6:
            # We assume there's a self.constructor_end_time set somewhere
            total_time = round(self.constructor_end_time - self.constructor_start_time, 2)
            logger.info(f"[{total_time:.2f}s] LZGraph Created Successfully.")
        elif message_number == 7:
            logger.info(f"[{elapsed:.2f}s] Terminal State Map Derived (Event 7).")
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
        return choice(self.initial_states_probability.index, self.initial_states_probability.values)

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

    def _update_terminal_states(self, terminal_state):
        self.terminal_states[terminal_state] = self.terminal_states.get(terminal_state, 0) + 1

    def _update_initial_states(self, initial_state):
        self.initial_states[initial_state] = self.initial_states.get(initial_state, 0) + 1

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
        # Recompute per_node_observed_frequency from actual edge counts.
        # freq[node] = sum of outgoing edge counts (matching construction convention).
        self.per_node_observed_frequency = {}
        for node in self.graph.nodes():
            outgoing_sum = sum(
                self.graph[node][succ]['data'].count
                for succ in self.graph.successors(node)
            )
            if outgoing_sum > 0:
                self.per_node_observed_frequency[node] = outgoing_sum

        # Initial state probabilities
        if isinstance(self.initial_states, pd.Series) and len(self.initial_states) > 0:
            total = self.initial_states.sum()
            if total > 0:
                self.initial_states_probability = self.initial_states / total

        # Length distribution probabilities
        if isinstance(self.terminal_states, pd.Series) and len(self.terminal_states) > 0:
            total = self.terminal_states.sum()
            if total > 0:
                self.length_distribution_proba = self.terminal_states / total

        # Normalize edge weights
        self._normalize_edge_weights()

        # Derived probability tables
        self._derive_subpattern_individual_probability()
        self._derive_terminal_state_map()
        self._derive_stop_probability_data()

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
        if first_node in self.initial_states.index:
            self.initial_states[first_node] = max(0, self.initial_states[first_node] - 1)
            if self.initial_states[first_node] <= 0:
                self.initial_states = self.initial_states.drop(first_node)
        if last_node in self.terminal_states.index:
            self.terminal_states[last_node] = max(0, self.terminal_states[last_node] - 1)
            if self.terminal_states[last_node] <= 0:
                self.terminal_states = self.terminal_states.drop(last_node)

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

        self.n_subpatterns = max(0, self.n_subpatterns - len(walk))
        self.n_transitions = max(0, self.n_transitions - max(0, len(walk) - 1))

        # Clear cached edges list
        self.edges_list = None

        # Recalculate derived state
        self.recalculate()

    def _derive_terminal_state_map(self):
        """
        For each terminal state, run a DFS from it to find other terminal
        states reachable from that node. Store the results in `self.terminal_state_map`.
        """
        terminal_states_set = set(self.terminal_states.index)
        reachability_dict = {}

        for terminal in self.terminal_states.index:
            visited = nx.dfs_preorder_nodes(self.graph, source=terminal)
            reachable_terminals = list(set(visited) & terminal_states_set)
            reachability_dict[terminal] = reachable_terminals

        self.terminal_state_map = pd.Series(reachability_dict, index=self.terminal_states.index)

    def _derive_stop_probability_data(self):
        """
        Example logic for computing stop probabilities at terminal states.
        (This logic might be specialized, but it's carried over from your original class.)
        """

        def freq_normalize(target):
            D = self.length_distribution_proba.loc[target].copy()
            D /= D.sum()
            return D

        def wont_stop_at_future_states(state, es):
            D = freq_normalize(es.decendent_end_states[state])
            current_freq = D.pop(state)
            if len(D) >= 1:
                D = 1 - D
                return D.product()
            else:
                return 1

        def didnt_stop_at_past(state, es):
            D = freq_normalize(es.ancestor_end_state[state])
            if state in D:
                D.pop(state)
            if len(D) >= 1:
                D = 1 - D
                return D.product()
            else:
                return 1

        es = self.terminal_state_map.to_frame().rename(columns={0: 'decendent_end_states'})
        es['n_alternative'] = es['decendent_end_states'].apply(lambda x: len(x) - 1)
        es['end_freq'] = self.length_distribution_proba

        es['wont_stop_in_future'] = es.index.to_series().apply(lambda x: wont_stop_at_future_states(x, es))
        es['state_end_proba'] = es.index.to_series().apply(lambda x: freq_normalize(es.decendent_end_states[x])[x])
        es['ancestor_end_state'] = es.index.to_series().apply(
            lambda x: list({ax for ax, i in zip(es.index, es['decendent_end_states']) if x in i})
        )
        es['state_end_proba_ancestor'] = es.index.to_series().apply(
            lambda x: freq_normalize(es.ancestor_end_state[x])[x]
        )
        es['didnt_stop_at_past'] = es.index.to_series().apply(lambda x: didnt_stop_at_past(x, es))

        es['wsif/sep'] = es['state_end_proba'] / es['wont_stop_in_future']
        es.loc[es['wsif/sep'] >= 1, 'wsif/sep'] = 1

        self.terminal_state_data = es

    def _length_specific_terminal_state(self, length):
        """
        Return all terminal states whose suffix (split by '_') equals the given `length`.
        """
        mask = self.terminal_states.index.to_series().str.split('_').apply(lambda x: int(x[-1])) == length
        return self.terminal_states[mask].index.to_list()

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

    @property
    def is_dag(self):
        """
        Check whether the graph is a Directed Acyclic Graph (DAG).
        """
        return nx.is_directed_acyclic_graph(self.graph)

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
        - P(init) comes from ``initial_states_probability``
        - P(edge_i) are the conditional transition probabilities on each edge
        - P(stop | last_node) comes from ``terminal_state_data['wsif/sep']``

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
        if first_node not in self.initial_states_probability.index:
            if verbose:
                logger.debug(f"First node {first_node} not in initial_states_probability")
            return float('-inf') if use_log else 0.0

        initial_prob = self.initial_states_probability[first_node]

        # 2. Edge transition probabilities
        missing_count = 0
        observed_count = 0

        if use_log:
            log_initial = np.log(initial_prob)
            log_edge_sum = 0.0

            for step1, step2 in window(walk_, 2):
                if self.graph.has_edge(step1, step2):
                    log_edge_sum += np.log(self.graph[step1][step2]['data'].weight)
                    observed_count += 1
                else:
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
            if hasattr(self, 'terminal_state_data') and last_node in self.terminal_state_data.index:
                stop_prob = self.terminal_state_data.loc[last_node, 'wsif/sep']
                log_result += np.log(max(stop_prob, np.finfo(float).eps))
            else:
                log_result += np.log(np.finfo(float).eps)

            return log_result

        else:
            edge_product = 1.0

            for step1, step2 in window(walk_, 2):
                if self.graph.has_edge(step1, step2):
                    edge_product *= self.graph[step1][step2]['data'].weight
                    observed_count += 1
                else:
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
            if hasattr(self, 'terminal_state_data') and last_node in self.terminal_state_data.index:
                stop_prob = self.terminal_state_data.loc[last_node, 'wsif/sep']
                result *= max(stop_prob, np.finfo(float).eps)
            else:
                result *= np.finfo(float).eps

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
        sequence = self.clean_node(random_init)

        while not self.is_stop_condition(current_state):
            current_state = self.random_step(current_state)
            walk.append(current_state)
            sequence += self.clean_node(current_state)

        return walk, sequence

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
        if not self.genetic:
            raise NoGeneDataError(
                operation="gene_variation",
                message="Cannot compute gene variation: this LZGraph has no gene data (genetic=False)."
            )

        encoded = self.encode_sequence(cdr3)

        n_v = [len(self.marginal_vgenes)]
        n_j = [len(self.marginal_jgenes)]

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

        # Helper to convert pandas objects
        def serialize_pandas(obj):
            if isinstance(obj, pd.DataFrame):
                return {'_type': 'DataFrame', 'data': obj.to_dict()}
            elif isinstance(obj, pd.Series):
                return {'_type': 'Series', 'data': obj.to_dict(), 'name': obj.name}
            return obj

        data = {
            '_class': self.__class__.__name__,
            '_module': self.__class__.__module__,
            'graph': graph_data,
            'genetic': self.genetic,
            'n_subpatterns': self.n_subpatterns,
            'n_transitions': self.n_transitions,
            'initial_states': dict(self.initial_states) if isinstance(self.initial_states, pd.Series) else self.initial_states,
            'terminal_states': dict(self.terminal_states) if isinstance(self.terminal_states, pd.Series) else self.terminal_states,
            'lengths': self.lengths,
            'per_node_observed_frequency': self.per_node_observed_frequency,
            'initial_states_probability': serialize_pandas(self.initial_states_probability),
            'length_distribution_proba': serialize_pandas(self.length_distribution_proba),
            'subpattern_individual_probability': serialize_pandas(self.subpattern_individual_probability),
            'impute_missing_edges': self.impute_missing_edges,
            'smoothing_alpha': self.smoothing_alpha,
            'initial_state_threshold': self.initial_state_threshold,
        }

        # Add gene-related attributes if genetic
        if self.genetic:
            if hasattr(self, 'marginal_vgenes'):
                data['marginal_vgenes'] = serialize_pandas(self.marginal_vgenes)
            if hasattr(self, 'marginal_jgenes'):
                data['marginal_jgenes'] = serialize_pandas(self.marginal_jgenes)
            if hasattr(self, 'vj_probabilities'):
                data['vj_probabilities'] = serialize_pandas(self.vj_probabilities)
            if hasattr(self, 'length_distribution'):
                data['length_distribution'] = serialize_pandas(self.length_distribution)
            if hasattr(self, 'observed_vgenes'):
                data['observed_vgenes'] = list(self.observed_vgenes)
            if hasattr(self, 'observed_jgenes'):
                data['observed_jgenes'] = list(self.observed_jgenes)

        # Terminal state data
        if hasattr(self, 'terminal_state_map'):
            data['terminal_state_map'] = serialize_pandas(self.terminal_state_map)
        if hasattr(self, 'terminal_state_data'):
            data['terminal_state_data'] = serialize_pandas(self.terminal_state_data)

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
        # Helper to deserialize pandas objects
        def deserialize_pandas(obj):
            if isinstance(obj, dict) and '_type' in obj:
                if obj['_type'] == 'DataFrame':
                    return pd.DataFrame(obj['data'])
                elif obj['_type'] == 'Series':
                    return pd.Series(obj['data'], name=obj.get('name'))
            return obj

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
        per_node_freq = data.get('per_node_observed_frequency', {})
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
        instance.genetic = data.get('genetic', False)
        instance.genetic_walks_black_list = {}
        instance.n_subpatterns = data.get('n_subpatterns', 0)
        instance.n_transitions = data.get('n_transitions', 0)

        # Restore PGEN configuration
        instance.impute_missing_edges = data.get('impute_missing_edges', False)
        instance.smoothing_alpha = data.get('smoothing_alpha', 0.0)
        instance.initial_state_threshold = data.get('initial_state_threshold', 0)

        # Restore dictionaries (may need to convert back to Series for some)
        instance.initial_states = data.get('initial_states', {})
        instance.terminal_states = data.get('terminal_states', {})
        instance.lengths = data.get('lengths', {})
        instance.cac_graphs = {}
        instance.n_neighbours = {}
        instance.per_node_observed_frequency = data.get('per_node_observed_frequency', {})

        # Restore pandas objects
        instance.initial_states_probability = deserialize_pandas(
            data.get('initial_states_probability', {'_type': 'Series', 'data': {}})
        )
        instance.length_distribution_proba = deserialize_pandas(
            data.get('length_distribution_proba', {'_type': 'Series', 'data': {}})
        )
        instance.subpattern_individual_probability = deserialize_pandas(
            data.get('subpattern_individual_probability', {'_type': 'Series', 'data': {}})
        )

        # Restore gene-related attributes if present
        if instance.genetic:
            if 'marginal_vgenes' in data:
                instance.marginal_vgenes = deserialize_pandas(data['marginal_vgenes'])
            if 'marginal_jgenes' in data:
                instance.marginal_jgenes = deserialize_pandas(data['marginal_jgenes'])
            if 'vj_probabilities' in data:
                instance.vj_probabilities = deserialize_pandas(data['vj_probabilities'])
            if 'length_distribution' in data:
                instance.length_distribution = deserialize_pandas(data['length_distribution'])
            if 'observed_vgenes' in data:
                instance.observed_vgenes = set(data['observed_vgenes'])
            if 'observed_jgenes' in data:
                instance.observed_jgenes = set(data['observed_jgenes'])

        # Restore terminal state data
        if 'terminal_state_map' in data:
            instance.terminal_state_map = deserialize_pandas(data['terminal_state_map'])
        if 'terminal_state_data' in data:
            instance.terminal_state_data = deserialize_pandas(data['terminal_state_data'])

        # Convert initial/terminal states to Series if they're dicts
        if isinstance(instance.initial_states, dict):
            instance.initial_states = pd.Series(instance.initial_states)
        if isinstance(instance.terminal_states, dict):
            instance.terminal_states = pd.Series(instance.terminal_states)

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
            'genetic': self.genetic,
            'n_subpatterns': self.n_subpatterns,
            'n_transitions': self.n_transitions,
            'n_initial_states': len(self.initial_states) if hasattr(self, 'initial_states') else 0,
            'n_terminal_states': len(self.terminal_states) if hasattr(self, 'terminal_states') else 0,
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
