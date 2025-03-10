import logging
import heapq
import re
from collections import Counter
from multiprocessing.pool import ThreadPool
from time import time

import networkx as nx
import numpy as np
import pandas as pd

# Utility functions
from ..Utilities.misc import chunkify, choice, get_dictionary_subkeys

# The three mixins
from ..Mixins import GeneLogicMixin
from ..Mixins import RandomWalkMixin
from ..Mixins import GenePredictionMixin

# Create a logger for this module
logger = logging.getLogger(__name__)

class LZGraphBase(GeneLogicMixin, RandomWalkMixin, GenePredictionMixin):
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

    def __eq__(self, other):
        """
        This method tests whether two LZGraphs are equal, i.e. have the same node, edges,
        and metadata on the edges.
        """
        if nx.utils.graphs_equal(self.graph, other.graph):
            aux = 0
            aux += self.genetic_walks_black_list != other.genetic_walks_black_list
            aux += self.n_subpatterns != other.n_subpatterns

            # Compare initial_states, terminal_states, etc.
            aux += not self.initial_states.round(3).equals(other.initial_states.round(3))
            aux += not self.terminal_states.round(3).equals(other.terminal_states.round(3))

            # Compare gene-related distributions
            aux += not other.marginal_vgenes.round(3).equals(self.marginal_vgenes.round(3))
            aux += not other.vj_probabilities.round(3).equals(self.vj_probabilities.round(3))
            aux += not other.length_distribution.round(3).equals(self.length_distribution.round(3))
            aux += not other.terminal_states.round(3).equals(self.terminal_states.round(3))
            aux += not other.length_distribution_proba.round(3).equals(self.length_distribution_proba.round(3))

            return (aux == 0)
        else:
            return False

    @staticmethod
    def encode_sequence(sequence):
        """
        Abstract method: should be overridden by subclasses.
        Processes a sequence of symbols (nucleotides/amino acids) into a list of subpatterns (nodes).
        """
        raise NotImplementedError

    @staticmethod
    def clean_node(base):
        """
        Extract only nucleotides from a node string containing frame/position info.
        """
        match = re.search(r'[ATGC]+', base)
        return match.group(0) if match else ""

    def _decomposed_sequence_generator(self, data):
        """
        Abstract method: should be overridden by subclasses.
        Yields (steps, locations[, v, j]) per row for building the graph.
        """
        raise NotImplementedError

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
                # ensure final node is counted
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)
        else:
            for output in processing_stream:
                steps, locations = output
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = f"{A}_{loc_a}"
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = f"{B}_{loc_b}"
                    self._insert_edge_and_information_no_genes(A_, B_)
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)

    def _insert_edge_and_information_no_genes(self, node_a, node_b):
        """
        Insert or update an edge (node_a -> node_b) with no gene info, incrementing weight by 1.
        """
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]["weight"] += 1
        else:
            self.graph.add_edge(node_a, node_b, weight=1)
        self.n_transitions += 1

    def _normalize_edge_weights(self):
        """
        Normalize each outgoing edge weight by the total count for its source node.
        The sum of outgoing edges from any node then becomes 1.
        """
        for edge_a, edge_b in self.graph.edges:
            total = self.per_node_observed_frequency[edge_a]
            if total > 0:
                self.graph[edge_a][edge_b]['weight'] /= total

    def _get_node_info_df(self, node_a, V=None, J=None, condition='and'):
        """
        Returns a DataFrame containing metadata for edges from node_a,
        optionally filtered by presence of V/J genes (with 'and'/'or' logic).
        """
        if V is None or J is None:
            return pd.DataFrame(dict(self.graph[node_a]))
        else:
            node_data = self.graph[node_a]
            if condition == 'and':
                partial_dict = {
                    pk: node_data[pk] for pk in node_data
                    if V in node_data[pk] and J in node_data[pk]
                }
            else:
                partial_dict = {
                    pk: node_data[pk] for pk in node_data
                    if V in node_data[pk] or J in node_data[pk]
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
            df = pd.DataFrame(dict(node_data))
            return df if not asdict else df.to_dict()

        # Filter edges to those containing V and J
        partial_dict = {
            pk: {feature: node_data[pk][feature]}
            for pk in node_data
            if V in node_data[pk] and J in node_data[pk]
        }
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
                edge_info = dict(self.graph[state])
                observed_gene_paths = set(get_dictionary_subkeys(edge_info))
                if len(set(observed_gene_paths) & {selected_v, selected_j}) != 2:
                    neighbours = 0
                else:
                    neighbours = 2
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
        Summation of edge weights by source node -> yields empirical probabilities for each node.
        """
        weight_df = pd.Series(nx.get_edge_attributes(self.graph, 'weight')).reset_index()
        self.subpattern_individual_probability = weight_df.groupby('level_0').sum(numeric_only=True).rename(columns={0: 'proba'})
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
            probabilities = [node_data[s]['weight'] for s in states]
            return states, probabilities
        else:
            df = pd.DataFrame(dict(self.graph[node])).T
            return df

    def _batch_gene_weight_normalization(self, n_process=3, verbose=False):
        """
        Chunk all edges, then call `_normalize_gene_weights` in parallel across processes.
        """
        edges_list = list(self.graph.edges)
        if not edges_list:
            return
        batches = chunkify(edges_list, max(len(edges_list) // 3, 1))
        with ThreadPool(n_process) as pool:
            pool.map(self._normalize_gene_weights, batches)

    def _update_terminal_states(self, terminal_state):
        self.terminal_states[terminal_state] = self.terminal_states.get(terminal_state, 0) + 1

    def _update_initial_states(self, initial_state):
        self.initial_states[initial_state] = self.initial_states.get(initial_state, 0) + 1

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
        return nx.algorithms.eigenvector_centrality(self.graph, weight='weight', max_iter=max_iter)

    def isolates(self):
        """
        Return a list of isolate nodes (nodes with zero edges).
        """
        return list(nx.isolates(self.graph))

    def drop_isolates(self):
        """
        Remove isolates (nodes with zero edges) from the graph.
        """
        self.graph.remove_nodes_from(self.isolates())

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
