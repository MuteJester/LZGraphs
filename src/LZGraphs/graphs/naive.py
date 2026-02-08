import logging
from collections import defaultdict
from time import time

import networkx as nx
import numpy as np
import pandas as pd

from .lz_graph_base import LZGraphBase
from ..utilities import saturation_function, weight_function
from ..utilities.decomposition import lempel_ziv_decomposition
from ..utilities.misc import window

logger = logging.getLogger(__name__)

__all__ = ["NaiveLZGraph"]


class NaiveLZGraph(LZGraphBase):
    """
          This class implements the logic and infrastructure of the "Naive" version of the LZGraph
          The nodes of this graph are LZ sub-patterns alone without any other additions,
          This class best fits when the objective is extracting features from a repertoire.

          ...

          Methods
          -------

          walk_probability(walk,verbose=True):
              returns the PGEN of the given walk (list of sub-patterns)

          random_walk(steps):
             given a number of steps (sub-patterns) returns a random walk on the graph between a random inital state
             to a random terminal state in the given number of steps

          random_walk_ber_shortest(steps, sfunc_h=0.6, sfunc_k=12):
              given a number of steps (sub-patterns) returns a random walk on the graph between a random inital state
             to a random terminal state, the closer the walk is to the number of selected steps, the higher the
             probability that the next state will be selected using the shortest-path via dijkstra algorithm.
             the saturation function which controls the probability of the selecting a node base on the shortest path
             from the current state is given by the hill function that has 2 parameters, "h" and "h",
             and can be changed by passing value for the "sfunc_h" parameter and the "sfunc_k"  parameter.

          unsupervised_random_walk():
            a random initial state and a random terminal state are selected and a random unsupervised walk is
            carried out until the randomly selected terminal state is reached.

          eigenvector_centrality():
            return the eigen vector centrality value for each node (this function is used as the feature extractor
            for the LZGraph)


          sequence_variation_curve(cdr3_sample):
            given a cdr3 sequence, the function will calculate the value of the variation curve and return
            2 arrays, 1 of the sub-patterns and 1 for the number of out neighbours for each sub-pattern

          graph_summary():
            the function will return a pandas DataFrame containing the graphs
            Chromatic Number,Number of Isolates,Max In Deg,Max Out Deg,Number of Edges

           Attributres
          -------
                nodes:
                    returns the nodes of the graph
                edges:
                    return the edges of the graph


    """

    def __init__(self, cdr3_list, dictionary, verbose=True, smoothing_alpha=0.0):
        """
        in order to derive the dictionary you can use the heleper function "generate_dictionary"
        :param cdr3_list: a list of nucleotide sequence
        :param dictionary: a list of strings, where each string is a sub-pattern that will be converted into a node
        :param verbose:
        :param smoothing_alpha: Laplace smoothing parameter for edge weights. 0.0 means no smoothing.
        """
        super().__init__()
        self.smoothing_alpha = smoothing_alpha

        self.dictionary = dictionary
        # Pre-populate graph with dictionary nodes
        self.graph.add_nodes_from(self.dictionary)

        self._naive_graph_construction(cdr3_list)
        self.verbose_driver(1, verbose)

        self.terminal_states = pd.Series(self.terminal_states)
        self.initial_states = pd.Series(self.initial_states)
        self.length_distribution_proba = self.terminal_states / self.terminal_states.sum()
        self.initial_states_probability = self.initial_states / self.initial_states.sum()
        self.verbose_driver(2, verbose)

        self._derive_subpattern_individual_probability()
        self.verbose_driver(8, verbose)
        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        self._derive_terminal_state_map()
        self.verbose_driver(7, verbose)
        self._derive_stop_probability_data()
        self.verbose_driver(8, verbose)

        self.constructor_end_time = time()
        self.verbose_driver(6, verbose)

    def __repr__(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        return f"NaiveLZGraph(nodes={n_nodes}, edges={n_edges})"

    def __eq__(self, other):
        if not isinstance(other, NaiveLZGraph):
            return NotImplemented
        if nx.utils.graphs_equal(self.graph, other.graph):
            aux = 0
            aux += not self.terminal_states.round(3).equals(other.terminal_states.round(3))
            aux += not self.initial_states.round(3).equals(other.initial_states.round(3))

            # test final_state
            aux += not other.terminal_states.round(3).equals(self.terminal_states.round(3))

            # test length_distribution_proba
            aux += not other.length_distribution_proba.round(3).equals(self.length_distribution_proba.round(3))

            # test subpattern_individual_probability
            aux += not other.subpattern_individual_probability['proba'].round(3).equals(
                self.subpattern_individual_probability['proba'].round(3))

            if aux == 0:
                return True
            else:
                return False

        else:
            return False

    @staticmethod
    def encode_sequence(cdr3):
        """
        Encode a sequence of nucleotides into the NaiveLZGraph format.

        For NaiveLZGraph, this is simply the LZ decomposition without any
        position information.

        Args:
            cdr3 (str): A nucleotide sequence to encode.

        Returns:
            list: A list of LZ sub-patterns.
        """
        return lempel_ziv_decomposition(cdr3)

    def _decomposed_sequence_generator(self, data):
        """NaiveLZGraph uses _naive_graph_construction instead of the base
        class's _simultaneous_graph_construction. This method is not used."""
        return iter([])

    @staticmethod
    def clean_node(node: str) -> str:
        """
        Return the clean subpattern from a node.

        For NaiveLZGraph, nodes are already just the raw LZ subpatterns
        without any position information, so this returns the node unchanged.

        Args:
            node (str): A node identifier (LZ subpattern).

        Returns:
            str: The same subpattern (no transformation needed).
        """
        return node

    def _naive_graph_construction(self, data):
        """Build the graph from a list of nucleotide sequences.

        Performs LZ decomposition on each sequence and constructs edges
        between consecutive subpatterns, tracking edge counts, node
        frequencies, and initial/terminal states.
        """
        edge_counts = defaultdict(int)  # (A_, B_) -> count

        # First pass: collect all edges and frequencies
        for cdr3 in data:
            lz_components = lempel_ziv_decomposition(cdr3)
            edges = window(lz_components, 2)
            for (A, B) in edges:
                edge_counts[(A, B)] += 1
                self.per_node_observed_frequency[A] = self.per_node_observed_frequency.get(A, 0) + 1

            # Ensure the last node appears in frequency counts
            last_subpattern = lz_components[-1]
            self.per_node_observed_frequency[last_subpattern] = \
                self.per_node_observed_frequency.get(last_subpattern, 0)

            # Track terminal and initial states
            self._update_terminal_states(lz_components[-1])
            self._update_initial_states(lz_components[0])

        # Second pass: insert into the graph in bulk
        for (A_, B_), weight_val in edge_counts.items():
            # If nodes were not already in the graph:
            if A_ not in self.graph:
                self.graph.add_node(A_)
            if B_ not in self.graph:
                self.graph.add_node(B_)

            # Add or update the edge with the aggregated weight
            self.graph.add_edge(A_, B_, weight=weight_val)

    def random_walk(self, steps):

        """
           given a number of steps (sub-patterns) returns a random walk on the graph between a random inital state
             to a random terminal state in the given number of steps


                      Parameters:
                              steps (int): number of sub-patterns the resulting walk should contain
                      Returns:
                              (list,str) : a list of LZ sub-patterns representing the random walk and a string
                              matching the walk only translated back into a sequence.
       """
        MAX_TOLERANCE = 1000
        current_state = self._random_initial_state()
        value = [current_state]
        seq = ''
        tolerance = 0

        while len(value) != steps or len(seq) % 3 != 0:
            if tolerance >= MAX_TOLERANCE:
                logger.warning(f"random_walk exceeded {MAX_TOLERANCE} retries, returning current walk")
                break

            w = pd.Series(self.graph[current_state])

            # if terminal state
            if len(w) == 0:
                value = value[:np.random.randint(1, max(len(value), 2))]
                tolerance += 1
                current_state = value[-1]
                seq = ''.join(value)
                continue

            current_state = self.random_step(current_state)
            value.append(current_state)
            seq += current_state

            if current_state in self.terminal_states and len(value) == steps and len(seq) % 3 == 0:
                return value, seq
            elif len(value) == steps and current_state not in self.terminal_states:
                value = value[:np.random.randint(1, max(len(value), 2))]
                tolerance += 1
                current_state = value[-1]
                seq = ''.join(value)
            elif len(value) == steps and current_state in self.terminal_states and len(seq) % 3 != 0:
                value = value[:np.random.randint(1, max(len(value), 2))]
                tolerance += 1
                current_state = value[-1]
                seq = ''.join(value)

        return value, seq

    def random_walk_ber_shortest(self, steps, sfunc_h=0.6, sfunc_k=12):
        """
             given a number of steps (sub-patterns) returns a random walk on the graph between a random inital state
             to a random terminal state, the closer the walk is to the number of selected steps, the higher the
             probability that the next state will be selected using the shortest-path via dijkstra algorithm.
             the saturation function which controls the probability of the selecting a node base on the shortest path
             from the current state is given by the hill function that has 2 parameters, "h" and "h",
             and can be changed by passing value for the "sfunc_h" parameter and the "sfunc_k"  parameter.

             The saturation function formally defined as : 1 / (1 + ((h / x) ** k))

                      Parameters:
                              steps (int): number of sub-patterns the resulting walk should contain
                              sfunc_h (float): the h parameter of the saturation hill function
                              sfunc_k (int): the k parameter of the saturation hill function
                      Returns:
                              (list,str) : a list of LZ sub-patterns representing the random walk and a string
                              matching the walk only translated back into a sequence.
       """
        MAX_TOLERANCE = 1000
        current_state = self._random_initial_state()
        istate = current_state
        value = [current_state]
        seq = ''
        tolerance = 0

        while len(value) != steps or len(seq) % 3 != 0:
            if tolerance >= MAX_TOLERANCE:
                logger.warning(f"random_walk_ber_shortest exceeded {MAX_TOLERANCE} retries, returning current walk")
                break

            w = pd.Series(self.graph[current_state])

            # if terminal state
            if len(w) == 0:
                value = value[:np.random.randint(1, max(len(value), 2))]
                tolerance += 1
                current_state = value[-1]
                seq = ''.join(value)
                continue

            SP = nx.shortest_path(self.graph, source=current_state, target=istate, weight=weight_function)

            if np.random.binomial(1,
                                  saturation_function(((len(value) / steps)), sfunc_h, sfunc_k)) == 1 and len(
                SP) >= 3:
                current_state = SP[1]
                value.append(current_state)
                seq += current_state
            else:
                current_state = self.random_step(current_state)
                value.append(current_state)
                seq += current_state

            if current_state in self.terminal_states and len(value) == steps and len(seq) % 3 == 0:
                return value, seq
            elif len(value) == steps and current_state not in self.terminal_states:
                value = value[:np.random.randint(1, max(len(value), 2))]
                tolerance += 1
                current_state = value[-1]
                seq = ''.join(value)
            elif len(value) == steps and current_state in self.terminal_states and len(seq) % 3 != 0:
                value = value[:np.random.randint(1, max(len(value), 2))]
                tolerance += 1
                current_state = value[-1]
                seq = ''.join(value)

        return value, seq
