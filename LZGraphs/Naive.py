from itertools import product
import networkx as nx
import numpy as np
import pandas as pd
from .misc import window
from .decomposition import lempel_ziv_decomposition
from time import time


def saturation_function(x, h, k):
    """
          a version of the hill saturation function used in the "random_walk_ber_shortest" random walk method
          where based on the parameters the function controls the probability of choosing the shortest path action
          at each step

                  Parameters:
                          x (float): the length of the input at time t divided by the target length
                          h (float): the saturation constant
                          k (int): the saturation factor degree

                  Returns:
                          float : value between 0 - 1 (used as probability for bernoulli trail)
   """
    return 1 / (1 + ((h / x) ** k))


# a lambda function for networkx proper weight usage
wfs = lambda x, y, z: 1 - z['weight']


def generate_dictionary(max_len):
    """
        this function will generate all unique K-Mers for k starting at 1 up to max_len
        this is a helper function used to derive the node dictionary for the naive LZ-Graph
        where in general the length distribution of nucleotide sub-patterns are maxed at about 6

                  Parameters:
                          max_len (int): the length of maximal K-Mer family


                  Returns:
                          list : a list of all unique K-Mers for K =1 --> k = max_len
   """

    N = max_len
    DICT = []
    for i in range(1, N + 1):
        DICT += [''.join(i) for i in product(['A', 'T', 'G', 'C'], repeat=i)]

    return DICT


class NaiveLZGraph:
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

    def __init__(self, cdr3_list, dictionary, verbose=True):
        """
        in order to derive the dictionary you can use the heleper function "generate_dictionary"
        :param cdr3_list: a list of nucleotide sequence
        :param dictionary: a list of strings, where each string is a sub-pattern that will be converted into a node
        :param verbose:
        """
        self.constructor_start_time = time()

        self.dictionary = dictionary
        lz_components = []
        self.terminal_states = dict()
        self.initial_states = dict()
        self.per_node_observed_frequency = dict()
        # create graph
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.dictionary)
        # extract lz components

        self.__simultaneous_graph_construction(cdr3_list)
        self.verbose_driver(1, verbose)

        self.terminal_states = pd.Series(self.terminal_states)
        self.initial_states = pd.Series(self.initial_states)
        self.length_distribution_proba = self.terminal_states / self.terminal_states.sum()
        self.verbose_driver(2, verbose)

        self._derive_subpattern_individual_probability()
        self.verbose_driver(8, verbose)
        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        self.__derive_terminal_state_map()
        self.verbose_driver(7, verbose)
        self.derive_final_state_data()
        self.verbose_driver(8, verbose)

        self.constructor_end_time = time()
        self.verbose_driver(6, verbose)
        self.verbose_driver(-2, verbose)


    def verbose_driver(self, message_number, verbose):
        if not verbose:
            return None

        if message_number == -2:
            print("===" * 10)
            print('\n')
        elif message_number == 0:
            CT = round(time() - self.constructor_start_time, 2)
            print("Gene Information Loaded..", '| ', CT, ' Seconds')
        elif message_number == 1:
            CT = round(time() - self.constructor_start_time, 2)
            print("Graph Constructed..", '| ', CT, ' Seconds')
        elif message_number == 2:
            CT = round(time() - self.constructor_start_time, 2)
            print("Graph Metadata Derived..", '| ', CT, ' Seconds')
        elif message_number == 3:
            CT = round(time() - self.constructor_start_time, 2)
            print("Graph Edge Weight Normalized..", '| ', CT, ' Seconds')
        elif message_number == 4:
            CT = round(time() - self.constructor_start_time, 2)
            print("Graph Edge Gene Weights Normalized..", '| ', CT, ' Seconds')
        elif message_number == 5:
            CT = round(time() - self.constructor_start_time, 2)
            print("Terminal State Map Derived..", '| ', CT, ' Seconds')
        elif message_number == 6:
            CT = round(self.constructor_end_time - self.constructor_start_time, 2)
            print("LZGraph Created Successfully..", '| ', CT, ' Seconds')
        elif message_number == 7:
            CT = round(time() - self.constructor_start_time, 2)
            print("Terminal State Map Derived..", '| ', CT, ' Seconds')
        elif message_number == 8:
            CT = round(time() - self.constructor_start_time, 2)
            print("Individual Subpattern Empirical Probability Derived..", '| ', CT, ' Seconds')
        elif message_number == 9:
            CT = round(time() - self.constructor_start_time, 2)
            print("Terminal State Conditional Probabilities Map Derived..", '| ', CT, ' Seconds')

    def __eq__(self, other):
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

    def _derive_subpattern_individual_probability(self):
        weight_df = pd.Series(nx.get_edge_attributes(self.graph, 'weight')).reset_index()
        self.subpattern_individual_probability = weight_df.groupby('level_0').sum().rename(columns={0: 'proba'})
        self.subpattern_individual_probability /= self.subpattern_individual_probability.proba.sum()

    def __simultaneous_graph_construction(self, data):
        for cdr3 in (data):
            lz_components = lempel_ziv_decomposition(cdr3)
            edges = (window(lz_components, 2))

            for (A, B) in edges:
                self.per_node_observed_frequency[A] = self.per_node_observed_frequency.get(A, 0) + 1
                self._insert_edge_and_information(A, B)
            self.per_node_observed_frequency[B] = self.per_node_observed_frequency.get(B, 0)

            self._update_terminal_states(lz_components[-1])
            self._update_initial_states(lz_components[0])

    def _normalize_edge_weights(self):
        # normalize edges
        # weight_df = pd.Series(nx.get_edge_attributes(self.graph, 'weight')).reset_index()
        # for idx, group in weight_df.groupby('level_0'):
        #     weight_df.loc[group.index, 0] /= group[0].sum()
        # nx.set_edge_attributes(self.graph, weight_df.set_index(['level_0', 'level_1']).to_dict()[0], 'weight')

        for edge_a, edge_b in self.graph.edges:
            node_observed_total = self.per_node_observed_frequency[edge_a]
            self.graph[edge_a][edge_b]['weight'] /= node_observed_total

    def _insert_edge_and_information(self, A_, B_):
        if self.graph.has_edge(A_, B_):
            self.graph[A_][B_]["weight"] += 1
        else:
            self.graph.add_edge(A_, B_, weight=1)

    def _update_terminal_states(self, terminal_state):
        self.terminal_states[terminal_state] = self.terminal_states.get(terminal_state, 0) + 1

    def _update_initial_states(self, initial_state):
        self.initial_states[initial_state] = self.initial_states.get(initial_state, 0) + 1

    def isolates(self):
        """
              A function that returns the list of all isolates in the graph.
              an isolate is a node that is connected to 0 edges (unseen sub-pattern).

                      Parameters:
                              None

                      Returns:
                              list : a list of isolates
       """
        return list(nx.isolates(self.graph))

    def drop_isolates(self):
        """
                 A function to drop all isolates from the graph.

                         Parameters:
                                 None

                         Returns:
                                 None
          """
        self.graph.remove_nodes_from(self.isolates())

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def __derive_terminal_state_map(self):
        """
            This function derives a mapping between each terminal state and all terminal state that could
            be reached from it
          """
        terminal_state_map = np.zeros((len(self.terminal_states), len(self.terminal_states)))
        for pos_1, terminal_1 in enumerate(self.terminal_states.index):
            for pos_2, terminal_2 in enumerate(self.terminal_states.index):
                terminal_state_map[pos_1][pos_2] = nx.has_path(self.graph, source=terminal_1, target=terminal_2)
        terminal_state_map = pd.DataFrame(terminal_state_map,
                                          columns=self.terminal_states.index,
                                          index=self.terminal_states.index).apply(
            lambda x: x.apply(lambda y: x.name if y == 1 else np.nan), axis=0)
        # np.fill_diagonal(terminal_state_map.values, np.nan)

        self.terminal_state_map = pd.Series(terminal_state_map.apply(lambda x: (x.dropna().to_list()), axis=1),
                                            index=self.terminal_states.index)

    def derive_final_state_data(self):

        """
        This function derives a dataframe that contains all terminal state info used for probability normalization
        of stopping at a terminal state,
        the function mainly calculates all terminal state that colud be visited before reach any other terminal state
        and all terminal state that could be visited from any given terminal state.
        :return:
        """

        def freq_normalize(target):
            # all possible alternative future terminal states from current state
            D = self.length_distribution_proba.loc[target].copy()
            # normalize observed frequencey
            D /= D.sum()
            return D

        def wont_stop_at_future_states(state, es):
            D = freq_normalize(es.decendent_end_states[state])
            # remove current state
            current_freq = D.pop(state)
            if len(D) >= 1:
                D = 1 - D
                return D.product()
            else:
                return 1

        def didnt_stop_at_past(state, es):
            # all possible alternative future terminal states from current state
            D = freq_normalize(es.ancestor_end_state[state])
            # remove current state
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
        es['wont_stop_in_future'] = 0
        es['wont_stop_in_future'] = es.index.to_series().apply(lambda x: wont_stop_at_future_states(x, es))

        es['state_end_proba'] = es.index.to_series().apply(lambda x: freq_normalize(es.decendent_end_states[x])[x])
        es['ancestor_end_state'] = es.index.to_series() \
            .apply(
            lambda x: list(set([ax for ax, i in zip(es.index, es['decendent_end_states']) if x in i])))

        es['state_end_proba_ancestor'] = es.index.to_series().apply(
            lambda x: freq_normalize(es.ancestor_end_state[x])[x])

        es['didnt_stop_at_past'] = 1
        es['didnt_stop_at_past'] = es.index.to_series().apply(lambda x: didnt_stop_at_past(x, es))
        # state end freq normalized by wont stop in future

        es['wsif/sep'] = es['state_end_proba'] / es['wont_stop_in_future']
        es.loc[es['wsif/sep'] >= 1, 'wsif/sep'] = 1

        # ancestor and decendent product
        # es['normalized'] = (es['state_end_proba']*es['state_end_proba_ancestor']) / (es['wont_stop_in_future']*es['didnt_stop_at_past'])

        self.terminal_state_data = es

    def walk_probability(self, walk, verbose=True):
        """
             given a walk (a sequence converted into LZ sub-pattern) return the probability of generation (PGEN)
             of the walk.

             you can use "lempel_ziv_decomposition" from this libraries decomposition module in order to convert a
             sequence into LZ sub-patterns

                      Parameters:
                              walk (list): a list of LZ - sub-patterns

                      Returns:
                              float : the probability of generating such a walk (PGEN)
       """
        if type(walk) == str:
            LZ = lempel_ziv_decomposition(walk)
            walk_ = LZ
        else:
            walk_ = walk

        proba = self.subpattern_individual_probability['proba'][walk_[0]]
        for step1, step2 in window(walk_, 2):
            if self.graph.has_edge(step1, step2):
                proba *= self.graph.get_edge_data(step1, step2)['weight']
            else:
                if verbose:
                    print('No Edge Connecting| ', step1, '-->', step2)
                return 0
        return proba

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
        current_state = self.__random_initial_state()
        value = [current_state]
        seq = ''
        tolorance = 0
        # if tolorance == 5:
        #     return None

        while len(value) != steps or len(seq) % 3 != 0:
            # for _ in range(steps):
            w = pd.Series(self.graph[current_state])

            # if terminal state
            if len(w) == 0:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i[:-1] for i in value])
                continue

            # w = w.apply(lambda x : x['weight'])

            current_state = self.__random_step(current_state)
            value.append(current_state)
            seq += current_state

            if current_state in self.terminal_states and len(value) == steps and len(seq) % 3 == 0:
                return value, seq
            elif len(value) == steps and current_state not in self.terminal_states:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])
            elif len(value) == steps and current_state in self.terminal_states and len(seq) % 3 != 0:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])

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
        current_state = self.__random_initial_state()
        istate = current_state
        value = [current_state]
        seq = ''
        tolorance = 0
        # if tolorance == 5:
        #     return None

        while len(value) != steps or len(seq) % 3 != 0:
            # for _ in range(steps):
            w = pd.Series(self.graph[current_state])

            # if terminal state
            if len(w) == 0:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])
                continue

            SP = nx.shortest_path(self.graph, source=current_state, target=istate, weight=wfs)

            if np.random.binomial(1,
                                  saturation_function(((len(current_state) / steps)), sfunc_h, sfunc_k)) == 1 and len(
                SP) >= 3:
                current_state = SP[1]
                value.append(current_state)
                seq += current_state
            else:
                current_state = self.__random_step(current_state)
                value.append(current_state)
                seq += current_state

            if current_state in self.terminal_states and len(value) == steps and len(seq) % 3 == 0:
                return value, seq
            elif len(value) == steps and current_state not in self.terminal_states:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])
            elif len(value) == steps and current_state in self.terminal_states and len(seq) % 3 != 0:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])

        return value, seq

        # return self.random_walk(steps, initial_state, final_states,tolorance+1)

    def __get_state_weights(self, node, v=None, j=None):
        """
        Given a node, return all the possible translation from that node and their respective weights
        :param node:
        :param v:
        :param j:
        :return:
        """
        if v is None and j is None:
            node_data = self.graph[node]
            states = list(node_data.keys())
            probabilities = [node_data[i]['weight'] for i in states]
            return states, probabilities
        else:
            return pd.DataFrame(dict(self.graph[node])).T

    def is_stop_condition(self, state, selected_gene_path_v=None, selected_gene_path_j=None):
        """
        give a state, return True if stop condition is met, else return False
        :param state:
        :param selected_gene_path_v:
        :param selected_gene_path_j:
        :return:
        """
        if state not in self.terminal_states:
            return False
        if selected_gene_path_j is not None:
            edge_info = pd.DataFrame(dict(self.graph[state]))
            if len(set(edge_info.index) & {selected_gene_path_v, selected_gene_path_j}) != 2:
                neighbours = 0
            else:
                neighbours = 2
        else:
            neighbours = self.graph.out_degree(state)
        if (neighbours) == 0:
            return True
        else:
            #             D = self.length_distribution_proba.loc[self.terminal_state_map[state]].copy()
            #             D /=D.sum()
            #             end_porba = D[state]

            #             D.pop(state)
            #             if len(D) >= 1:
            #                 D=1-D
            #                 end_porba /= D.product()

            #             decision = np.random.binomial(1,end_porba)==1
            stop_probability = self.terminal_state_data.loc[state, 'wsif/sep']
            decision = np.random.binomial(1, stop_probability) == 1
            return decision

    def __random_step(self, state):
        """
        Given the current state, pick and take a random step based on the translation probabilities
        :param state:
        :return:
        """
        states, probabilities = self.__get_state_weights(state)
        return np.random.choice(states, size=1, p=probabilities).item()

    def __random_initial_state(self):
        """
        Select a random initial state based on the marginal distribution of initial states.
        :return:
        """
        first_states = self.initial_states / self.initial_states.sum()
        return np.random.choice(first_states.index, size=1, p=first_states.values)[0]

    def unsupervised_random_walk(self):
        """
             a random initial state and a random terminal state are selected and a random unsupervised walk is
            carried out until the randomly selected terminal state is reached.

                      Parameters:
                              None

                      Returns:
                              (list,str) : a list of LZ sub-patterns representing the random walk and a string
                              matching the walk only translated back into a sequence.
       """
        random_initial_state = self.__random_initial_state()

        current_state = random_initial_state
        walk = [random_initial_state]
        sequence = random_initial_state

        while not self.is_stop_condition(current_state):
            # take a random step
            current_state = self.__random_step(current_state)

            walk.append(current_state)
            sequence += (current_state)
        return walk, sequence

    def eigenvector_centrality(self):
        """
           return the eigen vector centrality value for each node (this function is used as the feature extractor
            for the LZGraph)
        :return:
        """
        return nx.algorithms.eigenvector_centrality(self.graph, weight='weight')

    def voterank(self, n_nodes=25):
        """
         Uses the VoteRank algorithm to return the top N influential nodes in the graph, where N is equal to n_nodes

                  Parameters:
                          n_nodes (int): the number of most influential nodes to find

                  Returns:
                          list : a list of top influential nodes
        """
        return nx.algorithms.voterank(self.graph, number_of_nodes=n_nodes)

    def sequence_variation_curve(self, cdr3_sample):
        """
        given a sequence this function will return 2 list,
        the first is the lz-subpattern path through the graph and the second list is the number
        of possible choices that can be made at each sub-pattern
        :param cdr3_sample:
        :return:
        """
        encoded = lempel_ziv_decomposition(cdr3_sample)
        curve = [self.graph.out_degree(i) for i in encoded]
        return encoded, curve

    def graph_summary(self):
        """
          the function will return a pandas DataFrame containing the graphs
            Chromatic Number,Number of Isolates,Max In Deg,Max Out Deg,Number of Edges
        """
        R = pd.Series({
            'Chromatic Number': max(nx.greedy_color(self.graph).values()) + 1,
            'Number of Isolates': nx.number_of_isolates(self.graph),
            'Max In Deg': max(dict(self.graph.in_degree).values()),
            'Max Out Deg': max(dict(self.graph.out_degree).values()),
            'Number of Edges': len(self.graph.edges)
        })
        return R
