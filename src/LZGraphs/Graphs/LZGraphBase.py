import heapq
import re
from collections import Counter
from multiprocessing.pool import ThreadPool
from time import time
import networkx as nx
import numpy as np
import pandas as pd
from ..Utilities.misc import chunkify, choice, get_dictionary_subkeys


class LZGraphBase:
    """

         This abstract class provides the base functionality and attributes shared between the different LZGraphs
         (excluding the Naive LZGraph).

         Args:
             Abstract Class

         Attributes:

             constructor_start_time (int): An integer recording the time in seconds the constructor was initiated
             (returned via the python time library time function)
             graph (netowrkx.DiGraph): An empty DiGraph based on the networkx library, this object will keep the information
             we derive throughout the constructor routine
             genetic (bool): if True this flag will allow genetic function to be used as V and J annotation were provoided
             along with the sequences. False means no genetic information is encoded into the graph.
             genetic_walks_black_list (dict): This dictionary maps (edge / edge+vj) to another edge that leads a
             generated random walk to a "dead-end" as no more steps can be taken due to missing v/j information.
             (this applies only to genomic random walk where the randomly selected v and j genes have to appear at each edge).
             The more sequences are generated by a particular LZGraph the more dead ends will be found, the quicker
             the sequences will be generated.
             n_subpatterns (int):  The total number of sequences used to construct the LZGraph.
             initial_states (dict): A dictionary containing all initial state and the number of time each one of them
             was observed.
             terminal_states (dict): A dictionary containing all terminal state and the number of time each one of them
             was observed.
             initial_states_probability (pd.Series): A pandas series containing the normalized counts of the initial
             states.
             lengths (dict): a dictionary that maps the different observed sequences length to the number of time
             that length has been observed.
             cac_graphs (dict): This is dictionary used to keep different sub-graphs based on the condition in the
             key, currently deprecated.
             n_transitions (int): The total number of transition observed while constructing the graph.
             n_neighbours (dict): Keep track of the number of neighbours given a certain condition in the key.
             currently deprecated.
             length_distribution_proba (pd.Series): A pandas series mapping sequence length to a probability value
             based on the sequence used to construct the LZGraph.
             subpattern_individual_probability (pd.Series): A pandas series mapping the different graph nodes to the
             emperical probability of observing that specific node independent of the graph structure.


         """
    def __init__(self):
        # start time of constructor
        self.constructor_start_time = time()
        # create graph
        self.graph = nx.DiGraph()
        # check for V and J gene data in input
        self.genetic = False
        # a list of invalid genetic walks
        self.genetic_walks_black_list = {}
        # total number of sub-patterns
        self.n_subpatterns = 0

        self.initial_states, self.terminal_states = dict(), dict()
        self.initial_states_probability = pd.Series(dtype=np.float64)
        self.lengths = dict()
        self.cac_graphs = dict()
        self.n_transitions = 0
        self.n_neighbours = dict()
        self.length_distribution_proba = pd.Series(dtype=np.float64)
        self.subpattern_individual_probability = pd.Series(dtype=np.float64)
        # per node observed frequency for unity operation
        self.per_node_observed_frequency = dict()

    def __eq__(self, other):
        """ This method test whether two LZGraph are equal, i.e. have the same node,edges and metadata on the edges.
                                    Returns:
                                        bool: True if graph equal else False
          """
        if nx.utils.graphs_equal(self.graph, other.graph):
            aux = 0
            aux += self.genetic_walks_black_list != other.genetic_walks_black_list
            aux += self.n_subpatterns != other.n_subpatterns
            aux += not self.initial_states.round(3).equals(other.initial_states.round(3))
            aux += not self.terminal_states.round(3).equals(other.terminal_states.round(3))

            # test marginal_vgenes
            aux += not other.marginal_vgenes.round(3).equals(self.marginal_vgenes.round(3))

            # test vj_probabilities
            aux += not other.vj_probabilities.round(3).equals(self.vj_probabilities.round(3))

            # test length_distribution
            aux += not other.length_distribution.round(3).equals(self.length_distribution.round(3))

            # test final_state
            aux += not other.terminal_states.round(3).equals(self.terminal_states.round(3))

            # test length_distribution_proba
            aux += not other.length_distribution_proba.round(3).equals(self.length_distribution_proba.round(3))

            if aux == 0:
                return True
            else:
                return False

        else:
            return False

    @staticmethod
    def encode_sequence(sequence):
        """
          This method is abstract, when creating a new LZGraph class, one should override this method with a method
          that process a sequence of symbols (nucleotides/amino acids) and return a list of string where each
          string represents a node in the graph. note that it is implied that each pair of sequential string in the
          returned list are connect by an edge.

                  Args:
                          sequence (str): a string from which to derive sub-patterns

                  Returns:
                          list : a list of unique sub-patterns
       """
        raise NotImplementedError

    @staticmethod
    def clean_node(base):
        """
          given a sub-pattern that has reading frame and position added to it, cleans it and returns
          only the nucleotides from the string

                  Args:
                          base (str): a node from the NDPLZGraph

                  Returns:
                          str : only the nucleotides of the node
     """
        return re.search(r'[ATGC]*', base).group()

    def _decomposed_sequence_generator(self,data):
        """
           This abstract method should be overwritten in any LZGraph class inheriting this baseclass,
           this method should return a generator that will take all the data provided to the constractor and
           yield the relevant node and edge data each time it is called to be added to graph.

                   Args:
                           data (pd.DataFrame): a pandas Dataframe containing all information relevant for the
                           construction of an LZGraph (sequences / genomic data etc).

                   Returns:
                           generator : a generator the yield the information needed to add nodes and edges to the graph
                           attribute.
      """
        raise NotImplementedError

    def _simultaneous_graph_construction(self, data):
        """
                  This method leverage the generator implemented in _decomposed_sequence_generator
                  in order to iterate over the information returned by the generator and each iteration insert
                  the node and edge data into the networkx DiGraph and embed and genetic information provided.

                          Args:
                                  data (pd.DataFrame): a pandas Dataframe containing all information relevant for the
                                  construction of an LZGraph (sequences / genomic data etc).

                          Returns:
                                  None
             """
        processing_stream = self._decomposed_sequence_generator(data)
        if self.genetic:
            for output in processing_stream:
                steps, locations,v,j = output

                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = A + '_' + str(loc_a)
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = B + '_' + str(loc_b)
                    self._insert_edge_and_information(A_, B_, v, j)
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)
        else:

            for output in processing_stream:
                steps, locations = output
                for (A, B), (loc_a, loc_b) in zip(steps, locations):
                    A_ = A + '_' + str(loc_a)
                    self.per_node_observed_frequency[A_] = self.per_node_observed_frequency.get(A_, 0) + 1
                    B_ = B + '_' + str(loc_b)
                    self._insert_edge_and_information_no_genes(A_, B_)
                self.per_node_observed_frequency[B_] = self.per_node_observed_frequency.get(B_, 0)


    def _normalize_edge_weights(self):
        """
             This method iterates over all the edges in the graph, normalizing edge edgre by the total number
             of observed edges at the source node of that edge. the resulting graph will contain nodes that the sum
             of weights for all edges expanding fit that node sum to 1.
            """
        for edge_a, edge_b in self.graph.edges:
            node_observed_total = self.per_node_observed_frequency[edge_a]
            self.graph[edge_a][edge_b]['weight'] /= node_observed_total

    def _get_node_info_df(self, node_a, V=None, J=None, condition='and'):
        """
           This method returns a dataframe containing the information encoded in each edge spaning from a given node,
           the indecies of the dataframe are the edges and the column are the different attributes.

                   Args:
                           node_a (str): a string matching a node in the graph
                           V (str): optional, if provided together with J this can be used to filter out node edges
                           J (str): optional, if provided together with V this can be used to filter out node edges
                           condition (str):optional, one of the values = ("and"/"or"), based on the value the function
                           will return the edges of the given node that have information for V and J (the other
                           two parameters passed) or for edge containing V or J/

                   Returns:
                           pd.DataFrame : a DataFrame containing all edges and their metadata for a specific node.
            """
        if V is None or J is None:
            return pd.DataFrame(dict(self.graph[node_a]))
        else:
            node_data = self.graph[node_a]
            if condition == 'and':
                partial_dict = {pk: node_data[pk] for pk in node_data if V in node_data[pk] and J in node_data[pk]}
            else:
                partial_dict = {pk: node_data[pk] for pk in node_data if V in node_data[pk] or J in node_data[pk]}
            return pd.DataFrame(partial_dict)

    def _get_node_feature_info_df(self, node_a, feature, V=None, J=None, asdict=False):
        if V is None or J is None:
            return pd.DataFrame(dict(self.graph[node_a]))
        else:
            node_data = self.graph[node_a]
            partial_dict = {pk: {feature: node_data[pk][feature]} for pk in node_data \
                            if V in node_data[pk] and J in node_data[pk]}
            if asdict:
                return partial_dict
            else:
                return pd.DataFrame(partial_dict)

    def is_stop_condition(self, state, selected_v=None, selected_j=None):
        if state not in self.terminal_states:
            return False
        if self.genetic:
            if selected_j is not None:
                # edge_info = self._get_node_info_df(state, selected_v, selected_j,condition='or')
                edge_info = dict(self.graph[state])  # pd.DataFrame()
                observed_gene_paths = set(get_dictionary_subkeys(edge_info))
                if len(set(observed_gene_paths) & {selected_v, selected_j}) != 2:
                    neighbours = 0
                else:
                    neighbours = 2
            else:
                neighbours = self.graph.out_degree(state)
            if (neighbours) == 0:
                return True
            else:
                stop_probability = self.terminal_state_data.loc[state, 'wsif/sep']
                decision = np.random.binomial(1, stop_probability) == 1
                return decision
        else:
            stop_probability = self.terminal_state_data.loc[state, 'wsif/sep']
            decision = np.random.binomial(1, stop_probability) == 1
            return decision

    def genomic_random_walk(self, initial_state=None, vj_init='marginal'):
        """
             given a target sequence length and an initial state, the function will select a random
             V and a random J genes from the observed gene frequency in the graph's "Training data" and
             generate a walk on the graph from the initial state to a terminal state while making sure
             at each step that both the selected V and J genes were seen used by that specific sub-pattern.

             if seq_len is equal to "unsupervised" than a random seq len will be returned
        """
        self._raise_genetic_mode_error()
        selected_v, selected_j = self._select_random_vj_genes(vj_init)

        if initial_state is None:
            current_state = self._random_initial_state()
            walk = [current_state]
        else:
            current_state = initial_state
            walk = [initial_state]

        # while the walk is not in a valid final state
        while not self.is_stop_condition(current_state, selected_v, selected_j):
            # get the node_data for the current state
            edge_info = self._get_node_feature_info_df(current_state, 'weight', selected_v, selected_j, asdict=True)

            if (current_state, selected_v, selected_j) in self.genetic_walks_black_list:
                for col in self.genetic_walks_black_list[(current_state, selected_v, selected_j)]:
                    edge_info.pop(col)
                # edge_info = edge_info.drop(
                #     columns=self.genetic_walks_black_list[(current_state, selected_v, selected_j)])
            # check selected path has genes
            if len(edge_info) == 0:
                if len(walk) > 2:
                    self.genetic_walks_black_list[(walk[-2], selected_v, selected_j)] \
                        = self.genetic_walks_black_list.get((walk[-2], selected_v, selected_j),
                                                            []) + [walk[-1]]
                    current_state = walk[-2]
                    walk = walk[:-1]
                else:
                    walk = walk[:1]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)

                continue

            w = np.array([edge_info[i]['weight'] for i in edge_info])
            w = w / w.sum()
            current_state = choice([*edge_info], w)
            walk.append(current_state)

        return walk, selected_v, selected_j

    def random_walk(self, initial_state=None, vj_init='marginal'):
        if initial_state is None:
            current_state = self._random_initial_state()
            walk = [current_state]
        else:
            current_state = initial_state
            walk = [initial_state]

        # while the walk is not in a valid final state
        while not self.is_stop_condition(current_state):
            # get the node_data for the current state
            edge_info = self._get_node_feature_info_df(current_state, 'weight', asdict=True)

            if (current_state) in self.genetic_walks_black_list:
                for col in self.genetic_walks_black_list[(current_state)]:
                    edge_info.pop(col)
                # edge_info = edge_info.drop(
                #     columns=self.genetic_walks_black_list[(current_state, selected_v, selected_j)])
            # check selected path has genes
            if len(edge_info) == 0:
                if len(walk) > 2:
                    self.genetic_walks_black_list[(walk[-2])] \
                        = self.genetic_walks_black_list.get((walk[-2]),
                                                            []) + [walk[-1]]
                    current_state = walk[-2]
                    walk = walk[:-1]
                else:
                    walk = walk[:1]
                    current_state = walk[0]
                    selected_v, selected_j = self._select_random_vj_genes(vj_init)

                continue

            w = np.array([edge_info[i]['weight'] for i in edge_info])
            w = w / w.sum()
            current_state = choice([*edge_info], w)
            walk.append(current_state)

        return walk

    def _derive_subpattern_individual_probability(self):
        weight_df = pd.Series(nx.get_edge_attributes(self.graph, 'weight')).reset_index()
        self.subpattern_individual_probability = weight_df.groupby('level_0').sum(numeric_only=True).rename(columns={0: 'proba'})
        self.subpattern_individual_probability.proba /= self.subpattern_individual_probability.proba.sum()

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

    def random_step(self, state):
        """
           Given the current state, pick and take a random step based on the translation probabilities
           :param state:
           :return:
                       """
        states, probabilities = self._get_state_weights(state)
        return choice(states, probabilities)

    def _random_initial_state(self):
        """
       Select a random initial state based on the marginal distribution of initial states.
       :return:
       """
        return choice(self.initial_states_probability.index, self.initial_states_probability.values)

    def _select_random_vj_genes(self, type='marginal'):
        self._raise_genetic_mode_error()
        if type == 'marginal':
            V = choice(self.marginal_vgenes.index, self.marginal_vgenes.values)
            J = choice(self.marginal_jgenes.index, self.marginal_jgenes.values)
            return V, J
        elif type == 'combined':
            VJ = choice(self.vj_probabilities.index, self.vj_probabilities.values)
            V, J = VJ.split('_')
            return V, J

    def _insert_edge_and_information(self, A_, B_, Vgene, Jgene):
        # if self.graph.has_edge(A_, B_):
        try:  # assuming edge exists
            edge_pointer = self.graph[A_][B_]
            edge_pointer["weight"] += 1
            edge_pointer[Vgene] = edge_pointer.get(Vgene, 0) + 1
            edge_pointer[Jgene] = edge_pointer.get(Jgene, 0) + 1
            edge_pointer['Vsum'] += 1
            edge_pointer['Jsum'] += 1
        except KeyError as e:  # edge not fount
            attr = {'weight': 1, 'Vsum': 1, 'Jsum': 1}
            attr[Vgene] = 1
            attr[Jgene] = 1
            self.graph.add_edge(A_, B_, **attr)

        self.n_transitions += 1

    def _insert_edge_and_information_no_genes(self, A_, B_):
        if self.graph.has_edge(A_, B_):
            self.graph[A_][B_]["weight"] += 1
        else:
            self.graph.add_edge(A_, B_, weight=1)
        self.n_transitions += 1

    def _get_state_weights(self, node, v=None, j=None):
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

    def _batch_gene_weight_normalization(self, n_process=3, verbose=False):
        batches = chunkify(list(self.graph.edges), len(self.graph.edges) // 3)
        with ThreadPool(n_process) as pool:
            pool.map(self._normalize_gene_weights, list(batches))
        # self.normalize_gene_weights(self.graph.edges)

    def _normalize_gene_weights(self, edge_list):
        for n_a, n_b in (edge_list):
            e_data = self.graph.get_edge_data(n_a, n_b)
            vsum = e_data['Vsum']
            jsum = e_data['Jsum']
            genes = set(e_data) - {'Vsum', 'Jsum', 'weight'}

            for key in genes:
                if 'V' in key:
                    self.graph[n_a][n_b][key] /= vsum
                else:
                    self.graph[n_a][n_b][key] /= jsum

    def _update_terminal_states(self, terminal_state):
        self.terminal_states[terminal_state] = self.terminal_states.get(terminal_state, 0) + 1

    def _update_initial_states(self, initial_state):
        self.initial_states[initial_state] = self.initial_states.get(initial_state, 0) + 1

    def _load_gene_data(self, data):
        self.observed_vgenes = list(set(data['V']))
        self.observed_jgenes = list(set(data['J']))

        self.marginal_vgenes = data['V'].value_counts()
        self.marginal_jgenes = data['J'].value_counts()
        self.marginal_vgenes /= self.marginal_vgenes.sum()
        self.marginal_jgenes /= self.marginal_jgenes.sum()

        self.vj_probabilities = (data['V'] + '_' + data['J']).value_counts()
        self.vj_probabilities /= self.vj_probabilities.sum()

    def _derive_terminal_state_map(self):
        """
        create a matrix map between all terminal state,
        given that we have  K terminal states, the matrix will be of dim KxK
        where at each row reachability will be denoted by 1, i.e
        if I can reach e  state K_i from state k, the value at K[k][K_i] = 1
        :return:
        """
        terminal_state_map = np.zeros((len(self.terminal_states), len(self.terminal_states)))
        ts_index = {i: ax for ax, i in enumerate(self.terminal_states.index)}

        for pos_1, terminal_1 in enumerate(self.terminal_states.index):
            dfs_node = list(nx.dfs_preorder_nodes(self.graph, source=terminal_1))
            # for pos_2, terminal_2 in enumerate(self.terminal_states.index):
            #     terminal_state_map[pos_1][pos_2] = nx.has_path(self.graph, source=terminal_1, target=terminal_2)
            reachable_terminal_state = set(dfs_node) & set(self.terminal_states.index)
            for node in reachable_terminal_state:
                terminal_state_map[pos_1][ts_index[node]] = 1

        terminal_state_map = pd.DataFrame(terminal_state_map,
                                          columns=self.terminal_states.index,
                                          index=self.terminal_states.index).apply(
            lambda x: x.apply(lambda y: x.name if y == 1 else np.nan), axis=0)
        # np.fill_diagonal(terminal_state_map.values, np.nan)

        self.terminal_state_map = pd.Series(terminal_state_map.apply(lambda x: (x.dropna().to_list()), axis=1),
                                            index=self.terminal_states.index)

    def _derive_stop_probability_data(self):
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

    def _length_specific_terminal_state(self, length):

        return self.terminal_states[
            self.terminal_states.index.to_series().str.split('_').apply(lambda x: int(x[-1])) == length].index.to_list()

    def _max_sum_gene_prediction(self, walk, top_n=1):
        v_gene_agg = dict()
        j_gene_agg = dict()
        for i in range(0, len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ls = self.graph.get_edge_data(walk[i], walk[i + 1]).copy()
                for key in {*ls} - {'weight', 'Vsum', 'Jsum'}:
                    if 'V' in key:
                        v_gene_agg[key] = v_gene_agg.get(key, 0) + ls[key]
                    else:
                        j_gene_agg[key] = j_gene_agg.get(key, 0) + ls[key]

        if top_n == 1:
            return max(v_gene_agg, key=v_gene_agg.get), max(j_gene_agg, key=j_gene_agg.get)
        else:
            vs = {k for k in heapq.nlargest(top_n, v_gene_agg, key=lambda k: v_gene_agg[k])}
            js = {k for k in heapq.nlargest(top_n, j_gene_agg, key=lambda k: j_gene_agg[k])}
            return vs, js

    def _max_product_gene_prediction(self, walk, top_n=1):
        v_gene_agg = dict()
        j_gene_agg = dict()
        for i in range(0, len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ls = self.graph.get_edge_data(walk[i], walk[i + 1]).copy()
                for key in {*ls} - {'weight', 'Vsum', 'Jsum'}:
                    if 'V' in key:
                        v_gene_agg[key] = v_gene_agg.get(key, 1) * ls[key]
                    else:
                        j_gene_agg[key] = j_gene_agg.get(key, 1) * ls[key]

        if top_n == 1:
            return max(v_gene_agg, key=v_gene_agg.get), max(j_gene_agg, key=j_gene_agg.get)

        else:
            vs = {k for k in heapq.nlargest(top_n, v_gene_agg, key=lambda k: v_gene_agg[k])}
            js = {k for k in heapq.nlargest(top_n, j_gene_agg, key=lambda k: j_gene_agg[k])}
            return vs, js

    def _sampling_gene_prediction(self, walk, top_n=1):
        V = []
        J = []
        for i in range(0, len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ls = self.graph.get_edge_data(walk[i], walk[i + 1]).copy()
                vkeys = dict()
                jkeys = dict()
                for k in ls:
                    if k not in {'weight', 'Vsum', 'Jsum'}:
                        if 'V' in k:
                            vkeys[k] = ls[k]
                        else:
                            jkeys[k] = ls[k]

                for _ in range(25):
                    V.append(choice(list(vkeys.keys()), list(vkeys.values())))
                    J.append(choice(list(jkeys.keys()), list(jkeys.values())))
        vcounter = Counter(V)
        jcounter = Counter(J)

        if top_n == 1:

            return vcounter.most_common(1)[0][0], jcounter.most_common(1)[0][0]


        else:
            vs = [i[0] for i in vcounter]
            js = [i[0] for i in jcounter]
            return vs, js

    def _full_appearance_gene_prediction(self, walk, alpha=0):
        vgenes = list()
        jgenes = list()
        wl = len(walk) - (1 + alpha)
        for i in range(0, wl):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ls = self.graph.get_edge_data(walk[i], walk[i + 1]).copy()
                vgenes += [i for i in ({*ls} - {'weight', 'Vsum', 'Jsum'}) if 'V' in i]
                jgenes += [i for i in ({*ls} - {'weight', 'Vsum', 'Jsum'}) if 'J' in i]

        vcount = Counter(vgenes)
        jcount = Counter(vgenes)
        return [w for w in vcount if vcount[w] == wl], [w for w in jcount if jcount[w] == wl]

    def _raise_genetic_mode_error(self):
        if not self.genetic:
            raise Exception('Genomic Data Function Requires Gene Annotation Data, The Graph Has No Such Data')

    def predict_vj_genes(self, walk, top_n=1, mode='max', alpha=0):

        if mode == 'max_sum':
            return self._max_sum_gene_prediction(walk, top_n=top_n)
        elif mode == 'max_product':
            return self._max_product_gene_prediction(walk, top_n=top_n)
        elif mode == 'sampling':
            return self._sampling_gene_prediction(walk, top_n=top_n)
        elif mode == 'full':
            return self._full_appearance_gene_prediction(walk, alpha)

    def eigenvector_centrality(self,max_iter=500):
        return nx.algorithms.eigenvector_centrality(self.graph, weight='weight',max_iter=max_iter)

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

    def is_dag(self):
        """
           the function checks whether the graph is a Directed acyclic graph

               :return:
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
                          the function will return a pandas DataFrame containing the graphs
                            Chromatic Number,Number of Isolates,Max In Deg,Max Out Deg,Number of Edges
                        """
        R = pd.Series({
            'Chromatic Number': max(nx.greedy_color(self.graph).values()) + 1,
            'Number of Isolates': nx.number_of_isolates(self.graph),
            'Max In Deg': max(dict(self.graph.in_degree).values()),
            'Max Out Deg': max(dict(self.graph.out_degree).values()),
            'Number of Edges': len(self.graph.edges),
        })
        return R

    def voterank(self, n_nodes=25):
        """
                         Uses the VoteRank algorithm to return the top N influential nodes in the graph, where N is equal to n_nodes

                                  Parameters:
                                          n_nodes (int): the number of most influential nodes to find

                                  Returns:
                                          list : a list of top influential nodes
                        """
        return nx.algorithms.voterank(self.graph, number_of_nodes=n_nodes)
