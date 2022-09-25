from itertools import product
import networkx as nx
import numpy as np
import pandas as pd
from src.LZGraph.misc import chunkify, window
from src.LZGraph.decomposition import lempel_ziv_decomposition


def saturation_function(x, h, k):
    return 1 / (1 + ((h / x) ** k))


wfs = lambda x, y, z: 1 - z['weight']


def generate_dictionary(max_len):
    """
    generates all a set of all k-mers for k values from 1 to max_len
    :param max_len:
    :return:
    """

    N = max_len
    DICT = []
    for i in range(1, N + 1):
        DICT += [''.join(i) for i in product(['A', 'T', 'G', 'C'], repeat=i)]

    return DICT


class NaiveLZGraph:
    def __init__(self, cdr3_list, dictionary, verbose=False):
        """

        :param cdr3_list: a list of nucleotide sequence
        :param dictionary: a list of all sub-patterns to manifest as nodes
        :param verbose:
        """

        self.dictionary = dictionary
        lz_components = []
        self.n_subpatterns = 0
        self.terminal_states = []
        self.initial_states = []
        # extract lz components
        for cdr3 in (cdr3_list):
            LZ = lempel_ziv_decomposition(cdr3)
            lz_components.append(LZ)
            self.terminal_states.append(LZ[-1])
            self.initial_states.append(LZ[0])

            self.n_subpatterns += len(lz_components[-1])

        self.final_state = pd.Series(self.terminal_states).value_counts()
        # self.final_state = self.final_state[self.final_state >= (self.final_state.mean())]
        self.initial_states = pd.Series(self.initial_states).value_counts()

        if verbose == True:
            print('Extracted Positions and LZ Components...')

        # create graph
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.dictionary)
        # add node subpattern length

        self.n_transitions = 0
        for subpattern in (lz_components):
            steps = (window(subpattern, 2))
            for (A, B) in (steps):
                A_ = A
                B_ = B
                if self.graph.has_edge(A_, B_):
                    self.graph[A_][B_]["weight"] += 1

                else:
                    self.graph.add_edge(A_, B_, weight=1)
                self.n_transitions += 1

        if verbose == True:
            print('Created Graph...')
            # normalize edges
        weight_df = pd.Series(nx.get_edge_attributes(self.graph, 'weight')).reset_index()
        self.subpattern_individual_probability = weight_df.groupby('level_0').sum().rename(columns={0: 'proba'})
        self.subpattern_individual_probability /= self.subpattern_individual_probability.proba.sum()

        for idx, group in weight_df.groupby('level_0'):
            weight_df.loc[group.index, 0] /= group[0].sum()
        # weight_df.set_index(['level_0','level_1']).to_dict()[0]
        nx.set_edge_attributes(self.graph, weight_df.set_index(['level_0', 'level_1']).to_dict()[0], 'weight')

        if verbose == True:
            print('Normalized Weights...')

        self.length_distribution_proba = self.final_state / self.final_state.sum()
        self.__derive_terminal_state_map()
        self.derive_final_state_data()

    def isolates(self):
        return list(nx.isolates(self.graph))

    def drop_isolates(self):
        self.graph.remove_nodes_from(self.isolates())

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def __derive_terminal_state_map(self):
        terminal_state_map = np.zeros((len(self.final_state), len(self.final_state)))
        for pos_1, terminal_1 in enumerate(self.final_state.index):
            for pos_2, terminal_2 in enumerate(self.final_state.index):
                terminal_state_map[pos_1][pos_2] = nx.has_path(self.graph, source=terminal_1, target=terminal_2)
        terminal_state_map = pd.DataFrame(terminal_state_map,
                                          columns=self.final_state.index,
                                          index=self.final_state.index).apply(
            lambda x: x.apply(lambda y: x.name if y == 1 else np.nan), axis=0)
        # np.fill_diagonal(terminal_state_map.values, np.nan)

        self.terminal_state_map = pd.Series(terminal_state_map.apply(lambda x: (x.dropna().to_list()), axis=1),
                                            index=self.final_state.index)

    def derive_final_state_data(self):
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

        :param steps: number of sub-patterns the output should contain
        :param initial_state:
        :param final_states:
        :return:
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

            if current_state in self.final_state and len(value) == steps and len(seq) % 3 == 0:
                return value, seq
            elif len(value) == steps and current_state not in self.final_state:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])
            elif len(value) == steps and current_state in self.final_state and len(seq) % 3 != 0:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])

        return value, seq

    def random_walk_ber_shortest(self, steps, sfunc_h=0.6, sfunc_k=12):
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

            if current_state in self.final_state and len(value) == steps and len(seq) % 3 == 0:
                return value, seq
            elif len(value) == steps and current_state not in self.final_state:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])
            elif len(value) == steps and current_state in self.final_state and len(seq) % 3 != 0:
                value = value[:np.random.randint(1, len(value), 1)[0]]
                tolorance += 1
                current_state = value[-1]
                seq = ''.join([i for i in value])

        return value, seq

        # return self.random_walk(steps, initial_state, final_states,tolorance+1)

    def __get_state_weights(self, node, v=None, j=None):
        if v is None and j is None:
            node_data = self.graph[node]
            states = list(node_data.keys())
            probabilities = [node_data[i]['weight'] for i in states]
            return states, probabilities
        else:
            return pd.DataFrame(dict(self.graph[node])).T

    def is_stop_condition(self, state, selected_gene_path_v=None, selected_gene_path_j=None):
        if state not in self.final_state:
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
        states, probabilities = self.__get_state_weights(state)
        return np.random.choice(states, size=1, p=probabilities).item()

    def __random_initial_state(self):
        first_states = self.initial_states / self.initial_states.sum()
        return np.random.choice(first_states.index, size=1, p=first_states.values)[0]

    def unsupervised_random_walk(self):
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
        return nx.algorithms.eigenvector_centrality(self.graph, weight='weight')

    def voterank(self, n_nodes=25):
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
        R = pd.Series({
            'Chromatic Number': max(nx.greedy_color(self.graph).values()) + 1,
            'Number of Isolates': nx.number_of_isolates(self.graph),
            'Max In Deg': max(dict(self.graph.in_degree).values()),
            'Max Out Deg': max(dict(self.graph.out_degree).values()),
            'Number of Edges': len(self.graph.edges)
        })
        return R
