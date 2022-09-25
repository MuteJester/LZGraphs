#Nucleotide Double Positional
from multiprocessing.pool import ThreadPool

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.LZGraph.misc import chunkify, window
from tqdm.auto import tqdm
import re
from src.LZGraph.decomposition import lempel_ziv_decomposition
import seaborn as sns

def get_lz_and_pos(cdr3):
    lzc = lempel_ziv_decomposition(cdr3)
    cumlen = np.cumsum([len(i) for i in lzc])
    pos = []
    locations = []
    aux = 0
    for i, ll in zip(lzc, cumlen):
        pos.append((ll - len(i)) % 3)
        locations.append(aux)
        aux += 1

    return lzc, pos, cumlen


def clean_node(base):
    return re.search(r'[ATGC]*', base).group()


def encode_sequence(cdr3):
    lz, pos, loc = get_lz_and_pos(cdr3)
    return list(map(lambda x, y, z: x + str(y) + '_' + str(z), lz, pos, loc))


class NDPLZGraph:
    def __init__(self, data, verbose=False, dictionary=None):

        """

        :param data: a padnas dataframe with 1 mandatory column "cdr3_rearrangement" which is all the cdr3 neuclitode
        sequences , optinaly genes can be added to graph for gene inference via adding a "V" and "J" column
        :param verbose:
        :param dictionary:
        """

        cdr3_list = data['cdr3_rearrangement']
        lz_components, lz_positions, lz_locations = [], [], []
        self.genetic = True if 'V' in data.columns and 'J' in data.columns else False
        self.genetic_walks_black_list = None
        self.n_subpatterns = 0
        self.initial_states, self.terminal_states = [], []

        dictionary_flag = True if dictionary is None else False
        dictionary = set() if dictionary is None else dictionary

        self.__load_gene_data(data)

        # extract lz components
        for cdr3 in tqdm(cdr3_list, leave=False):
            LZ, POS, locations = get_lz_and_pos(cdr3)
            dictionary = dictionary | set(map(lambda x, y, z: x + str(y) + '_' + str(z), LZ, POS, locations))
            lz_components.append(LZ)
            lz_positions.append(POS)
            lz_locations.append(locations)
            self.terminal_states.append(LZ[-1] + str(POS[-1]) + '_' + str(locations[-1]))
            self.initial_states.append(LZ[0] + str(POS[0]) + '_' + str(locations[0]))
            self.n_subpatterns += len(lz_components[-1])

        self.final_state = pd.Series(self.terminal_states).value_counts()
        self.initial_states = pd.Series(self.initial_states).value_counts()

        if dictionary_flag:
            self.dictionary = list(dictionary)

        if verbose == True:
            print('Extracted Positions and LZ Components...')

        # create graph
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.dictionary)
        # add node subpattern length

        self.n_transitions = 0


        if self.genetic:
            # Encode Subpattern{ReadingFramePosition}_{SequencePosition} and gene at each node
            for subpattern, positions, location, Vgene, Jgene in zip(lz_components, lz_positions, lz_locations, data['V'],
                                                                     data['J']):

                self.__process_edge_info_batch(subpattern,positions,location,Vgene,Jgene)
        else:
            for subpattern, positions, location in zip(lz_components, lz_positions, lz_locations):
                self.__process_edge_info_batch_no_genes(subpattern, positions, location)

        if verbose == True:
            print('Created Graph...')

        self.__normalize_edge_weights(verbose)
        if self.genetic:
            # Normalized Gene Weights
            self.__batch_gene_weight_normalization(3,verbose)


        self.length_distribution_proba = self.final_state / self.final_state.sum()
        self.edges_list = list(self.graph.edges(data=True))
        self.__derive_terminal_state_map()
        self.derive_final_state_data()
        self.train_pgen = np.array(
            [self.walk_probability(encode_sequence(i), verbose=False) for i in data['cdr3_rearrangement']])

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

    def __normalize_edge_weights(self,verbose=False):
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

    def __add_vgene_head_nodes(self,verbose=False):
        for vgene in self.observed_vgenes:
            self.graph.add_node(vgene)
            for fs in self.initial_states.index:
                self.graph.add_edge(vgene, fs, weight=self.marginal_vgenes[vgene])
        if verbose == True:
            print('Added V Gene States...')

    def __batch_gene_weight_normalization(self,n_process = 3,verbose=False):
        batches = chunkify(list(self.graph.edges),len(self.graph.edges) // 3)
        print('Starting MP...')
        pool = ThreadPool(n_process)
        pool.map(self.__normalize_gene_weights,list(batches))

    def __normalize_gene_weights(self,edge_list):
        for n_a, n_b in (edge_list):
            e_data = dict(self.graph.get_edge_data(n_a, n_b))
            genes = pd.Series(e_data).iloc[1:]
            vsum = genes.pop('Vsum')
            jsum = genes.pop('Jsum')

            for key in genes.index:
                if 'V' in key:
                    self.graph[n_a][n_b][key] /= vsum
                else:
                    self.graph[n_a][n_b][key] /= jsum
        #if verbose == True:
        print('Normalized Gene Frequency...')

    def __process_edge_info_batch(self,subpattern,positions,location,Vgene,Jgene):
        steps = (window(subpattern, 2))
        reading_frames = (window(positions, 2))
        locations = (window(location, 2))

        for (A, B), (pos_a, pos_b), (loc_a, loc_b) in zip(steps, reading_frames, locations):
            A_ = A + str(pos_a) + '_' + str(loc_a)
            B_ = B + str(pos_b) + '_' + str(loc_b)
            self.__insert_edge_and_information(A_, B_, Vgene, Jgene)
    def __process_edge_info_batch_no_genes(self,subpattern,positions,location):
        steps = (window(subpattern, 2))
        reading_frames = (window(positions, 2))
        locations = (window(location, 2))

        for (A, B), (pos_a, pos_b), (loc_a, loc_b) in zip(steps, reading_frames, locations):
            A_ = A + str(pos_a) + '_' + str(loc_a)
            B_ = B + str(pos_b) + '_' + str(loc_b)
            self.__insert_edge_and_information(A_, B_)

    def __insert_edge_and_information(self, A_, B_, Vgene, Jgene):
        if self.graph.has_edge(A_, B_):
            self.graph[A_][B_]["weight"] += 1

            if Vgene in self.graph[A_][B_]:
                self.graph[A_][B_][Vgene] += 1

            else:
                self.graph[A_][B_][Vgene] = 1

            if Jgene in self.graph[A_][B_]:
                self.graph[A_][B_][Jgene] += 1
            else:
                self.graph[A_][B_][Jgene] = 1
            self.graph[A_][B_]['Vsum'] += 1
            self.graph[A_][B_]['Jsum'] += 1
        else:
            self.graph.add_edge(A_, B_, weight=1)
            self.graph[A_][B_][Vgene] = 1
            self.graph[A_][B_][Jgene] = 1
            self.graph[A_][B_]['Vsum'] = 1
            self.graph[A_][B_]['Jsum'] = 1

        self.n_transitions += 1
    def __insert_edge_and_information_no_genes(self, A_, B_):
        if self.graph.has_edge(A_, B_):
            self.graph[A_][B_]["weight"] += 1
        else:
            self.graph.add_edge(A_, B_, weight=1)
        self.n_transitions += 1

    def __load_gene_data(self, data):
        self.observed_vgenes = list(set(data['V']))
        self.observed_jgenes = list(set(data['J']))

        self.marginal_vgenes = data['V'].value_counts()
        self.marginal_jgenes = data['J'].value_counts()
        self.marginal_vgenes /= self.marginal_vgenes.sum()
        self.marginal_jgenes /= self.marginal_jgenes.sum()

        self.vj_probabilities = (data['V'] + '_' + data['J']).value_counts()
        self.vj_probabilities /= self.vj_probabilities.sum()

    def isolates(self):
        return list(nx.isolates(self.graph))

    def drop_isolates(self):
        self.graph.remove_nodes_from(self.isolates())

    def is_dag(self):
        return nx.is_directed_acyclic_graph(self.graph)

    @property
    def nodes(self):
        return self.graph.nodes

    @property
    def edges(self):
        return self.graph.edges

    def walk_probability(self, walk, verbose=True):
        if type(walk) == str:
            LZ, POS = get_lz_and_pos(walk)
            walk_ = [i + str(j) for i, j in zip(LZ, POS)]
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

    def __get_state_weights(self, node, v=None, j=None):
        if v is None and j is None:
            node_data = self.graph[node]
            states = list(node_data.keys())
            probabilities = [node_data[i]['weight'] for i in states]
            return states, probabilities
        else:
            return pd.DataFrame(dict(self.graph[node])).T

    def __random_step(self, state):
        states, probabilities = self.__get_state_weights(state)
        return np.random.choice(states, size=1, p=probabilities).item()

    def __random_initial_state(self):
        initial_states = self.initial_states / self.initial_states.sum()
        return np.random.choice(initial_states.index, size=1, p=initial_states.values)[0]

    def __length_specific_terminal_state(self, length):
        return self.final_state[self.final_state.index.str.contains(f'{length}')].index.to_list()

    def __select_random_vj_genes(self, type='marginal'):
        if type == 'marginal':
            V = np.random.choice(self.marginal_vgenes.index, size=1, p=self.marginal_vgenes.values)[0]
            J = np.random.choice(self.marginal_jgenes.index, size=1, p=self.marginal_jgenes.values)[0]
            return V, J
        elif type == 'combined':
            VJ = np.random.choice(self.vj_probabilities.index, size=1, p=self.vj_probabilities.values)[0]
            V, J = VJ.split('_')
            return V, J

    def random_walk(self, seq_len, initial_state):
        current_state = initial_state
        walk = [initial_state]
        sequence = clean_node(initial_state)

        terminal_states = self.__length_specific_terminal_state(seq_len)

        if len(terminal_states) < 1:
            raise Exception('Unfamiliar Seq Length')

        while current_state not in terminal_states:
            states, probabilities = self.__get_state_weights(current_state)
            # Try add dynamic dictionary of weight that will remove invalid paths

            # if went into a final path with mismatch length
            if len(probabilities) == 0:  # no options we can take from here
                # go back to the last junction where a different choice can be made
                for ax in range(len(walk) - 1, 1, -1):
                    for final_s in terminal_states:
                        try:
                            SP = nx.dijkstra_path(self.graph, source=walk[ax], target=final_s,
                                                  weight=lambda x, y, z: 1 - z['weight'])
                            walk = walk[:ax] + SP
                            sequence = ''.join([clean_node(i) for i in walk])
                            return walk
                        except nx.NetworkXNoPath:
                            continue

            current_state = np.random.choice(states, size=1, p=probabilities).item()
            walk.append(current_state)
            sequence += clean_node(current_state)

        return walk

    def gene_random_walk(self, seq_len, initial_state, vj_init='marginal'):

        selected_gene_path_v, selected_gene_path_j = self.__select_random_vj_genes(vj_init)

        if seq_len == 'unsupervised':
            terminal_states = self.terminal_states
        else:
            terminal_states = self.__length_specific_terminal_state(seq_len)

        current_state = initial_state
        walk = [initial_state]

        # nodes not to consider due to invalidity
        if self.genetic_walks_black_list is None:
            self.genetic_walks_black_list = dict()


        # while the walk is not in a valid final state
        while current_state not in terminal_states:
            # print('Blacklist: ',blacklist)
            # print('='*30)
            # get the node_data for the current state
            edge_info = pd.DataFrame(dict(self.graph[current_state]))

            if (current_state,selected_gene_path_v,selected_gene_path_j) in self.genetic_walks_black_list:
                edge_info = edge_info.drop(columns=self.genetic_walks_black_list[(current_state,selected_gene_path_v,selected_gene_path_j)])
            # check selected path has genes
            if len(set(edge_info.index) & {selected_gene_path_v, selected_gene_path_j}) != 2:
                # TODO: add a visited node stack to not repeat the same calls and mistakes
                if len(walk) > 2:
                    self.genetic_walks_black_list[(walk[-2],selected_gene_path_v,selected_gene_path_j)]\
                        = self.genetic_walks_black_list.get((walk[-2],selected_gene_path_v,selected_gene_path_j), []) + [walk[-1]]
                    current_state = walk[-2]
                    walk = walk[:-1]
                else:
                    walk = walk[:1]
                    current_state = walk[0]
                    selected_gene_path_v, selected_gene_path_j = self.__select_random_vj_genes(vj_init)

                continue

            # get paths containing selected_genes
            idf = edge_info.T[[selected_gene_path_v, selected_gene_path_j]].dropna()
            w = edge_info.loc['weight', idf.index]
            w = w / w.sum()

            if len(w) == 0:
                if len(walk) > 2:
                    self.genetic_walks_black_list[(walk[-2],selected_gene_path_v,selected_gene_path_j)] = \
                        self.genetic_walks_black_list.get((walk[-2],selected_gene_path_v,selected_gene_path_j), []) + [walk[-1]]
                    current_state = walk[-2]
                    walk = walk[:-1]
                else:
                    walk = walk[:1]
                    current_state = walk[0]
                    selected_gene_path_v, selected_gene_path_j = self.__select_random_vj_genes(vj_init)

                continue

            # if len(w) == 0:  # no options we can take from here
            #     # go back to the last junction where a different choice can be made
            #     for ax in range(len(walk) - 1, 1, -1):
            #         for final_s in terminal_states:
            #             try:
            #                 SP = nx.dijkstra_path(self.graph, source=walk[ax], target=final_s,
            #                                       weight=lambda x, y, z: 1 - z['weight'])
            #                 walk = walk[:ax] + SP
            #                 sequence = ''.join([clean_node(i) for i in walk])
            #                 raise Exception(f' Ended After Selecting SP '+str(walk))
            #             except nx.NetworkXNoPath:
            #                 continue

            current_state = np.random.choice(w.index, size=1, p=w.values).item()
            walk.append(current_state)

        return walk, selected_gene_path_v, selected_gene_path_j

    def unsupervised_random_walk(self):
        random_initial_state = self.__random_initial_state()

        current_state = random_initial_state
        walk = [random_initial_state]
        sequence = clean_node(random_initial_state)

        while current_state not in self.terminal_states:
            # take a random step
            current_state = self.__random_step(current_state)

            walk.append(current_state)
            sequence += clean_node(current_state)
        return walk, sequence

    def walk_genes(self, walk,dropna=True):
        trans_genes = []
        columns = []
        for i in range(0, len(walk) - 1):
            ls = self.graph.get_edge_data(walk[i], walk[i + 1]).copy()
            columns.append(walk[i] + '->' + walk[i + 1])
            ls.pop('weight')
            ls.pop('Vsum')
            ls.pop('Jsum')

            trans_genes.append(pd.Series(ls))

        cc = pd.concat(trans_genes, axis=1)
        if dropna:
            cc = cc.dropna()
        if cc.shape[0] == 0:
            raise Exception('No Constant Gene Flow F')

        cc.columns = columns
        cc['type'] = ['v' if 'v' in x.lower() else 'j' for x in cc.index]
        cc['sum'] = cc.sum(axis=1, numeric_only=True)
        cc = cc.sort_values(by='sum', ascending=False)

        return cc

    def eigenvector_centrality(self):
        return nx.algorithms.eigenvector_centrality(self.graph, weight='weight')

    def voterank(self, n_nodes=25):
        return nx.algorithms.voterank(self.graph, number_of_nodes=n_nodes)
    def sequence_variation_curve(self,cdr3_sample):
        """
        given a sequence this function will return 2 list,
        the first is the lz-subpattern path through the graph and the second list is the number
        of possible choices that can be made at each sub-pattern
        :param cdr3_sample:
        :return:
        """
        encoded = encode_sequence(cdr3_sample)
        curve = [self.graph.out_degree(i) for i in encoded]
        return encoded,curve


    def path_gene_table(self,cdr3_sample,threshold=None):
        length = len(encode_sequence(cdr3_sample))

        if threshold is None:
            threshold = length * (1 / 4)
        gene_table = self.walk_genes(encode_sequence(cdr3_sample), dropna=False)
        gene_table = gene_table[gene_table.isna().sum(axis=1) < threshold]
        vgene_table = gene_table[gene_table.index.str.contains('V')]

        gene_table = self.walk_genes(encode_sequence(cdr3_sample), dropna=False)
        gene_table = gene_table[gene_table.isna().sum(axis=1) < (length * (1 / 2))]
        jgene_table = gene_table[gene_table.index.str.contains('J')]

        jgene_table = jgene_table.loc[jgene_table.isna().sum(axis=1).sort_values(ascending=True).index, :]
        vgene_table = vgene_table.loc[vgene_table.isna().sum(axis=1).sort_values(ascending=True).index, :]

        return vgene_table,jgene_table


    def path_gene_table_plot(self,cdr3_sample,threshold=None,figsize=None):
        vgene_table, jgene_table = self.path_gene_table(cdr3_sample,threshold)
        plt.figure(figsize=(15, 8) if figsize is None else figsize)
        plt.subplot(1, 2, 1)
        ax = sns.heatmap(jgene_table.iloc[:, :-2],
                         xticklabels=[clean_node(i.split('->')[0]) + '->' + clean_node(i.split('->')[1]) for i in
                                      jgene_table.columns[:-2]],
                         cmap='coolwarm', linewidths=3)
        ax.set_facecolor('xkcd:black')

        label_col_vals = jgene_table.iloc[:, :-2].isna().any(axis=1)
        for i in ax.get_yticklabels():
            if not label_col_vals[i.get_text()]:
                i.set_color("red")

        plt.subplot(1, 2, 2)
        ax = sns.heatmap(vgene_table.iloc[:, :-2],
                         xticklabels=[clean_node(i.split('->')[0]) + '->' + clean_node(i.split('->')[1]) for i in
                                      jgene_table.columns[:-2]],
                         cmap='coolwarm', linewidths=3, yticklabels=vgene_table.index)

        label_col_vals = vgene_table.iloc[:, :-2].isna().any(axis=1)
        for i in ax.get_yticklabels():
            if not label_col_vals[i.get_text()]:
                i.set_color("red")

        ax.set_facecolor('xkcd:black')
        plt.gcf().suptitle(cdr3_sample, fontsize=26)

        plt.tight_layout()
        plt.show()

    def gene_variation(self,cdr3):
        if not self.genetic:
            raise Exception('The LZGraph Has No Gene Data')
        encoded_a = encode_sequence(cdr3)
        nv_genes = [len(self.marginal_vgenes)]
        nj_genes = [len(self.marginal_jgenes)]
        for node in encoded_a[1:]:
            inedges = self.graph.in_edges(node)
            v = set()
            j = set()
            for ea, eb in inedges:
                genes = pd.Series(self.graph[ea][eb]).drop(index=['Vsum', 'Jsum', 'weight'])
                v = v | set(genes[genes.index.str.contains('V')].index)
                j = j | set(genes[genes.index.str.contains('J')].index)
            nv_genes.append(len(v))
            nj_genes.append(len(j))

        nj_genes = np.array(nj_genes)
        nv_genes = np.array(nv_genes)

        j_df = pd.DataFrame(
            {'genes': list(nv_genes) + list(nj_genes), 'type': ['V'] * len(nv_genes) + ['J'] * len(nj_genes),
             'sp': lempel_ziv_decomposition(cdr3) + lempel_ziv_decomposition(cdr3)})
        return j_df

    def gene_variation_plot(self,cdr3):
        j_df = self.gene_variation(cdr3)
        sns.barplot(data=j_df, x='sp', y='genes', hue='type')
        plt.grid(lw=2, ls=':', axis='y')
        plt.ylabel('LZ Sub_Pattern')
        plt.ylabel('Unqiue Gene Possibilities')
        plt.legend()
        plt.show()

    def graph_summary(self):
        R = pd.Series({
            'Chromatic Number': max(nx.greedy_color(self.graph).values()) + 1,
            'Number of Isolates': nx.number_of_isolates(self.graph),
            'Max In Deg': max(dict(self.graph.in_degree).values()),
            'Max Out Deg': max(dict(self.graph.out_degree).values()),
            'Number of Edges': len(self.graph.edges)
        })
        return R

