# Amino Acid Positional
import re
from multiprocessing.pool import ThreadPool

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .LZGraphBase import LZGraphBase
from .decomposition import lempel_ziv_decomposition
from tqdm.auto import tqdm
import seaborn as sns
from .misc import chunkify, window, choice
from time import time


def derive_lz_and_position(cdr3):
    lzc = lempel_ziv_decomposition(cdr3)
    aux = 0
    cumlen = []
    for sp in lzc:
        aux+=len(sp)
        cumlen.append(aux)
    return lzc, cumlen



def path_to_sequence(lz_subpatterns):
    return ''.join([AAPLZGraph.clean_node(i) for i in lz_subpatterns])


# TO DO: SET AGG LIST (LIKE LENGTHS) TOO NONE TO FREE UP MEMORY

class AAPLZGraph(LZGraphBase):
    """
              This class implements the logic and infrastructure of the "Amino Acid Positional" version of the LZGraph
              The nodes of this graph are LZ sub-patterns based on amino acids with added start position
              in the sequence, formally: {lz_subpattern}_{start position in sequence},
              This class best fits analysis and inference of amino acid sequences.

        Args:

              walk_probability(walk,verbose=True):
                  returns the PGEN of the given walk (list of sub-patterns)


              is_dag():
                the function checks whether the graph is a Directed acyclic graph

              walk_genes(walk,dropna=True):
                give a walk on the graph (a list of nodes) the function will return a table
                representing the possible genes and their probabilities at each edge of the walk.

              path_gene_table(cdr3_sample,threshold=None):
                the function will return two tables of all possible v and j genes
                that colud be used to generate the sequence given by "cdr3_sample"


              path_gene_table_plot(threshold=None,figsize=None):
                the function plots two heatmap, one for V genes and one for J genes,
                and represents the probability at each edge to select that gene,
                the color at each cell is equal to the probability of selecting the gene, a black
                cell means that the graph didn't see that gene used with that sub-pattern.

                the data used to create the charts can be derived by using the "path_gene_table" method.

              gene_variation(cdr3):
                given a sequence, this will derive a charts that shows the number of V and J genes observed
                per node (LZ- subpattern).

              gene_variation_plot(cdr3):
                Plots the data derived at the "gene_variation" method as two bar charts overlayed, one for V gene count
                and one for J gene count.


              random_walk(steps):
                 given a number of steps (sub-patterns) returns a random walk on the graph between a random inital state
                 to a random terminal state in the given number of steps

              gene_random_walk(seq_len, initial_state):
                given a target sequence length and an initial state, the function will select a random
                V and a random J genes from the observed gene frequency in the graph's "Training data" and
                generate a walk on the graph from the initial state to a terminal state while making sure
                at each step that both the selected V and J genes were seen used by that specific sub-pattern.

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


         Attributes:
                    nodes:
                        returns the nodes of the graph
                    edges:
                        return the edges of the graph


        """

    def __init__(self, data, verbose=True, calculate_trainset_pgen=False):
        """
        data has to be a pandas dataframe, the cdr3 amino acid sequence has to be under a column named
        "cdr3_amino_acid"
        and optionally you can add two columns "V" and "J" with the gene annotation for each sequence

        Args:
            data (pd.DataFrame): a dataframe containing the sequences for which to consturct an LZGraph and any
            additional V/J Data given provided under the "V" column and a "J" column.
            verbose
        """
        super().__init__()

        # check for V and J gene data in input
        self.genetic = True if type(data) == pd.DataFrame and 'V' in data.columns and 'J' in data.columns else False

        if self.genetic:
            self._load_gene_data(data)
            self.verbose_driver(0, verbose)

        # construct the graph while iterating over the data
        self.__simultaneous_graph_construction(data)
        self.verbose_driver(1, verbose)

        # convert to pandas series and  normalize
        self.length_distribution = pd.Series(self.lengths)
        self.terminal_states = pd.Series(self.terminal_states)
        self.initial_states = pd.Series(self.initial_states)
        self.length_distribution_proba = self.terminal_states / self.terminal_states.sum()
        self.initial_states = self.initial_states[self.initial_states > 5]
        self.initial_states_probability = self.initial_states/self.initial_states.sum()

        self.verbose_driver(2, verbose)

        self._derive_subpattern_individual_probability()
        self.verbose_driver(8, verbose)
        self._normalize_edge_weights()
        self.verbose_driver(3, verbose)

        if self.genetic:
            # Normalized Gene Weights
            self._batch_gene_weight_normalization(3, verbose)
            self.verbose_driver(4, verbose)

        self.edges_list = None
        self._derive_terminal_state_map()
        self.verbose_driver(7, verbose)
        self._derive_stop_probability_data()
        self.verbose_driver(8, verbose)
        self.verbose_driver(5, verbose)

        if calculate_trainset_pgen:
            self.train_pgen = np.array(
                [self.walk_probability(self.encode_sequence(i), verbose=False) for i in data.cdr3_amino_acid])

        self.constructor_end_time = time()
        self.verbose_driver(6, verbose)
        self.verbose_driver(-2, verbose)

    @staticmethod
    def encode_sequence(amino_acid):
        """
        This function will take a sequence and return it as LZ sub-patterns with added position
        the general format is given as {LZ-subpattern}_{start_index}

        Args:
            amino_acid (str)
        """
        lz, loc = derive_lz_and_position(amino_acid)
        return list(map(lambda x, z: x + '_' + str(z), lz, loc))
    @staticmethod
    def clean_node(base):
        """
        This Function will take in a sub-pattern that has position added to it and clean
        the added values returning only the amino acid value
        Args:
            base (str)
        """
        return re.search(r'[A-Z]*', base).group()

    def _decomposed_sequence_generator(self,data):
        if self.genetic:
            for cdr3,v,j in tqdm(zip(data['cdr3_amino_acid'],data['V'],data['J']), leave=False):

                LZ, locs = derive_lz_and_position(cdr3)

                steps = (window(LZ, 2))
                locations = (window(locs, 2))
                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + 1

                self._update_terminal_states(LZ[-1] + '_' + str(locs[-1]))
                self._update_initial_states(LZ[0] + '_1')
                yield steps,locations,v,j
        else:
            for cdr3 in tqdm(list(data), leave=False):
                LZ, locations_ = derive_lz_and_position(cdr3)
                steps = (window(LZ, 2))
                locations = (window(locations_, 2))

                self.lengths[len(cdr3)] = self.lengths.get(len(cdr3), 0) + 1
                self._update_terminal_states(LZ[-1] + '_' + str(locations_[-1]))
                self._update_initial_states(LZ[0] + '_1')
                yield steps,locations

    def __simultaneous_graph_construction(self, data):
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



    def walk_probability(self, walk, verbose=True, use_epsilon=False):
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
            LZ, POS = derive_lz_and_position(walk)
            walk_ = [i + str(j) for i, j in zip(LZ, POS)]
        else:
            walk_ = walk

        if walk_[0] not in self.subpattern_individual_probability['proba']:
            return np.finfo(float).eps ** 2
        proba = self.subpattern_individual_probability['proba'][walk_[0]]
        n_missing = 0
        total = 0
        for step1, step2 in window(walk_, 2):
            if self.graph.has_edge(step1, step2):
                proba *= self.graph.get_edge_data(step1, step2)['weight']
            else:
                if verbose:
                    print('No Edge Connecting| ', step1, '-->', step2)
                n_missing += 1
            total += 1

        if n_missing > 0:
            gmean = np.power(proba, (1 / total))
            proba = proba * (gmean ** n_missing)
        return proba

    def walk_gene_probability(self, walk, v, j, verbose=True, use_epsilon=False):
        if type(walk) == str:
            LZ, POS = derive_lz_and_position(walk)
            walk_ = [i + str(j) for i, j in zip(LZ, POS)]
        else:
            walk_ = walk

        proba_v = self.marginal_vgenes.loc[v]
        proba_j = self.marginal_jgenes.loc[j]
        for step1, step2 in window(walk_, 2):
            if self.graph.has_edge(step1, step2):
                proba_v *= self.graph.get_edge_data(step1, step2)[v]
                proba_j *= self.graph.get_edge_data(step1, step2)[j]
            else:
                if verbose:
                    print('No Edge Connecting| ', step1, '-->', step2)
                if use_epsilon:
                    return np.finfo(np.float64).eps
                else:
                    return 0
        return proba_v, proba_j

    # def random_walk(self, seq_len, initial_state):
    #     """
    #       given a number of steps (sub-patterns) returns a random walk on the graph between a random inital state
    #         to a random terminal state in the given number of steps
    #
    #
    #                  Parameters:
    #                          steps (int): number of sub-patterns the resulting walk should contain
    #                  Returns:
    #                          (list) : a list of LZ sub-patterns representing the random walk
    #                   """
    #     current_state = initial_state
    #     walk = [initial_state]
    #     sequence = clean_node(initial_state)
    #
    #     final_states = self._length_specific_terminal_state(seq_len)
    #
    #     if len(final_states) < 1:
    #         raise Exception('Unfamiliar Seq Length')
    #
    #     while current_state not in final_states:
    #         states, probabilities = self._get_state_weights(current_state)
    #         # Try add dynamic dictionary of weight that will remove invalid paths
    #
    #         # if went into a final path with mismatch length
    #         if len(probabilities) == 0:  # no options we can take from here
    #             # go back to the last junction where a different choice can be made
    #             for ax in range(len(walk) - 1, 1, -1):
    #                 for final_s in final_states:
    #                     try:
    #                         SP = nx.dijkstra_path(self.graph, source=walk[ax], target=final_s,
    #                                               weight=lambda x, y, z: 1 - z['weight'])
    #                         walk = walk[:ax] + SP
    #                         sequence = ''.join([clean_node(i) for i in walk])
    #                         return walk
    #                     except nx.NetworkXNoPath:
    #                         continue
    #
    #         current_state = np.random.choice(states, size=1, p=probabilities).item()
    #         walk.append(current_state)
    #         sequence += clean_node(current_state)
    #
    #     return walk


    def multi_gene_random_walk(self, N, seq_len, initial_state=None, vj_init='marginal'):

        selected_gene_path_v, selected_gene_path_j = self._select_random_vj_genes(vj_init)

        if seq_len == 'unsupervised':
            final_states = self.terminal_states.index.to_list().copy()
        else:
            final_states = self._length_specific_terminal_state(seq_len)

        # nodes not to consider due to invalidity
        if self.genetic_walks_black_list is None:
            self.genetic_walks_black_list = dict()

        results = []

        lengths = pd.Series(self.terminal_states).value_counts()
        max_length = lengths.idxmax()
        for _ in tqdm(range(N)):
            if initial_state is None:
                current_state = self._random_initial_state()
                walk = [current_state]
            else:
                current_state = initial_state
                walk = [initial_state]

            # while the walk is not in a valid final state
            while current_state not in lengths.index:
                # print('Blacklist: ',blacklist)
                # print('='*30)
                # get the node_data for the current state
                edge_info = pd.DataFrame(dict(self.graph[current_state]))

                if (current_state, selected_gene_path_v, selected_gene_path_j) in self.genetic_walks_black_list:
                    edge_info = edge_info.drop(columns=self.genetic_walks_black_list[
                        (current_state, selected_gene_path_v, selected_gene_path_j)])
                # check selected path has genes
                if len(set(edge_info.index) & {selected_gene_path_v, selected_gene_path_j}) != 2:
                    # TODO: add a visited node stack to not repeat the same calls and mistakes
                    if len(walk) > 2:
                        self.genetic_walks_black_list[(walk[-2], selected_gene_path_v, selected_gene_path_j)] \
                            = self.genetic_walks_black_list.get((walk[-2], selected_gene_path_v, selected_gene_path_j),
                                                                []) + [walk[-1]]
                        current_state = walk[-2]
                        walk = walk[:-1]
                    else:
                        walk = walk[:1]
                        current_state = walk[0]
                        selected_gene_path_v, selected_gene_path_j = self._select_random_vj_genes(vj_init)

                    continue

                # get paths containing selected_genes
                idf = edge_info.T[[selected_gene_path_v, selected_gene_path_j]].dropna()
                w = edge_info.loc['weight', idf.index]
                w = w / w.sum()

                if len(w) == 0:
                    if len(walk) > 2:
                        self.genetic_walks_black_list[(walk[-2], selected_gene_path_v, selected_gene_path_j)] = \
                            self.genetic_walks_black_list.get((walk[-2], selected_gene_path_v, selected_gene_path_j),
                                                              []) + [walk[-1]]
                        current_state = walk[-2]
                        walk = walk[:-1]
                    else:
                        walk = walk[:1]
                        current_state = walk[0]
                        selected_gene_path_v, selected_gene_path_j = self._select_random_vj_genes(vj_init)

                    continue

                current_state = np.random.choice(w.index, size=1, p=w.values).item()
                walk.append(current_state)

            results.append((walk, selected_gene_path_v, selected_gene_path_j))

            if walk[-1] in lengths.index and walk[-1] != max_length:  # [lengths <= lengths.max()].index:
                lengths[walk[-1]] -= 1
                if lengths[walk[-1]] < 0:
                    lengths.pop(walk[-1])

        return results


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
        random_initial_state = self._random_initial_state()

        current_state = random_initial_state
        walk = [random_initial_state]
        sequence = self.clean_node(random_initial_state)

        while not self.is_stop_condition(current_state):
            # take a random step
            current_state = self.random_step(current_state)

            walk.append(current_state)
            sequence += self.clean_node(current_state)
        return walk, sequence

    def walk_genes(self, walk, dropna=True,raise_error=True):
        """
               give a walk on the graph (a list of nodes) the function will return a table
                   representing the possible genes and their probabilities at each edge of the walk.
           Args:
            walk (list): a list of nodes representing a walk on the graph.
            dropna (bool): whether to drop the edges that are missing from the graph.
           """
        trans_genes = dict()
        for i in range(0, len(walk) - 1):
            if self.graph.has_edge(walk[i], walk[i + 1]):
                ls = self.graph.get_edge_data(walk[i], walk[i + 1]).copy()
                ls.pop('weight')
                ls.pop('Vsum')
                ls.pop('Jsum')

                trans_genes[walk[i] + '->' + walk[i + 1]] = ls

        cc = pd.DataFrame(trans_genes)

        if dropna:
            cc = cc.dropna()
        if cc.shape[0] == 0 and raise_error:
            raise Exception('No Constant Gene Flow F')

        cc['type'] = ['v' if 'v' in x.lower() else 'j' for x in cc.index]
        cc['sum'] = cc.sum(axis=1, numeric_only=True)
        #cc = cc.sort_values(by='sum', ascending=False)

        return cc

    def random_walk_distribution_based(self, length_distribution):
        N = length_distribution.sum()
        N = N * 3

        rwalks = []
        rseqs = []
        for _ in tqdm(range(N)):
            rw = self.unsupervised_random_walk()
            rwalks.append(rw[0])
            rseqs.append(rw[1])
        R = pd.DataFrame({'Seqs': rseqs, 'Walks': rwalks})
        R['L'] = R['Seqs'].str.len()

        samples = []
        for length in length_distribution.index:
            samples.append(R[R['L'] == length].sample(length_distribution[length]))
        return pd.concat(samples).iloc[:, :-1].values

    def get_gene_graph(self, v, j):
        to_drop = []
        if self.edges_list is None:
            self.edges_list = list(self.graph.edges(data=True))

        for edge in self.edges_list:
            if v in edge[2] and j in edge[2]:
                continue
            else:
                to_drop.append((edge[0], edge[1]))

        G = self.graph.copy()
        G.remove_edges_from(to_drop)
        G.remove_nodes_from(list(nx.isolates(G)))
        return G

    def cac_random_gene_walk(self, initial_state=None, vj_init='combined'):
        selected_gene_path_v, selected_gene_path_j = self._select_random_vj_genes(vj_init)

        if (selected_gene_path_v, selected_gene_path_j) not in self.cac_graphs:
            G = self.get_gene_graph(selected_gene_path_v, selected_gene_path_j)
            self.cac_graphs[(selected_gene_path_v, selected_gene_path_j)] = G
        else:
            G = self.cac_graphs[(selected_gene_path_v, selected_gene_path_j)]

        final_states = self.terminal_states.copy()
        final_states = list(set(final_states) & set(G.nodes))

        first_states = self.initial_states.copy()
        first_states = first_states.loc[list(set(first_states.index) & set(G.nodes))]
        first_states = (first_states / first_states.sum())

        current_state = np.random.choice(first_states.index, size=1, p=first_states.values)[0]
        walk = [current_state]

        # nodes not to consider due to invalidity
        if self.genetic_walks_black_list is None:
            self.genetic_walks_black_list = dict()

        # while the walk is not in a valid final state
        while current_state not in final_states:
            # get the node_data for the current state
            edge_info = pd.DataFrame(dict(G[current_state]))

            if (selected_gene_path_v, selected_gene_path_j, current_state) in self.genetic_walks_black_list:
                edge_info = edge_info.drop(
                    columns=self.genetic_walks_black_list[(selected_gene_path_v, selected_gene_path_j, current_state)])

            if edge_info.shape[1] == 0:
                self.genetic_walks_black_list[(selected_gene_path_v, selected_gene_path_j, walk[-2])] = \
                    self.genetic_walks_black_list.get((selected_gene_path_v, selected_gene_path_j, walk[-2]), []) + [
                        current_state]
                walk = walk[:-1]
                current_state = walk[-1]
                continue

            # get paths containing selected_genes
            idf = edge_info.T[[selected_gene_path_v, selected_gene_path_j]].dropna()
            w = edge_info.loc['weight', idf.index]
            w = w / w.sum()

            current_state = np.random.choice(w.index, size=1, p=w.values).item()
            walk.append(current_state)

        return walk, selected_gene_path_v, selected_gene_path_j

    def sequence_variation_curve(self, cdr3_sample):
        """
        given a sequence this function will return 2 list,
        the first is the lz-subpattern path through the graph and the second list is the number
        of possible choices that can be made at each sub-pattern
        :param cdr3_sample:
        :return:
        """
        encoded = self.encode_sequence(cdr3_sample)
        curve = [self.graph.out_degree(i) for i in encoded]
        return encoded, curve

    def path_gene_table(self, cdr3_sample, threshold=None):
        """
               the function will return two tables of all possible v and j genes
                   that colud be used to generate the sequence given by "cdr3_sample"
               :param cdr3_sample: a cdr3 sequence
               :param threshold: drop genes that are missing from threshold % of the sequence
               :return:
               """
        length = len(self.encode_sequence(cdr3_sample))

        if threshold is None:
            threshold = length * (1 / 4)
        gene_table = self.walk_genes(self.encode_sequence(cdr3_sample), dropna=False)
        gene_table = gene_table[gene_table.isna().sum(axis=1) < threshold]
        vgene_table = gene_table[gene_table.index.str.contains('V')]

        gene_table = self.walk_genes(self.encode_sequence(cdr3_sample), dropna=False)
        gene_table = gene_table[gene_table.isna().sum(axis=1) < (length * (1 / 2))]
        jgene_table = gene_table[gene_table.index.str.contains('J')]

        jgene_table = jgene_table.loc[jgene_table.isna().sum(axis=1).sort_values(ascending=True).index, :]
        vgene_table = vgene_table.loc[vgene_table.isna().sum(axis=1).sort_values(ascending=True).index, :]

        return vgene_table, jgene_table

    def gene_variation(self, cdr3):
        """
               Plots the data derived at the "gene_variation" method as two bar charts overlayed, one for V gene count
                   and one for J gene count.
               :param cdr3:
               :return:
               """
        if not self.genetic:
            raise Exception('The LZGraph Has No Gene Data')
        encoded_a = self.encode_sequence(cdr3)
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

