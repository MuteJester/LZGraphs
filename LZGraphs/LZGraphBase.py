from time import time
from multiprocessing.pool import ThreadPool
import networkx as nx
import pandas as pd

from .misc import chunkify, window
class LZGraphBase:
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
        self.lengths = []
        self.cac_graphs = dict()
        self.n_transitions = 0
        self.n_neighbours = dict()

        # per node observed frequency for unity operation
        self.per_node_observed_frequency = dict()


# TO DO OPTIMIZE NORMALIZATION !
    def _normalize_edge_weights(self):
        # normalize edges
        # weight_df = pd.Series(nx.get_edge_attributes(self.graph, 'weight')).reset_index()
        # for idx, group in weight_df.groupby('level_0'):
        #     weight_df.loc[group.index, 0] /= group[0].sum()
        # nx.set_edge_attributes(self.graph, weight_df.set_index(['level_0', 'level_1']).to_dict()[0], 'weight')

        for edge_a,edge_b in self.graph.edges:
            node_observed_total = self.per_node_observed_frequency[edge_a]
            self.graph[edge_a][edge_b]['weight']/=node_observed_total



    def _batch_gene_weight_normalization(self, n_process=3, verbose=False):
        batches = chunkify(list(self.graph.edges), len(self.graph.edges) // 3)
        pool = ThreadPool(n_process)
        pool.map(self._normalize_gene_weights, list(batches))
        #self.normalize_gene_weights(self.graph.edges)
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
    def _update_terminal_states(self,terminal_state):
        self.terminal_states[terminal_state] = self.terminal_states.get(terminal_state,0)+1
    def _update_initial_states(self,initial_state):
        self.initial_states[initial_state] = self.initial_states.get(initial_state,0)+1
    def _load_gene_data(self, data):
        self.observed_vgenes = list(set(data['V']))
        self.observed_jgenes = list(set(data['J']))

        self.marginal_vgenes = data['V'].value_counts()
        self.marginal_jgenes = data['J'].value_counts()
        self.marginal_vgenes /= self.marginal_vgenes.sum()
        self.marginal_jgenes /= self.marginal_jgenes.sum()

        self.vj_probabilities = (data['V'] + '_' + data['J']).value_counts()
        self.vj_probabilities /= self.vj_probabilities.sum()

    def eigenvector_centrality(self):
        return nx.algorithms.eigenvector_centrality(self.graph, weight='weight')
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
