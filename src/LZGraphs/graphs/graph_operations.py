import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .naive import NaiveLZGraph
from ..utilities.helpers import restore_gene_counts, renormalize_edge_genes
from ..exceptions import IncompatibleGraphsError


__all__ = ['graph_union']


def graph_union(graphA, graphB):
    """ This function performs a union operation between two graphs, graphA will be updated to be the
    equivalent of the union of both.
    The result is logically equal to constructing a graph out of the union sequences, of two separate repertoires.

                 Args:
                     graphA (LZGraph): An LZGraph
                     graphB (LZGraph)  An LZGraph of the same class as graphA
                 Returns:
                     LZGraph: The resulting LZGraph from the union of graphA and graphB
           """


    if not isinstance(graphA, type(graphB)) and not isinstance(graphB, type(graphA)):
        raise IncompatibleGraphsError(
            type1=type(graphA).__name__,
            type2=type(graphB).__name__,
            message="Both graphs must be of the same type for union operation."
        )

    if isinstance(graphA, NaiveLZGraph):
        # Merge NaiveLZGraph: combine edges, states, and re-normalize
        for (A_, B_), data in graphB.graph.edges.items():
            weightB = data.get('weight', 0)
            # Convert normalized weight back to count
            freqB = graphB.per_node_observed_frequency.get(A_, 0)
            countB = weightB * freqB if freqB > 0 else 0

            if graphA.graph.has_edge(A_, B_):
                freqA = graphA.per_node_observed_frequency.get(A_, 0)
                countA = graphA.graph[A_][B_]['weight'] * freqA if freqA > 0 else 0
                graphA.graph[A_][B_]['weight'] = countA + countB
            else:
                if A_ not in graphA.graph:
                    graphA.graph.add_node(A_)
                if B_ not in graphA.graph:
                    graphA.graph.add_node(B_)
                graphA.graph.add_edge(A_, B_, weight=countB)

        # Merge per_node_observed_frequency
        for node, freq in graphB.per_node_observed_frequency.items():
            graphA.per_node_observed_frequency[node] = graphA.per_node_observed_frequency.get(node, 0) + freq

        # Merge initial and terminal states (they are pd.Series at this point)
        graphA.initial_states = graphA.initial_states.combine(graphB.initial_states, lambda x, y: x + y, fill_value=0)
        graphA.terminal_states = graphA.terminal_states.combine(graphB.terminal_states, lambda x, y: x + y, fill_value=0)
        graphA.length_distribution_proba = graphA.terminal_states / graphA.terminal_states.sum()

        # Re-normalize edge weights by per_node_observed_frequency
        for edge_a, edge_b in graphA.graph.edges:
            node_total = graphA.per_node_observed_frequency.get(edge_a, 1)
            if node_total > 0:
                graphA.graph[edge_a][edge_b]['weight'] /= node_total

        # Re-derive subpattern probabilities
        graphA._derive_subpattern_individual_probability()

        return graphA
    else:
        #graphA.genetic_walks_black_list.merge(graphB.genetic_walks_black_list if type(graphB.genetic_walks_black_list) is not None else {})
        graphA.n_subpatterns += graphB.n_subpatterns
        graphA.initial_states = graphA.initial_states.combine(graphB.initial_states, lambda x, y: x + y, fill_value=0)

        # not necceseray
        # lengths
        # observed_vgenes
        # observed_jgenes
        # dictionary
        # n_transitions
        # edges_list

        graphA.marginal_vgenes = (graphA.marginal_vgenes.combine(graphB.marginal_vgenes, lambda x, y: x + y,
                                                               fill_value=0)) / 2
        graphA.vj_probabilities = (graphA.vj_probabilities.combine(graphB.vj_probabilities, lambda x, y: x + y,
                                                                 fill_value=0)) / 2
        graphA.length_distribution = (
            graphA.length_distribution.combine(graphB.length_distribution, lambda x, y: x + y, fill_value=0))
        graphA.terminal_states = (graphA.terminal_states.combine(graphB.terminal_states, lambda x, y: x + y, fill_value=0))
        graphA.length_distribution_proba = (graphA.length_distribution_proba.combine(graphB.length_distribution_proba,
                                                                                   lambda x, y: x + y,
                                                                                   fill_value=0)) / 2
        graphA.subpattern_individual_probability = (graphA.subpattern_individual_probability.combine(
            graphB.subpattern_individual_probability, lambda x, y: x + y, fill_value=0)) / 2


        # recalculate
        #terminal_state_map
        #terminal_state_data

        union_graph = nx.digraph.DiGraph()
        union_graph.add_nodes_from(set(graphA.nodes) | set(graphB.nodes))


        for node in tqdm(set(graphA.nodes) | set(graphB.nodes)):
            if (node in graphA.per_node_observed_frequency) and (node in graphB.per_node_observed_frequency):
                h1 = pd.DataFrame(dict(graphA.graph[node]))
                h2 = pd.DataFrame(dict(graphB.graph[node]))

                if len(h1) == 0 and len(h2) == 0:
                    continue

                # renormalize weight
                h1_node_sum = graphA.per_node_observed_frequency[node]
                h2_node_sum = graphB.per_node_observed_frequency[node]

                if 'weight' in h1.index:
                    h1.loc['weight', :] *= h1_node_sum
                    h1 = h1.apply(restore_gene_counts)
                if 'weight' in h2.index:
                     h2.loc['weight', :] *= h2_node_sum
                     h2 = h2.apply(restore_gene_counts)


                factor = (h1_node_sum + h2_node_sum)

                c = h1.combine(h2, lambda x, y: x + y, fill_value=0).replace(0, np.nan)
                if factor > 0:
                    c.loc['weight', :] /= factor
                # gene renormalization
                c = c.apply(renormalize_edge_genes)
                c = c.round(10)

                # save new counts
                graphA.per_node_observed_frequency[node] = factor

                # update new graph
                for column in c.columns:
                    union_graph.add_edge(node, column)
                    for key in c.index:
                        union_graph[node][column][key] = c[column][key]
            elif node in graphA.per_node_observed_frequency and node not in graphB.per_node_observed_frequency:
                c = pd.DataFrame(dict(graphA.graph[node]))
                # save new counts
                for column in c.columns:
                    union_graph.add_edge(node, column)
                    for key in c.index:
                        union_graph[node][column][key] = c[column][key]
                continue
            elif node in graphB.per_node_observed_frequency:
                c = pd.DataFrame(dict(graphB.graph[node]))
                # save new counts
                graphA.per_node_observed_frequency[node] = graphB.per_node_observed_frequency[node]
                # update new graph
                for column in c.columns:
                    union_graph.add_edge(node, column)
                    for key in c.index:
                        union_graph[node][column][key] = c[column][key]

        graphA.graph = union_graph

        return graphA
