from .AAPLZGraph import AAPLZGraph
from .NaiveLZGraph import NaiveLZGraph
from .NDPLZGraph import NDPLZGraph


def graph_union(graphA,graphB):
    """
    This function performs a union operation between two graphs, graphA will be updated to be the
    equivalent of the union of both.
    The result is logically equal to constructing a graph out of the union sequences, of two separate repertoires.
    :param graphA:
    :param graphB:
    :return:
    """

    if type(graphA) != type(graphB):
        raise Exception('Both Graphs Must Be of Same Type!')

    if type(graphA) == NaiveLZGraph:
        pass
    else:
        graphA.genetic_walks_black_list.merge(graphB if type(graphB) is not None else {})
        graphA.n_subpatterns += graphB.n_subpatterns
        graphA.initial_states = graphA.initial_states.combine(graphB.initial_states, lambda x, y: x + y, fill_value=0)
        graphA.terminal_states = graphA.terminal_states + graphB.terminal_states

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
        graphA.final_state = (graphA.final_state.combine(graphB.final_state, lambda x, y: x + y, fill_value=0))
        graphA.length_distribution_proba = (graphA.length_distribution_proba.combine(graphB.length_distribution_proba,
                                                                                   lambda x, y: x + y,
                                                                                   fill_value=0)) / 2
        graphA.subpattern_individual_probability = (graphA.subpattern_individual_probability.combine(
            graphB.subpattern_individual_probability, lambda x, y: x + y, fill_value=0)) / 2


        # recalculate
        #terminal_state_map
        #terminal_state_data
