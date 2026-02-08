import numpy as np
import pandas as pd

from .edge_data import EdgeData
from ..exceptions import IncompatibleGraphsError


__all__ = ['graph_union']


def graph_union(graphA, graphB):
    """Perform a union operation between two graphs.

    graphA will be updated in-place to be the equivalent of the union
    of both. The result is logically equal to constructing a graph from
    the combined sequences of two separate repertoires.

    Since EdgeData stores raw counts as the source of truth, the union
    simply merges counts and then recalculates all derived probabilities.

    Args:
        graphA (LZGraph): An LZGraph (will be modified in-place).
        graphB (LZGraph): An LZGraph of the same class as graphA.

    Returns:
        LZGraph: graphA, updated with the union of both graphs.
    """
    if not isinstance(graphA, type(graphB)) and not isinstance(graphB, type(graphA)):
        raise IncompatibleGraphsError(
            type1=type(graphA).__name__,
            type2=type(graphB).__name__,
            message="Both graphs must be of the same type for union operation."
        )

    # 1. Merge edges (raw counts)
    for a, b in graphB.graph.edges:
        ed_b = graphB.graph[a][b]['data']
        if graphA.graph.has_edge(a, b):
            graphA.graph[a][b]['data'].merge(ed_b)
        else:
            # Ensure nodes exist
            if a not in graphA.graph:
                graphA.graph.add_node(a)
            if b not in graphA.graph:
                graphA.graph.add_node(b)
            # Deep copy EdgeData from B
            ed_new = EdgeData()
            ed_new.merge(ed_b)
            graphA.graph.add_edge(a, b, data=ed_new)

    # Also add any nodes from B that have no edges
    for node in graphB.graph.nodes:
        if node not in graphA.graph:
            graphA.graph.add_node(node)

    # 2. Merge sequence-level counts
    # (per_node_observed_frequency is recomputed in recalculate())
    graphA.initial_states = graphA.initial_states.combine(
        graphB.initial_states, lambda x, y: x + y, fill_value=0
    )
    graphA.terminal_states = graphA.terminal_states.combine(
        graphB.terminal_states, lambda x, y: x + y, fill_value=0
    )
    graphA.n_subpatterns += graphB.n_subpatterns
    graphA.n_transitions += graphB.n_transitions

    # Merge lengths
    if hasattr(graphB, 'lengths'):
        for length, count in graphB.lengths.items():
            graphA.lengths[length] = graphA.lengths.get(length, 0) + count

    # 4. Merge gene-level data (if genetic)
    if graphA.genetic and graphB.genetic:
        # Weighted average of marginal gene distributions
        nA = graphA.initial_states.sum()
        nB = graphB.initial_states.sum()
        nTotal = nA + nB
        if nTotal > 0:
            graphA.marginal_vgenes = (
                graphA.marginal_vgenes.combine(graphB.marginal_vgenes,
                    lambda x, y: x * nA / nTotal + y * nB / nTotal, fill_value=0)
            )
            graphA.marginal_jgenes = (
                graphA.marginal_jgenes.combine(graphB.marginal_jgenes,
                    lambda x, y: x * nA / nTotal + y * nB / nTotal, fill_value=0)
            )
            graphA.vj_probabilities = (
                graphA.vj_probabilities.combine(graphB.vj_probabilities,
                    lambda x, y: x * nA / nTotal + y * nB / nTotal, fill_value=0)
            )

        # Merge length_distribution counts
        if hasattr(graphA, 'length_distribution') and hasattr(graphB, 'length_distribution'):
            graphA.length_distribution = graphA.length_distribution.combine(
                graphB.length_distribution, lambda x, y: x + y, fill_value=0
            )

        # Merge observed gene sets
        if hasattr(graphB, 'observed_vgenes'):
            graphA.observed_vgenes = list(
                set(graphA.observed_vgenes) | set(graphB.observed_vgenes)
            )
        if hasattr(graphB, 'observed_jgenes'):
            graphA.observed_jgenes = list(
                set(graphA.observed_jgenes) | set(graphB.observed_jgenes)
            )

    # 5. Recalculate ALL derived state from raw counts
    graphA.recalculate()

    # Clear cached edges list
    if hasattr(graphA, 'edges_list'):
        graphA.edges_list = None

    return graphA
