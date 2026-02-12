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
    # (node_outgoing_counts is recomputed in recalculate())
    for k, v in graphB.initial_state_counts.items():
        graphA.initial_state_counts[k] = graphA.initial_state_counts.get(k, 0) + v
    for k, v in graphB.terminal_state_counts.items():
        graphA.terminal_state_counts[k] = graphA.terminal_state_counts.get(k, 0) + v
    graphA.num_subpatterns += graphB.num_subpatterns
    graphA.num_transitions += graphB.num_transitions

    # Merge lengths
    if hasattr(graphB, 'lengths'):
        for length, count in graphB.lengths.items():
            graphA.lengths[length] = graphA.lengths.get(length, 0) + count

    # 4. Merge gene-level data (if genetic)
    if graphA.has_gene_data and graphB.has_gene_data:
        # Weighted average of marginal gene distributions
        nA = sum(graphA.initial_state_counts.values())
        nB = sum(graphB.initial_state_counts.values())
        nTotal = nA + nB
        if nTotal > 0:
            all_v = set(graphA.marginal_v_genes) | set(graphB.marginal_v_genes)
            graphA.marginal_v_genes = {
                g: graphA.marginal_v_genes.get(g, 0) * nA / nTotal
                   + graphB.marginal_v_genes.get(g, 0) * nB / nTotal
                for g in all_v
            }
            all_j = set(graphA.marginal_j_genes) | set(graphB.marginal_j_genes)
            graphA.marginal_j_genes = {
                g: graphA.marginal_j_genes.get(g, 0) * nA / nTotal
                   + graphB.marginal_j_genes.get(g, 0) * nB / nTotal
                for g in all_j
            }
            all_vj = set(graphA.vj_probabilities) | set(graphB.vj_probabilities)
            graphA.vj_probabilities = {
                g: graphA.vj_probabilities.get(g, 0) * nA / nTotal
                   + graphB.vj_probabilities.get(g, 0) * nB / nTotal
                for g in all_vj
            }

        # Merge length_distribution counts
        if hasattr(graphA, 'length_counts') and hasattr(graphB, 'length_counts'):
            for k, v in graphB.length_counts.items():
                graphA.length_counts[k] = graphA.length_counts.get(k, 0) + v

        # Merge observed gene sets
        if hasattr(graphB, 'observed_v_genes'):
            graphA.observed_v_genes = list(
                set(graphA.observed_v_genes) | set(graphB.observed_v_genes)
            )
        if hasattr(graphB, 'observed_j_genes'):
            graphA.observed_j_genes = list(
                set(graphA.observed_j_genes) | set(graphB.observed_j_genes)
            )

    # 5. Recalculate ALL derived state from raw counts
    graphA.recalculate()

    # Clear cached edges list
    if hasattr(graphA, '_edges_cache'):
        graphA._edges_cache = None

    return graphA
