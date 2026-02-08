"""
Convenience functions for common repertoire analysis tasks.

These functions wrap lower-level LZGraphs functionality into easy-to-use
high-level operations for comparing and summarizing repertoires.
"""

import numpy as np
import pandas as pd

from .entropy import (
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    jensen_shannon_divergence,
    cross_entropy,
    kl_divergence,
)


__all__ = ['compare_repertoires']


def compare_repertoires(graph1, graph2):
    """
    Compare two LZGraph repertoire representations using multiple metrics.

    Computes a suite of comparison metrics between two LZGraphs, including
    divergence measures, entropy differences, and structural overlap.

    Args:
        graph1: First LZGraph instance.
        graph2: Second LZGraph instance.

    Returns:
        pd.Series: Named series containing comparison metrics:
            - js_divergence: Jensen-Shannon divergence (0=identical, 1=different)
            - cross_entropy_1_2: Cross-entropy H(graph1, graph2)
            - cross_entropy_2_1: Cross-entropy H(graph2, graph1)
            - kl_divergence_1_2: KL divergence D_KL(graph1 || graph2)
            - kl_divergence_2_1: KL divergence D_KL(graph2 || graph1)
            - node_entropy_1: Node entropy of graph1
            - node_entropy_2: Node entropy of graph2
            - edge_entropy_1: Edge entropy of graph1
            - edge_entropy_2: Edge entropy of graph2
            - shared_nodes: Number of nodes in common
            - shared_edges: Number of edges in common
            - jaccard_nodes: Jaccard similarity of node sets
            - jaccard_edges: Jaccard similarity of edge sets

    Example:
        >>> graph_healthy = AAPLZGraph(healthy_data)
        >>> graph_disease = AAPLZGraph(disease_data)
        >>> comparison = compare_repertoires(graph_healthy, graph_disease)
        >>> print(comparison)
    """
    nodes1 = set(graph1.graph.nodes())
    nodes2 = set(graph2.graph.nodes())
    edges1 = set(graph1.graph.edges())
    edges2 = set(graph2.graph.edges())

    shared_nodes = len(nodes1 & nodes2)
    shared_edges = len(edges1 & edges2)

    union_nodes = len(nodes1 | nodes2)
    union_edges = len(edges1 | edges2)

    return pd.Series({
        'js_divergence': jensen_shannon_divergence(graph1, graph2),
        'cross_entropy_1_2': cross_entropy(graph1, graph2),
        'cross_entropy_2_1': cross_entropy(graph2, graph1),
        'kl_divergence_1_2': kl_divergence(graph1, graph2),
        'kl_divergence_2_1': kl_divergence(graph2, graph1),
        'node_entropy_1': node_entropy(graph1),
        'node_entropy_2': node_entropy(graph2),
        'edge_entropy_1': edge_entropy(graph1),
        'edge_entropy_2': edge_entropy(graph2),
        'shared_nodes': shared_nodes,
        'shared_edges': shared_edges,
        'jaccard_nodes': shared_nodes / union_nodes if union_nodes > 0 else 0.0,
        'jaccard_edges': shared_edges / union_edges if union_edges > 0 else 0.0,
    })
