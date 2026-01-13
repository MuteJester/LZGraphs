"""
Information-theoretic metrics for LZGraph analysis.

This module provides entropy-based measures for quantifying the information
content and diversity of T-cell receptor repertoires encoded as LZGraphs.

Metrics included:
- Node and edge entropy: Shannon entropy of probability distributions
- Graph entropy: Combined information content measure
- Perplexity: Model "surprise" when encountering sequences
- Cross-entropy and divergence: For comparing repertoires
"""

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import entropy as scipy_entropy


def node_entropy(lzgraph, base: float = 2) -> float:
    """
    Compute the Shannon entropy of the node probability distribution.

    This measures the uncertainty/randomness in the LZ-subpattern usage.
    Higher entropy indicates more uniform usage of patterns (more diverse),
    while lower entropy indicates concentration on few patterns.

    Args:
        lzgraph: An LZGraph instance (AAPLZGraph, NDPLZGraph, etc.)
        base: Logarithm base (default 2 for bits)

    Returns:
        float: Shannon entropy in specified base (bits if base=2)

    Example:
        >>> graph = AAPLZGraph(data)
        >>> h = node_entropy(graph)
        >>> print(f"Node entropy: {h:.2f} bits")
    """
    if not hasattr(lzgraph, 'subpattern_individual_probability'):
        raise ValueError("LZGraph does not have subpattern probability data")

    probs = lzgraph.subpattern_individual_probability['proba'].values
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]

    if len(probs) == 0:
        return 0.0

    return float(scipy_entropy(probs, base=base))


def edge_entropy(lzgraph, base: float = 2) -> float:
    """
    Compute the Shannon entropy of the edge transition probability distribution.

    This measures the uncertainty in transition patterns between nodes.
    Higher entropy indicates more uniform transitions (more diverse pathways),
    while lower entropy indicates concentration on few common transitions.

    Args:
        lzgraph: An LZGraph instance
        base: Logarithm base (default 2 for bits)

    Returns:
        float: Shannon entropy of edge weights in specified base

    Example:
        >>> graph = AAPLZGraph(data)
        >>> h = edge_entropy(graph)
        >>> print(f"Edge entropy: {h:.2f} bits")
    """
    if lzgraph.graph.number_of_edges() == 0:
        return 0.0

    weights = np.array([
        lzgraph.graph[u][v]['weight']
        for u, v in lzgraph.graph.edges()
    ])

    # Normalize to probability distribution
    total = weights.sum()
    if total <= 0:
        return 0.0

    probs = weights / total
    probs = probs[probs > 0]

    return float(scipy_entropy(probs, base=base))


def graph_entropy(lzgraph, base: float = 2) -> float:
    """
    Compute the combined graph entropy (node + edge entropy).

    This provides an overall measure of the information content of the
    repertoire representation. Higher values indicate more complex/diverse
    repertoires.

    Args:
        lzgraph: An LZGraph instance
        base: Logarithm base (default 2 for bits)

    Returns:
        float: Combined entropy H(nodes) + H(edges)

    Example:
        >>> graph = AAPLZGraph(data)
        >>> h = graph_entropy(graph)
        >>> print(f"Graph entropy: {h:.2f} bits")
    """
    return node_entropy(lzgraph, base) + edge_entropy(lzgraph, base)


def normalized_graph_entropy(lzgraph) -> float:
    """
    Compute graph entropy normalized by theoretical maximum.

    The normalized entropy ranges from 0 to 1:
    - 0: Minimum entropy (completely deterministic)
    - 1: Maximum entropy (uniform distribution over all nodes/edges)

    This allows comparison across graphs of different sizes.

    Args:
        lzgraph: An LZGraph instance

    Returns:
        float: Normalized entropy in range [0, 1]

    Example:
        >>> graph = AAPLZGraph(data)
        >>> h_norm = normalized_graph_entropy(graph)
        >>> print(f"Normalized entropy: {h_norm:.3f}")
    """
    n_nodes = lzgraph.graph.number_of_nodes()
    n_edges = lzgraph.graph.number_of_edges()

    if n_nodes <= 1 and n_edges <= 1:
        return 0.0

    # Maximum entropy: log2(n) for uniform distribution
    max_node_entropy = np.log2(n_nodes) if n_nodes > 0 else 0
    max_edge_entropy = np.log2(n_edges) if n_edges > 0 else 0
    max_entropy = max_node_entropy + max_edge_entropy

    if max_entropy <= 0:
        return 0.0

    return graph_entropy(lzgraph, base=2) / max_entropy


def sequence_perplexity(lzgraph, sequence: str) -> float:
    """
    Compute the perplexity of a sequence under the LZGraph model.

    Perplexity measures how "surprised" the model is by a sequence.
    - Lower perplexity: Sequence fits well within the repertoire
    - Higher perplexity: Sequence is unusual/novel for this repertoire

    Formula: Perplexity = 2^(-1/n * log2(P(sequence)))

    where n is the number of subpatterns and P(sequence) is the
    generation probability.

    Args:
        lzgraph: An LZGraph instance with walk_probability method
        sequence: A CDR3 sequence string

    Returns:
        float: Perplexity score (lower = better fit)

    Example:
        >>> graph = AAPLZGraph(data)
        >>> ppl = sequence_perplexity(graph, "CASSLGIRRTNTEAFF")
        >>> print(f"Perplexity: {ppl:.2f}")
    """
    if not hasattr(lzgraph, 'walk_probability'):
        raise ValueError("LZGraph does not have walk_probability method")

    # Get log probability
    log_prob = lzgraph.walk_probability(sequence, verbose=False, use_log=True)

    # Get sequence length in subpatterns
    encoded = lzgraph.encode_sequence(sequence)
    n = len(encoded)

    if n == 0:
        return float('inf')

    # Perplexity = 2^(-log_prob / n) where log_prob is natural log
    # Convert to log2: log2(x) = ln(x) / ln(2)
    log2_prob = log_prob / np.log(2)
    perplexity = 2 ** (-log2_prob / n)

    return float(perplexity)


def repertoire_perplexity(lzgraph, sequences: List[str],
                          aggregation: str = 'geometric') -> float:
    """
    Compute the aggregate perplexity across a repertoire of sequences.

    This provides an overall measure of how well the LZGraph model
    explains a set of sequences.

    Args:
        lzgraph: An LZGraph instance
        sequences: A list of CDR3 sequences
        aggregation: How to aggregate individual perplexities:
            - 'geometric': Geometric mean (default, standard for perplexity)
            - 'arithmetic': Arithmetic mean
            - 'median': Median value

    Returns:
        float: Aggregate perplexity score

    Example:
        >>> graph = AAPLZGraph(train_data)
        >>> test_seqs = test_data['cdr3_amino_acid'].tolist()
        >>> ppl = repertoire_perplexity(graph, test_seqs)
        >>> print(f"Repertoire perplexity: {ppl:.2f}")
    """
    if not sequences:
        raise ValueError("sequences list cannot be empty")

    perplexities = []
    for seq in sequences:
        try:
            ppl = sequence_perplexity(lzgraph, seq)
            if np.isfinite(ppl):
                perplexities.append(ppl)
        except (KeyError, ValueError, ZeroDivisionError, FloatingPointError):
            # Skip sequences that can't be scored (missing nodes, invalid input, etc.)
            continue

    if not perplexities:
        return float('inf')

    perplexities = np.array(perplexities)

    if aggregation == 'geometric':
        # Geometric mean = exp(mean(log(x)))
        return float(np.exp(np.mean(np.log(perplexities))))
    elif aggregation == 'arithmetic':
        return float(np.mean(perplexities))
    elif aggregation == 'median':
        return float(np.median(perplexities))
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")


def jensen_shannon_divergence(lzgraph1, lzgraph2) -> float:
    """
    Compute the Jensen-Shannon divergence between two LZGraphs.

    JS divergence is a symmetric measure of similarity between two
    probability distributions. It ranges from 0 (identical) to 1
    (completely different) when using log base 2.

    This is useful for comparing repertoires, e.g., healthy vs disease.

    Args:
        lzgraph1: First LZGraph instance
        lzgraph2: Second LZGraph instance

    Returns:
        float: JS divergence in range [0, 1]

    Example:
        >>> graph_healthy = AAPLZGraph(healthy_data)
        >>> graph_disease = AAPLZGraph(disease_data)
        >>> js = jensen_shannon_divergence(graph_healthy, graph_disease)
        >>> print(f"JS divergence: {js:.3f}")
    """
    # Get all unique nodes from both graphs
    nodes1 = set(lzgraph1.graph.nodes())
    nodes2 = set(lzgraph2.graph.nodes())
    all_nodes = sorted(nodes1 | nodes2)

    if not all_nodes:
        return 0.0

    # Build probability vectors
    prob1 = lzgraph1.subpattern_individual_probability.get('proba', {})
    prob2 = lzgraph2.subpattern_individual_probability.get('proba', {})

    p1 = np.array([prob1.get(n, 0) for n in all_nodes])
    p2 = np.array([prob2.get(n, 0) for n in all_nodes])

    # Normalize (in case they don't sum to 1 due to missing nodes)
    p1_sum = p1.sum()
    p2_sum = p2.sum()

    if p1_sum > 0:
        p1 = p1 / p1_sum
    if p2_sum > 0:
        p2 = p2 / p2_sum

    # Compute JS divergence
    m = 0.5 * (p1 + p2)

    # Handle zeros by adding small epsilon
    eps = np.finfo(float).eps
    p1 = np.clip(p1, eps, 1)
    p2 = np.clip(p2, eps, 1)
    m = np.clip(m, eps, 1)

    kl1 = scipy_entropy(p1, m, base=2)
    kl2 = scipy_entropy(p2, m, base=2)

    return float(0.5 * (kl1 + kl2))


def cross_entropy(lzgraph_model, lzgraph_test) -> float:
    """
    Compute the cross-entropy H(P, Q) where P is test and Q is model.

    Cross-entropy measures how well the model distribution (lzgraph_model)
    explains the test distribution (lzgraph_test). Lower is better.

    H(P, Q) = -sum(P(x) * log(Q(x)))

    Args:
        lzgraph_model: The model LZGraph (Q)
        lzgraph_test: The test LZGraph (P)

    Returns:
        float: Cross-entropy in bits

    Example:
        >>> model = AAPLZGraph(train_data)
        >>> test = AAPLZGraph(test_data)
        >>> ce = cross_entropy(model, test)
        >>> print(f"Cross-entropy: {ce:.2f} bits")
    """
    # Get test distribution
    test_prob = lzgraph_test.subpattern_individual_probability.get('proba', {})
    model_prob = lzgraph_model.subpattern_individual_probability.get('proba', {})

    if not test_prob or not model_prob:
        return float('inf')

    eps = np.finfo(float).eps
    cross_ent = 0.0

    for node, p_test in test_prob.items():
        if p_test > 0:
            p_model = model_prob.get(node, eps)
            p_model = max(p_model, eps)  # Avoid log(0)
            cross_ent -= p_test * np.log2(p_model)

    return float(cross_ent)


def kl_divergence(lzgraph_p, lzgraph_q) -> float:
    """
    Compute the Kullback-Leibler divergence D_KL(P || Q).

    KL divergence measures how much P differs from Q. It is asymmetric:
    D_KL(P || Q) != D_KL(Q || P).

    Note: KL divergence can be infinite if Q has zero probability where
    P is non-zero. Use jensen_shannon_divergence for a bounded measure.

    Args:
        lzgraph_p: The "true" distribution P
        lzgraph_q: The "approximate" distribution Q

    Returns:
        float: KL divergence in bits (can be infinite)

    Example:
        >>> kl = kl_divergence(graph_empirical, graph_model)
        >>> print(f"KL divergence: {kl:.3f} bits")
    """
    p_prob = lzgraph_p.subpattern_individual_probability.get('proba', {})
    q_prob = lzgraph_q.subpattern_individual_probability.get('proba', {})

    if not p_prob:
        return 0.0

    eps = np.finfo(float).eps
    kl = 0.0

    for node, p_val in p_prob.items():
        if p_val > 0:
            q_val = q_prob.get(node, 0)
            if q_val <= 0:
                return float('inf')  # Undefined when Q=0 and P>0
            kl += p_val * np.log2(p_val / q_val)

    return float(kl)


def mutual_information_genes(lzgraph, gene_type: str = 'V',
                             n_samples: int = 1000) -> float:
    """
    Estimate mutual information between sequence paths and V/J genes.

    I(Path; Gene) measures how much information the LZ-decomposition path
    provides about which V or J gene was used. Higher values indicate
    stronger relationship between sequence structure and gene usage.

    This is estimated via sampling random walks from the graph.

    Args:
        lzgraph: An LZGraph instance with genetic information
        gene_type: 'V' for V genes, 'J' for J genes
        n_samples: Number of random walks for estimation

    Returns:
        float: Estimated mutual information in bits

    Example:
        >>> graph = AAPLZGraph(data)  # With V and J columns
        >>> mi_v = mutual_information_genes(graph, gene_type='V')
        >>> print(f"I(Path; V) = {mi_v:.3f} bits")
    """
    if not lzgraph.genetic:
        raise ValueError("LZGraph must have genetic information")

    # Get marginal gene distribution
    if gene_type == 'V':
        marginal = lzgraph.marginal_vgenes
    elif gene_type == 'J':
        marginal = lzgraph.marginal_jgenes
    else:
        raise ValueError("gene_type must be 'V' or 'J'")

    # Marginal entropy H(Gene)
    h_gene = scipy_entropy(marginal.values, base=2)

    # Estimate conditional entropy H(Gene|Path) via sampling
    # This is complex to compute exactly, so we use a simple approximation
    # based on the gene prediction confidence

    conditional_entropies = []

    for _ in range(min(n_samples, 100)):
        try:
            walk, v, j = lzgraph.genomic_random_walk(clear_blacklist=True)

            # Get gene probabilities for this walk
            gene_probs = []
            for i in range(len(walk) - 1):
                if lzgraph.graph.has_edge(walk[i], walk[i+1]):
                    edge_data = lzgraph.graph[walk[i]][walk[i+1]]
                    if gene_type == 'V':
                        probs = {k: v for k, v in edge_data.items()
                                if k.startswith('V') and k not in ['Vsum']}
                    else:
                        probs = {k: v for k, v in edge_data.items()
                                if k.startswith('J') and k not in ['Jsum']}
                    if probs:
                        vals = np.array(list(probs.values()))
                        vals = vals / vals.sum()
                        conditional_entropies.append(scipy_entropy(vals, base=2))
        except (KeyError, ValueError, ZeroDivisionError, IndexError):
            # Skip walks that can't be completed (missing edges, empty probabilities, etc.)
            continue

    if not conditional_entropies:
        return 0.0

    # Average conditional entropy
    h_gene_given_path = np.mean(conditional_entropies)

    # Mutual information I(Gene; Path) = H(Gene) - H(Gene|Path)
    mi = max(0.0, h_gene - h_gene_given_path)

    return float(mi)
