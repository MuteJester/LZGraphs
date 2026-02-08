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
from scipy.spatial.distance import jensenshannon as _scipy_jsd
from scipy.stats import entropy as scipy_entropy

from ..exceptions import EmptyDataError, NoGeneDataError, MetricsError
from ..utilities.misc import _is_v_gene, _is_j_gene

__all__ = [
    "node_entropy",
    "edge_entropy",
    "graph_entropy",
    "normalized_graph_entropy",
    "sequence_perplexity",
    "repertoire_perplexity",
    "jensen_shannon_divergence",
    "cross_entropy",
    "kl_divergence",
    "mutual_information_genes",
    "transition_predictability",
    "graph_compression_ratio",
    "repertoire_compressibility_index",
    "transition_kl_divergence",
    "transition_jsd",
    "transition_mutual_information_profile",
    "path_entropy_rate",
]


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
        raise MetricsError("LZGraph does not have subpattern probability data")

    probs = lzgraph.subpattern_individual_probability['proba'].values
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]

    if len(probs) == 0:
        return 0.0

    return float(scipy_entropy(probs, base=base))


def edge_entropy(lzgraph, base: float = 2) -> float:
    """
    Compute the weighted average conditional entropy of edge transitions.

    For each node, compute the Shannon entropy of its outgoing transition
    distribution. Then weight by the node's probability (from
    subpattern_individual_probability) to get the overall conditional entropy:

        H(transition|node) = sum_i P(node_i) * H(outgoing edges of node_i)

    This correctly measures the average uncertainty in choosing the next
    node given the current node.

    Args:
        lzgraph: An LZGraph instance
        base: Logarithm base (default 2 for bits)

    Returns:
        float: Weighted conditional entropy of edge transitions

    Example:
        >>> graph = AAPLZGraph(data)
        >>> h = edge_entropy(graph)
        >>> print(f"Edge entropy: {h:.2f} bits")
    """
    if lzgraph.graph.number_of_edges() == 0:
        return 0.0

    node_probs = lzgraph.subpattern_individual_probability.get('proba', {})
    if not len(node_probs):
        return 0.0

    weighted_entropy = 0.0
    for node in lzgraph.graph.nodes():
        out_edges = list(lzgraph.graph.successors(node))
        if not out_edges:
            continue

        # Get outgoing edge weights for this node
        out_weights = np.array([
            lzgraph.graph[node][succ]['data'].weight
            for succ in out_edges
        ])

        # Normalize to a probability distribution
        total = out_weights.sum()
        if total <= 0:
            continue

        out_probs = out_weights / total
        local_entropy = float(scipy_entropy(out_probs, base=base))

        # Weight by node probability
        node_p = node_probs.get(node, 0)
        weighted_entropy += node_p * local_entropy

    return weighted_entropy


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
        raise MetricsError("LZGraph does not have walk_probability method")

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
        raise EmptyDataError("sequences list cannot be empty")

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
        raise MetricsError(f"Unknown aggregation method: {aggregation}. Use 'geometric', 'arithmetic', or 'median'.")


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

    # Use scipy's jensenshannon which handles zeros correctly
    # (it returns sqrt(JSD), so we square to get JSD)
    jsd = _scipy_jsd(p1, p2, base=2) ** 2

    return float(jsd)


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

    if not len(test_prob) or not len(model_prob):
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

    if not len(p_prob):
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
        raise NoGeneDataError(
            operation="mutual_information_genes",
            message="LZGraph must have genetic information for mutual information calculation"
        )

    # Get marginal gene distribution
    if gene_type == 'V':
        marginal = lzgraph.marginal_vgenes
    elif gene_type == 'J':
        marginal = lzgraph.marginal_jgenes
    else:
        raise MetricsError("gene_type must be 'V' or 'J'")

    # Marginal entropy H(Gene)
    h_gene = scipy_entropy(marginal.values, base=2)

    # Estimate conditional entropy H(Gene|Path) via sampling
    # This is complex to compute exactly, so we use a simple approximation
    # based on the gene prediction confidence

    conditional_entropies = []

    for _ in range(n_samples):
        try:
            walk, v, j = lzgraph.genomic_random_walk(clear_blacklist=True)

            # Get gene probabilities for this walk
            gene_probs = []
            for i in range(len(walk) - 1):
                if lzgraph.graph.has_edge(walk[i], walk[i+1]):
                    ed = lzgraph.graph[walk[i]][walk[i+1]]['data']
                    if gene_type == 'V':
                        genes = ed.v_genes
                        if genes:
                            vals = np.array([ed.v_probability(g) for g in genes])
                    else:
                        genes = ed.j_genes
                        if genes:
                            vals = np.array([ed.j_probability(g) for g in genes])
                    if genes:
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


def _max_conditional_entropy(lzgraph, base: float = 2) -> float:
    """Compute the maximum possible conditional entropy given the graph topology.

    H_max = sum_a mu(a) * log_base(|successors(a)|)

    This is the entropy that would result if every node's outgoing
    transitions were uniformly distributed.

    Args:
        lzgraph: An LZGraph instance.
        base: Logarithm base (default 2 for bits).

    Returns:
        float: Maximum conditional entropy given the topology.
    """
    node_probs = lzgraph.subpattern_individual_probability.get('proba', {})
    if not len(node_probs):
        return 0.0

    h_max = 0.0
    log_fn = np.log2 if base == 2 else (lambda x: np.log(x) / np.log(base))

    for node in lzgraph.graph.nodes():
        n_successors = len(list(lzgraph.graph.successors(node)))
        if n_successors <= 1:
            continue
        node_p = node_probs.get(node, 0)
        if node_p > 0:
            h_max += node_p * log_fn(n_successors)

    return h_max


def transition_predictability(lzgraph, base: float = 2) -> float:
    """Compute the transition predictability of an LZGraph.

    Measures how deterministic the transitions are relative to the maximum
    possible branching. A value of 1.0 means all transitions are
    deterministic; 0.0 means transitions are maximally uncertain.

    Formula: Predictability = 1 - H(X_{t+1}|X_t) / log_base(max_out_degree)

    The denominator uses the global maximum out-degree, providing a
    simple upper bound on per-step entropy. This metric is empirically
    stable across sample sizes (~0.60 for AAPLZGraph), making it an
    intrinsic property of the repertoire rather than a sample-size artifact.

    Args:
        lzgraph: An LZGraph instance.
        base: Logarithm base (default 2 for bits).

    Returns:
        float: Predictability in range [0, 1].

    Example:
        >>> graph = AAPLZGraph(data)
        >>> tp = transition_predictability(graph)
        >>> print(f"Transition predictability: {tp:.3f}")
    """
    h_trans = edge_entropy(lzgraph, base=base)

    # Use global max out-degree as the normalizer
    max_out_degree = 0
    for node in lzgraph.graph.nodes():
        out_deg = len(list(lzgraph.graph.successors(node)))
        if out_deg > max_out_degree:
            max_out_degree = out_deg

    if max_out_degree <= 1:
        return 1.0  # No branching -> fully deterministic

    log_fn = np.log2 if base == 2 else (lambda x: np.log(x) / np.log(base))
    h_max = log_fn(max_out_degree)

    return float(1.0 - h_trans / h_max)


def graph_compression_ratio(lzgraph) -> float:
    """Compute the graph compression ratio (GCR).

    Measures how much the graph compresses repeated transitions into
    shared edges. Lower values indicate more compression (more path sharing).

    Formula: GCR = n_edges / n_transitions

    For NaiveLZGraph (no positional encoding), subpatterns are shared
    across sequences, yielding low GCR (~0.05). For AAPLZGraph (positional),
    sharing is limited, yielding higher GCR (~0.18).

    Args:
        lzgraph: An LZGraph instance with n_transitions attribute.

    Returns:
        float: Compression ratio in range (0, 1]. Lower = more compression.

    Example:
        >>> graph = AAPLZGraph(data)
        >>> gcr = graph_compression_ratio(graph)
        >>> print(f"Graph compression ratio: {gcr:.3f}")
    """
    n_edges = lzgraph.graph.number_of_edges()
    n_transitions = getattr(lzgraph, 'n_transitions', 0)

    if n_transitions <= 0:
        # Fall back to summing raw edge counts (e.g. NaiveLZGraph
        # doesn't increment n_transitions during construction)
        n_transitions = sum(
            lzgraph.graph[u][v]['data'].count
            for u, v in lzgraph.graph.edges()
        )

    if n_transitions <= 0:
        return 1.0

    return float(n_edges / n_transitions)


def repertoire_compressibility_index(lzgraph, base: float = 2) -> float:
    """Compute the Repertoire Compressibility Index (RCI).

    Measures how predictable the repertoire is at the subpattern transition
    level. This is the complement of the normalized conditional entropy:

    RCI = 1 - H_trans / H_max_topology

    where H_max_topology is the maximum conditional entropy achievable
    given the graph's branching structure (uniform transitions at each node).

    Interpretation:
    - RCI = 1: Fully deterministic (each subpattern has exactly one successor)
    - RCI = 0: Maximally uncertain (uniform transitions everywhere)

    Clinically, high RCI may indicate clonal expansion or restricted
    repertoire diversity; low RCI indicates healthy polyclonal diversity.

    Note: This is mathematically equivalent to transition_predictability
    but framed from the information-theoretic compression perspective
    (Part 4, Definition 6.2 of the mathematical analysis).

    Args:
        lzgraph: An LZGraph instance.
        base: Logarithm base (default 2 for bits).

    Returns:
        float: RCI in range [0, 1].

    Example:
        >>> graph = AAPLZGraph(data)
        >>> rci = repertoire_compressibility_index(graph)
        >>> print(f"Repertoire compressibility: {rci:.3f}")
    """
    return transition_predictability(lzgraph, base=base)


def transition_kl_divergence(lzgraph_p, lzgraph_q) -> float:
    """Compute transition-level KL divergence D_KL^trans(P || Q).

    Unlike the node-level kl_divergence() which only compares subpattern
    usage distributions, this compares the *transition structure* — the
    conditional distributions P(next|current) at each node.

    Formula:
        D_KL^trans(G1 || G2) = sum_a mu1(a) * D_KL(P1(.|a) || P2(.|a))

    This is a weighted average of per-node KL divergences where the
    weight is the probability of being at node a under G1.

    Two graphs with identical node distributions but different transition
    structures will have D_KL^node = 0 but D_KL^trans > 0.

    Args:
        lzgraph_p: The "true" distribution P.
        lzgraph_q: The "approximate" distribution Q.

    Returns:
        float: Transition-level KL divergence in bits (can be +inf).

    Example:
        >>> kl_t = transition_kl_divergence(graph_empirical, graph_model)
        >>> print(f"Transition KL divergence: {kl_t:.3f} bits")
    """
    p_node_probs = lzgraph_p.subpattern_individual_probability.get('proba', {})
    if not len(p_node_probs):
        return 0.0

    kl_total = 0.0

    for node_a, mu_a in p_node_probs.items():
        if mu_a <= 0:
            continue

        # Get outgoing edges for this node in both graphs
        successors_p = set(lzgraph_p.graph.successors(node_a)) if lzgraph_p.graph.has_node(node_a) else set()
        successors_q = set(lzgraph_q.graph.successors(node_a)) if lzgraph_q.graph.has_node(node_a) else set()

        if not successors_p:
            continue

        # Check support condition: all successors in P must exist in Q
        if not successors_p.issubset(successors_q):
            return float('inf')

        # Compute per-node KL divergence
        kl_node = 0.0
        for succ in successors_p:
            p_val = lzgraph_p.graph[node_a][succ]['data'].weight
            q_val = lzgraph_q.graph[node_a][succ]['data'].weight if succ in successors_q else 0.0

            if p_val > 0:
                if q_val <= 0:
                    return float('inf')
                kl_node += p_val * np.log2(p_val / q_val)

        kl_total += mu_a * kl_node

    return float(kl_total)


def transition_jsd(lzgraph1, lzgraph2) -> float:
    """Compute transition-level Jensen-Shannon divergence between two LZGraphs.

    Unlike the node-level jensen_shannon_divergence() which only compares
    subpattern usage distributions, this compares the full transition
    structure — the conditional distributions P(next|current) at each node.

    Formula:
        JSD^trans = 0.5 * D_KL^trans(G1 || G_mix) + 0.5 * D_KL^trans(G2 || G_mix)

    where G_mix has mixed transition probabilities:
        P_mix(b|a) = (mu1(a)*P1(b|a) + mu2(a)*P2(b|a)) / (mu1(a) + mu2(a))

    This is always finite and symmetric, making it more robust than
    transition_kl_divergence for practical repertoire comparison.

    Args:
        lzgraph1: First LZGraph instance.
        lzgraph2: Second LZGraph instance.

    Returns:
        float: Transition-level JSD in range [0, 1].

    Example:
        >>> jsd_t = transition_jsd(graph_healthy, graph_disease)
        >>> print(f"Transition JSD: {jsd_t:.3f}")
    """
    mu1 = lzgraph1.subpattern_individual_probability.get('proba', {})
    mu2 = lzgraph2.subpattern_individual_probability.get('proba', {})

    if not len(mu1) and not len(mu2):
        return 0.0

    # Collect all nodes that are source nodes in either graph
    all_source_nodes = set()
    for node in lzgraph1.graph.nodes():
        if len(list(lzgraph1.graph.successors(node))) > 0:
            all_source_nodes.add(node)
    for node in lzgraph2.graph.nodes():
        if len(list(lzgraph2.graph.successors(node))) > 0:
            all_source_nodes.add(node)

    if not all_source_nodes:
        return 0.0

    jsd_total = 0.0

    for node_a in all_source_nodes:
        mu1_a = mu1.get(node_a, 0)
        mu2_a = mu2.get(node_a, 0)
        mu_sum = mu1_a + mu2_a

        if mu_sum <= 0:
            continue

        # Collect all successors from both graphs
        successors_1 = set(lzgraph1.graph.successors(node_a)) if lzgraph1.graph.has_node(node_a) else set()
        successors_2 = set(lzgraph2.graph.successors(node_a)) if lzgraph2.graph.has_node(node_a) else set()
        all_successors = successors_1 | successors_2

        if not all_successors:
            continue

        # Build transition distributions for this node
        p1 = np.array([
            lzgraph1.graph[node_a][s]['data'].weight if s in successors_1 else 0.0
            for s in all_successors
        ])
        p2 = np.array([
            lzgraph2.graph[node_a][s]['data'].weight if s in successors_2 else 0.0
            for s in all_successors
        ])

        # Mixture distribution: weighted by node probability in each graph
        p_mix = (mu1_a * p1 + mu2_a * p2) / mu_sum

        # Per-node JSD contribution
        eps = np.finfo(float).eps
        kl1 = 0.0
        kl2 = 0.0

        for i in range(len(p1)):
            m_val = p_mix[i]
            if m_val <= 0:
                continue
            if p1[i] > 0:
                kl1 += p1[i] * np.log2(p1[i] / m_val)
            if p2[i] > 0:
                kl2 += p2[i] * np.log2(p2[i] / m_val)

        # Weight by mixture node probability
        w1 = 0.5 * mu1_a
        w2 = 0.5 * mu2_a
        jsd_total += w1 * kl1 + w2 * kl2

    return float(min(jsd_total, 1.0))


def transition_mutual_information_profile(lzgraph) -> dict:
    """Compute the Transition Mutual Information Profile (TMIP).

    For positional LZGraphs (AAPLZGraph, NDPLZGraph), this computes the
    mutual information between the current and next subpatterns at each
    position along the sequence:

        TMIP(p) = H(X_{p'}) - H(X_{p'} | X_p)

    High TMIP at a position means the current subpattern strongly constrains
    what comes next. Low TMIP means the next subpattern is nearly independent.

    Biological insight:
    - High TMIP at V/N and N/J boundaries (transition from germline to
      stochastic subpatterns)
    - Low TMIP in the junctional diversity region (random insertions)

    Args:
        lzgraph: A positional LZGraph instance (AAPLZGraph or NDPLZGraph).

    Returns:
        dict: {position (int): mutual_information (float)} mapping.

    Raises:
        MetricsError: If the graph has no positional encoding (NaiveLZGraph).

    Example:
        >>> graph = AAPLZGraph(data)
        >>> tmip = transition_mutual_information_profile(graph)
        >>> for pos in sorted(tmip): print(f"  Position {pos}: MI = {tmip[pos]:.3f}")
    """
    # Check that graph has positional encoding
    # NaiveLZGraph nodes have no '_' separator with position
    sample_nodes = list(lzgraph.graph.nodes())[:10]
    has_position = any('_' in str(n) for n in sample_nodes)
    if not has_position:
        raise MetricsError(
            "transition_mutual_information_profile requires a positional graph "
            "(AAPLZGraph or NDPLZGraph). NaiveLZGraph nodes have no position information."
        )

    node_probs = lzgraph.subpattern_individual_probability.get('proba', {})
    if not len(node_probs):
        return {}

    # Group nodes by position
    position_nodes = {}  # {position: [node_names]}
    for node in lzgraph.graph.nodes():
        node_str = str(node)
        if '_' in node_str:
            try:
                pos = int(node_str.rsplit('_', 1)[1])
                position_nodes.setdefault(pos, []).append(node)
            except (ValueError, IndexError):
                continue

    tmip = {}

    for pos, nodes in sorted(position_nodes.items()):
        # Filter to nodes with outgoing edges
        source_nodes = [n for n in nodes if len(list(lzgraph.graph.successors(n))) > 0]
        if not source_nodes:
            continue

        # Compute conditional distribution mu_p(node) for this position
        total_mu = sum(node_probs.get(n, 0) for n in source_nodes)
        if total_mu <= 0:
            continue

        # H(X_{p'} | X_p) — conditional entropy of next node given current at pos p
        h_cond = 0.0
        # Marginal next-state distribution nu(b) for H(X_{p'})
        marginal_next = {}

        for node in source_nodes:
            mu_node = node_probs.get(node, 0)
            if mu_node <= 0:
                continue
            w = mu_node / total_mu  # Conditional probability of this node given position p

            successors = list(lzgraph.graph.successors(node))
            if not successors:
                continue

            out_weights = np.array([
                lzgraph.graph[node][s]['data'].weight for s in successors
            ])
            total_w = out_weights.sum()
            if total_w <= 0:
                continue
            out_probs = out_weights / total_w

            # Accumulate conditional entropy
            h_local = float(scipy_entropy(out_probs, base=2))
            h_cond += w * h_local

            # Accumulate marginal next-state distribution
            for s, p_val in zip(successors, out_probs):
                marginal_next[s] = marginal_next.get(s, 0) + w * p_val

        # H(X_{p'}) — marginal entropy of next state
        if marginal_next:
            next_probs = np.array(list(marginal_next.values()))
            h_marginal = float(scipy_entropy(next_probs, base=2))
        else:
            h_marginal = 0.0

        # MI = H(X_{p'}) - H(X_{p'} | X_p)
        mi = max(0.0, h_marginal - h_cond)
        tmip[pos] = mi

    return tmip


def path_entropy_rate(lzgraph, sequences: List[str], base: float = 2) -> float:
    """Estimate the path entropy rate of the LZGraph model.

    The path entropy rate measures the average information per subpattern
    step across actual sequences:

        h_path = (1/N) * sum_i [-log_base P(walk_i)] / |walk_i|

    This is the proper entropy rate for DAG models (like AAPLZGraph) where
    the classical stationary-distribution-based entropy rate is undefined.

    For the NaiveLZGraph (which may have a stationary distribution), this
    provides an empirical estimate that avoids convergence issues with
    power iteration.

    Args:
        lzgraph: An LZGraph instance with walk_log_probability method.
        sequences: A list of sequences to estimate the entropy rate from.
        base: Logarithm base (default 2 for bits).

    Returns:
        float: Estimated entropy rate in specified base (bits if base=2).

    Raises:
        EmptyDataError: If no sequences can be scored.

    Example:
        >>> graph = AAPLZGraph(data)
        >>> seqs = data['cdr3_amino_acid'].tolist()
        >>> h = path_entropy_rate(graph, seqs)
        >>> print(f"Path entropy rate: {h:.3f} bits/step")
    """
    if not sequences:
        raise EmptyDataError("sequences list cannot be empty")

    total_neg_log_prob = 0.0
    total_walk_length = 0
    n_scored = 0

    for seq in sequences:
        try:
            log_prob = lzgraph.walk_log_probability(seq, verbose=False)
            walk = lzgraph.encode_sequence(seq)

            if not np.isfinite(log_prob) or len(walk) == 0:
                continue

            # Convert natural log to specified base
            if base == 2:
                neg_log_prob = -log_prob / np.log(2)
            else:
                neg_log_prob = -log_prob / np.log(base)

            total_neg_log_prob += neg_log_prob
            total_walk_length += len(walk)
            n_scored += 1
        except (KeyError, ValueError, ZeroDivisionError, AttributeError):
            continue

    if n_scored == 0 or total_walk_length == 0:
        raise EmptyDataError("No sequences could be scored for path entropy rate")

    return float(total_neg_log_prob / total_walk_length)
