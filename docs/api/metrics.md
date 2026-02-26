# Metrics

Functions for measuring repertoire diversity, entropy, and similarity.

## Import

```python
from LZGraphs import (
    k1000_diversity,
    k_diversity,
    k100_diversity,
    k500_diversity,
    k5000_diversity,
    adaptive_k_diversity,
    lz_centrality,
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    sequence_perplexity,
    repertoire_perplexity,
    jensen_shannon_divergence,
    cross_entropy,
    kl_divergence,
    mutual_information_genes,
    transition_predictability,
    graph_compression_ratio,
    repertoire_compressibility_index,
    transition_kl_divergence,
    transition_jsd,
    transition_mutual_information_profile,
    path_entropy_rate,
    compare_repertoires,
    LZPgenDistribution,
    compare_lzpgen_distributions,
)
```

## K-Diversity Functions

### k1000_diversity

Calculate K1000 diversity index.

```python
from LZGraphs import k1000_diversity, AAPLZGraph

sequences = data['cdr3_amino_acid'].tolist()
k1000 = k1000_diversity(sequences, AAPLZGraph.encode_sequence, draws=30)
print(f"K1000: {k1000:.1f}")
```

!!! note "Function Signature"
    `k1000_diversity(list_of_sequences, lzgraph_encoding_function, draws=25, return_stats=False, confidence_level=0.95)`

    Returns the mean K1000 diversity index. When `return_stats=True`, returns a tuple `(mean, std, ci_lower, ci_upper)`.

### k_diversity

General K-diversity with configurable parameters.

```python
from LZGraphs import k_diversity

mean, std, ci_lower, ci_upper = k_diversity(
    sequences,
    AAPLZGraph.encode_sequence,
    sample_size=1000,
    draws=100,
    return_stats=True
)
print(f"Mean: {mean:.1f}, CI: [{ci_lower:.1f}, {ci_upper:.1f}]")
```

!!! note "Function Signature"
    `k_diversity(list_of_sequences, lzgraph_encoding_function, sample_size=1000, draws=25, return_stats=False, confidence_level=0.95)`

    General K-diversity calculation with configurable sample size. When `return_stats=True`, returns a tuple `(mean, std, ci_lower, ci_upper)`. When `return_stats=False`, returns a single float (mean).

### Other K-Diversity Variants

| Function | Sample Size | Use Case |
|----------|-------------|----------|
| `k100_diversity` | 100 | Small repertoires |
| `k500_diversity` | 500 | Medium repertoires |
| `k1000_diversity` | 1000 | Standard analysis |
| `k5000_diversity` | 5000 | Large repertoires |
| `adaptive_k_diversity` | Auto | Automatic selection |

---

## lz_centrality

Measure sequence centrality within a repertoire.

```python
from LZGraphs import lz_centrality

centrality = lz_centrality(graph, "CASSLEPSGGTDTQYF")
print(f"Centrality: {centrality:.4f}")
```

!!! note "Function Signature"
    `lz_centrality(graph, sequence) -> float`

    Calculates the lz_centrality of a sequence within the given graph's structure.

---

## Entropy Functions

### node_entropy

Entropy of node (pattern) distribution.

```python
from LZGraphs import node_entropy

h = node_entropy(graph)
print(f"Node entropy: {h:.2f} bits")
```

### edge_entropy

Entropy of edge (transition) distribution.

```python
from LZGraphs import edge_entropy

h = edge_entropy(graph)
print(f"Edge entropy: {h:.2f} bits")
```

### graph_entropy

Combined graph entropy measure.

```python
from LZGraphs import graph_entropy, normalized_graph_entropy

h = graph_entropy(graph)
h_norm = normalized_graph_entropy(graph)
print(f"Graph entropy: {h:.2f} bits (normalized: {h_norm:.4f})")
```

---

## Perplexity Functions

### sequence_perplexity

Perplexity of a single sequence.

```python
from LZGraphs import sequence_perplexity

perp = sequence_perplexity(graph, "CASSLEPSGGTDTQYF")
print(f"Perplexity: {perp:.2f}")
```

### repertoire_perplexity

Average perplexity across sequences.

```python
from LZGraphs import repertoire_perplexity

avg_perp = repertoire_perplexity(graph, sequences)
print(f"Average perplexity: {avg_perp:.2f}")
```

---

## Divergence Functions

### jensen_shannon_divergence

Symmetric divergence between two repertoires.

```python
from LZGraphs import jensen_shannon_divergence

jsd = jensen_shannon_divergence(graph1, graph2)
print(f"JS Divergence: {jsd:.4f}")  # 0 to 1
```

!!! note "Function Signature"
    `jensen_shannon_divergence(lzgraph1, lzgraph2) -> float`

    Calculates the Jensen-Shannon divergence between two LZGraph objects based on their edge weight distributions.

### cross_entropy

Cross-entropy between repertoires.

```python
from LZGraphs import cross_entropy

ce = cross_entropy(graph1, graph2)
print(f"Cross entropy: {ce:.2f}")
```

### kl_divergence

Kullback-Leibler divergence (asymmetric).

```python
from LZGraphs import kl_divergence

kl = kl_divergence(graph1, graph2)
print(f"KL Divergence: {kl:.4f}")
```

---

## Gene Information

### mutual_information_genes

Mutual information between genes and patterns.

```python
from LZGraphs import mutual_information_genes

mi_v = mutual_information_genes(graph, gene_type='V')
mi_j = mutual_information_genes(graph, gene_type='J')
print(f"MI (V): {mi_v:.4f}, MI (J): {mi_j:.4f}")
```

---

## Information-Theoretic Metrics

### transition_predictability

Measures how deterministic the graph transitions are relative to the maximum possible branching.

```python
from LZGraphs import transition_predictability

tp = transition_predictability(graph)
print(f"Transition predictability: {tp:.3f}")  # 0 to 1
```

!!! note "Function Signature"
    `transition_predictability(lzgraph, base=2) -> float`

    Returns a value in [0, 1]. Higher values indicate more deterministic transitions (restricted repertoire). Empirically stable at ~0.60 for AAPLZGraph across sample sizes.

### graph_compression_ratio

Measures how much the graph compresses repeated transitions into shared edges.

```python
from LZGraphs import graph_compression_ratio

gcr = graph_compression_ratio(graph)
print(f"Compression ratio: {gcr:.3f}")  # 0 to 1
```

!!! note "Function Signature"
    `graph_compression_ratio(lzgraph) -> float`

    Returns `n_edges / n_transitions`. Lower values indicate more path sharing. AAPLZGraph ~0.18, NaiveLZGraph ~0.05.

### repertoire_compressibility_index

Alias for `transition_predictability`, framed from a data compression perspective.

```python
from LZGraphs import repertoire_compressibility_index

rci = repertoire_compressibility_index(graph)
print(f"Compressibility: {rci:.3f}")  # 0 to 1
```

!!! note "Function Signature"
    `repertoire_compressibility_index(lzgraph, base=2) -> float`

    RCI = 1 means fully deterministic (compressible), RCI = 0 means maximally uncertain (incompressible).

### path_entropy_rate

Estimates the average information content per subpattern step across actual sequences.

```python
from LZGraphs import path_entropy_rate

sequences = data['cdr3_amino_acid'].tolist()
h = path_entropy_rate(graph, sequences)
print(f"Entropy rate: {h:.3f} bits/step")
```

!!! note "Function Signature"
    `path_entropy_rate(lzgraph, sequences, base=2) -> float`

    Uses `walk_log_probability()` internally. AAPLZGraph ~2.5 bits/step, NaiveLZGraph ~3.5 bits/step.

---

## Transition-Level Divergence

### transition_kl_divergence

Transition-level KL divergence — compares the transition structure, not just node distributions.

```python
from LZGraphs import transition_kl_divergence

kl = transition_kl_divergence(graph1, graph2)
print(f"Transition KL: {kl:.4f}")
```

!!! note "Function Signature"
    `transition_kl_divergence(lzgraph_p, lzgraph_q) -> float`

    Asymmetric, can be infinite. Use `transition_jsd` for a bounded alternative.

### transition_jsd

Transition-level Jensen-Shannon divergence — always finite, symmetric.

```python
from LZGraphs import transition_jsd

jsd_t = transition_jsd(graph1, graph2)
print(f"Transition JSD: {jsd_t:.4f}")  # 0 to 1
```

!!! note "Function Signature"
    `transition_jsd(lzgraph1, lzgraph2) -> float`

    Symmetric and bounded [0, 1]. Recommended for comparing repertoire transition structures.

### transition_mutual_information_profile

Position-specific mutual information along the CDR3 sequence.

```python
from LZGraphs import transition_mutual_information_profile

tmip = transition_mutual_information_profile(graph)
for pos in sorted(tmip):
    print(f"Position {pos}: MI = {tmip[pos]:.3f} bits")
```

!!! note "Function Signature"
    `transition_mutual_information_profile(lzgraph) -> dict`

    Returns `{position: mutual_information}`. Only works with positional graphs (AAPLZGraph, NDPLZGraph). Raises `MetricsError` for NaiveLZGraph.

---

## Convenience

### compare_repertoires

All-in-one repertoire comparison returning a dict of metrics.

```python
from LZGraphs import compare_repertoires

result = compare_repertoires(graph1, graph2)
print(result)
```

!!! note "Function Signature"
    `compare_repertoires(graph1, graph2) -> dict`

    Returns: `js_divergence`, `transition_jsd`, `cross_entropy_1_2`, `cross_entropy_2_1`, `kl_divergence_1_2`, `kl_divergence_2_1`, `node_entropy_1`, `node_entropy_2`, `edge_entropy_1`, `edge_entropy_2`, `transition_predictability_1`, `transition_predictability_2`, `shared_nodes`, `shared_edges`, `jaccard_nodes`, `jaccard_edges`.

---

## Analytical Pgen Distribution

### LZPgenDistribution

Analytical generation probability distribution derived from graph structure, represented as a Gaussian mixture model. No Monte Carlo sampling needed.

```python
from LZGraphs import LZPgenDistribution

# Compute from a graph
dist = graph.lzpgen_analytical_distribution()

# PDF and CDF
import numpy as np
x = np.linspace(-35, -10, 500)
pdf_values = dist.pdf(x)
cdf_values = dist.cdf(x)

# Confidence interval
ci_low, ci_high = dist.confidence_interval(0.05)
print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")

# Attributes
print(f"Components: {dist.n_components}")
print(f"Cumulants: {dist.cumulants}")
```

### compare_lzpgen_distributions

Compare two empirical log-probability distributions.

```python
from LZGraphs import compare_lzpgen_distributions

metrics = compare_lzpgen_distributions(log_probs_1, log_probs_2)
# Returns dict with: ks_statistic, ks_pvalue, jsd, overlap_coefficient,
# mean_diff, median_diff, effect_size
```

---

## See Also

- [Tutorials: Diversity Metrics](../tutorials/diversity-metrics.md)
- [How-To: Compare Repertoires](../how-to/repertoire-comparison.md)
- [Concepts: Probability Model](../concepts/probability-model.md)
- [Example: LZPgen](https://github.com/MuteJester/LZGraphs/blob/master/Examples/LZPgen%20Example.ipynb)
- [Example: Information-Theoretic Analysis](https://github.com/MuteJester/LZGraphs/blob/master/Examples/Information-Theoretic%20Analysis.ipynb)
