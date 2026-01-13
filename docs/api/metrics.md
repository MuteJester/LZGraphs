# Metrics

Functions for measuring repertoire diversity, entropy, and similarity.

## Import

```python
from LZGraphs import (
    K1000_Diversity,
    K_Diversity,
    K100_Diversity,
    K500_Diversity,
    K5000_Diversity,
    adaptive_K_Diversity,
    LZCentrality,
    node_entropy,
    edge_entropy,
    graph_entropy,
    normalized_graph_entropy,
    sequence_perplexity,
    repertoire_perplexity,
    jensen_shannon_divergence,
    cross_entropy,
    kl_divergence,
    mutual_information_genes
)
```

## K-Diversity Functions

### K1000_Diversity

Calculate K1000 diversity index.

```python
from LZGraphs import K1000_Diversity, AAPLZGraph

sequences = data['cdr3_amino_acid'].tolist()
k1000 = K1000_Diversity(
    sequences,
    encoding_function=AAPLZGraph.encode_sequence,
    draws=30
)
print(f"K1000: {k1000:.1f}")
```

!!! note "Function Signature"
    `K1000_Diversity(sequences, encoding_function, draws=30) -> float`

    Returns the mean K1000 diversity index across multiple resampling draws.

### K_Diversity

General K-diversity with configurable parameters.

```python
from LZGraphs import K_Diversity

result = K_Diversity(
    sequences,
    encoding_function=AAPLZGraph.encode_sequence,
    sample_size=1000,
    draws=100,
    return_stats=True
)
print(f"Mean: {result['mean']:.1f}, CI: [{result['ci_low']:.1f}, {result['ci_high']:.1f}]")
```

!!! note "Function Signature"
    `K_Diversity(sequences, encoding_function, sample_size=1000, draws=30, return_stats=False)`

    General K-diversity calculation with configurable sample size. When `return_stats=True`, returns a dictionary with mean, std, ci_low, and ci_high.

### Other K-Diversity Variants

| Function | Sample Size | Use Case |
|----------|-------------|----------|
| `K100_Diversity` | 100 | Small repertoires |
| `K500_Diversity` | 500 | Medium repertoires |
| `K1000_Diversity` | 1000 | Standard analysis |
| `K5000_Diversity` | 5000 | Large repertoires |
| `adaptive_K_Diversity` | Auto | Automatic selection |

---

## LZCentrality

Measure sequence centrality within a repertoire.

```python
from LZGraphs import LZCentrality

centrality = LZCentrality(graph, "CASSLEPSGGTDTQYF")
print(f"Centrality: {centrality:.4f}")
```

!!! note "Function Signature"
    `LZCentrality(graph, sequence) -> float`

    Calculates the LZCentrality of a sequence within the given graph's structure.

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

## See Also

- [Tutorials: Diversity Metrics](../tutorials/diversity-metrics.md)
- [How-To: Compare Repertoires](../how-to/repertoire-comparison.md)
- [Concepts: Probability Model](../concepts/probability-model.md)
