# Compare Repertoires

Learn how to measure similarity and differences between TCR repertoires.

## Quick Reference

```python
from LZGraphs import jensen_shannon_divergence

# Compare two graphs
jsd = jensen_shannon_divergence(graph1, graph2)
print(f"JS Divergence: {jsd:.4f}")  # 0 = identical, 1 = completely different
```

## Building Graphs for Comparison

```python
from LZGraphs import AAPLZGraph
import pandas as pd

# Load two repertoires
data1 = pd.read_csv("repertoire1.csv")
data2 = pd.read_csv("repertoire2.csv")

# Build graphs
graph1 = AAPLZGraph(data1, verbose=False)
graph2 = AAPLZGraph(data2, verbose=False)
```

## Jensen-Shannon Divergence

The most common method for comparing repertoires:

```python
from LZGraphs import jensen_shannon_divergence

jsd = jensen_shannon_divergence(graph1, graph2)
print(f"JS Divergence: {jsd:.4f}")
```

### Interpretation

| JSD Value | Interpretation |
|-----------|----------------|
| 0.0 | Identical distributions |
| 0.0-0.1 | Very similar |
| 0.1-0.3 | Moderately similar |
| 0.3-0.5 | Different |
| 0.5-1.0 | Very different |

### Properties

- **Symmetric**: JSD(A,B) = JSD(B,A)
- **Bounded**: Always between 0 and 1
- **Metric**: Satisfies triangle inequality (when square-rooted)

## Comparing Multiple Repertoires

### Pairwise Comparison

```python
from itertools import combinations
import pandas as pd

repertoires = {
    'healthy1': graph1,
    'healthy2': graph2,
    'disease1': graph3,
    'disease2': graph4
}

# Calculate all pairwise distances
results = []
for (name1, g1), (name2, g2) in combinations(repertoires.items(), 2):
    jsd = jensen_shannon_divergence(g1, g2)
    results.append({
        'repertoire1': name1,
        'repertoire2': name2,
        'jsd': jsd
    })

df = pd.DataFrame(results)
print(df.sort_values('jsd'))
```

### Distance Matrix

```python
import numpy as np

names = list(repertoires.keys())
n = len(names)
dist_matrix = np.zeros((n, n))

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        if i < j:
            jsd = jensen_shannon_divergence(
                repertoires[name1],
                repertoires[name2]
            )
            dist_matrix[i, j] = jsd
            dist_matrix[j, i] = jsd

# Create DataFrame
dist_df = pd.DataFrame(dist_matrix, index=names, columns=names)
print(dist_df)
```

### Heatmap Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(dist_df, annot=True, cmap='RdYlBu_r', vmin=0, vmax=1)
plt.title('Repertoire Similarity (JSD)')
plt.tight_layout()
plt.savefig('repertoire_heatmap.png', dpi=300)
```

## Comparing Diversity Metrics

### K1000 Comparison

```python
from LZGraphs import K1000_Diversity, AAPLZGraph

def compare_diversity(data1, data2, name1="Rep1", name2="Rep2"):
    seqs1 = data1['cdr3_amino_acid'].tolist()
    seqs2 = data2['cdr3_amino_acid'].tolist()

    k1000_1 = K1000_Diversity(seqs1, AAPLZGraph.encode_sequence, draws=30)
    k1000_2 = K1000_Diversity(seqs2, AAPLZGraph.encode_sequence, draws=30)

    print(f"{name1} K1000: {k1000_1:.1f}")
    print(f"{name2} K1000: {k1000_2:.1f}")
    print(f"Difference: {abs(k1000_1 - k1000_2):.1f}")

compare_diversity(data1, data2, "Healthy", "Disease")
```

### Entropy Comparison

```python
from LZGraphs import node_entropy, edge_entropy, graph_entropy

def compare_entropy(graph1, graph2, name1="Rep1", name2="Rep2"):
    metrics = {}

    for name, g in [(name1, graph1), (name2, graph2)]:
        metrics[name] = {
            'node_entropy': node_entropy(g),
            'edge_entropy': edge_entropy(g),
            'graph_entropy': graph_entropy(g)
        }

    df = pd.DataFrame(metrics).T
    print(df)
    return df

compare_entropy(graph1, graph2, "Healthy", "Disease")
```

## Cross-Repertoire Analysis

### Sequence Probability Across Repertoires

Check how likely sequences from one repertoire are in another:

```python
def cross_probability(sequences, source_graph, target_graph):
    """Calculate sequence probabilities in target graph."""
    results = []

    for seq in sequences:
        try:
            encoded = AAPLZGraph.encode_sequence(seq)
            source_prob = source_graph.walk_probability(encoded, use_log=True)
            target_prob = target_graph.walk_probability(encoded, use_log=True)

            results.append({
                'sequence': seq,
                'source_log_p': source_prob,
                'target_log_p': target_prob,
                'diff': source_prob - target_prob
            })
        except:
            pass

    return pd.DataFrame(results)

# Sample sequences from repertoire 1
sample_seqs = data1['cdr3_amino_acid'].sample(100).tolist()

# Calculate probabilities in both graphs
cross_df = cross_probability(sample_seqs, graph1, graph2)
print(cross_df.describe())
```

### Repertoire-Specific Sequences

Find sequences that are specific to one repertoire:

```python
def find_specific_sequences(sequences, graph1, graph2, threshold=-5):
    """Find sequences specific to graph1 (not in graph2)."""
    specific = []

    for seq in sequences:
        try:
            enc = AAPLZGraph.encode_sequence(seq)
            p1 = graph1.walk_probability(enc, use_log=True)
            p2 = graph2.walk_probability(enc, use_log=True)

            # Specific if much more likely in graph1
            if p1 - p2 > threshold:
                specific.append({
                    'sequence': seq,
                    'log_p_graph1': p1,
                    'log_p_graph2': p2,
                    'specificity': p1 - p2
                })
        except:
            pass

    return pd.DataFrame(specific).sort_values('specificity', ascending=False)

# Find repertoire 1 specific sequences
specific_seqs = find_specific_sequences(
    data1['cdr3_amino_acid'].tolist(),
    graph1, graph2
)
print(specific_seqs.head(10))
```

## Gene Usage Comparison

```python
def compare_gene_usage(graph1, graph2, name1="Rep1", name2="Rep2"):
    """Compare V/J gene usage between repertoires."""

    # V genes
    v1 = graph1.marginal_vgenes
    v2 = graph2.marginal_vgenes

    # Align indices
    all_v = set(v1.index) | set(v2.index)
    v_comparison = pd.DataFrame({
        name1: v1.reindex(all_v).fillna(0),
        name2: v2.reindex(all_v).fillna(0)
    })
    v_comparison['diff'] = v_comparison[name1] - v_comparison[name2]

    # J genes
    j1 = graph1.marginal_jgenes
    j2 = graph2.marginal_jgenes

    all_j = set(j1.index) | set(j2.index)
    j_comparison = pd.DataFrame({
        name1: j1.reindex(all_j).fillna(0),
        name2: j2.reindex(all_j).fillna(0)
    })
    j_comparison['diff'] = j_comparison[name1] - j_comparison[name2]

    return v_comparison, j_comparison

v_comp, j_comp = compare_gene_usage(graph1, graph2, "Healthy", "Disease")
print("Top V gene differences:")
print(v_comp.sort_values('diff', key=abs, ascending=False).head())
```

## Complete Comparison Pipeline

```python
def full_repertoire_comparison(data1, data2, name1="Rep1", name2="Rep2"):
    """Complete comparison of two repertoires."""

    # Build graphs
    print("Building graphs...")
    graph1 = AAPLZGraph(data1, verbose=False)
    graph2 = AAPLZGraph(data2, verbose=False)

    # Basic stats
    print(f"\n{'='*50}")
    print("BASIC STATISTICS")
    print(f"{'='*50}")
    print(f"{name1}: {data1.shape[0]} sequences, {graph1.graph.number_of_nodes()} nodes")
    print(f"{name2}: {data2.shape[0]} sequences, {graph2.graph.number_of_nodes()} nodes")

    # Divergence
    print(f"\n{'='*50}")
    print("DIVERGENCE")
    print(f"{'='*50}")
    jsd = jensen_shannon_divergence(graph1, graph2)
    print(f"Jensen-Shannon Divergence: {jsd:.4f}")

    # Diversity
    print(f"\n{'='*50}")
    print("DIVERSITY")
    print(f"{'='*50}")
    seqs1 = data1['cdr3_amino_acid'].tolist()
    seqs2 = data2['cdr3_amino_acid'].tolist()
    k1 = K1000_Diversity(seqs1, AAPLZGraph.encode_sequence, draws=30)
    k2 = K1000_Diversity(seqs2, AAPLZGraph.encode_sequence, draws=30)
    print(f"{name1} K1000: {k1:.1f}")
    print(f"{name2} K1000: {k2:.1f}")

    # Entropy
    print(f"\n{'='*50}")
    print("ENTROPY")
    print(f"{'='*50}")
    print(f"{name1} node entropy: {node_entropy(graph1):.2f}")
    print(f"{name2} node entropy: {node_entropy(graph2):.2f}")

    return graph1, graph2, jsd

# Run comparison
g1, g2, jsd = full_repertoire_comparison(data1, data2, "Healthy", "Disease")
```

## Next Steps

- [Tutorials: Diversity Metrics](../tutorials/diversity-metrics.md) - More metrics
- [Tutorials: Visualization](../tutorials/visualization.md) - Visualize comparisons
- [API: Metrics](../api/metrics.md) - All comparison functions
