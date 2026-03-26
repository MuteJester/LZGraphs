---
tags:
  - Comparison
  - Diversity
---

# Compare Repertoires

Learn how to measure similarity and differences between T-cell and B-cell receptor repertoires using graph-based metrics.

## Quick Reference

```python
from LZGraphs import LZGraph, jensen_shannon_divergence

# Compare two graphs
jsd = jensen_shannon_divergence(graph1, graph2)
print(f"JS Divergence: {jsd:.4f}")  # 0 = identical, 1 = completely different
```

## Jensen-Shannon Divergence (JSD)

The Jensen-Shannon Divergence is the standard metric in LZGraphs for comparing the structural and statistical similarity of two repertoires. It is symmetric, bounded between 0 and 1, and captures differences in both node (subpattern) usage and transition frequencies.

### Interpretation

| JSD Value | Interpretation |
|-----------|----------------|
| 0.0 | Identical distributions |
| 0.0-0.1 | Very similar (e.g., technical replicates) |
| 0.1-0.3 | Moderately similar (e.g., healthy cohort) |
| 0.3-0.5 | Different (e.g., distinct individuals) |
| 0.5-1.0 | Very different (e.g., different species or chains) |

### Basic Comparison

```python
from LZGraphs import LZGraph, jensen_shannon_divergence

# 1. Build graphs for two samples
graph_a = LZGraph(sequences_a, variant='aap')
graph_b = LZGraph(sequences_b, variant='aap')

# 2. Calculate divergence
jsd = jensen_shannon_divergence(graph_a, graph_b)
print(f"JSD: {jsd:.4f}")
```

## Comparing Multiple Repertoires

To compare a cohort of repertoires, calculate all pairwise distances and visualize them as a heatmap.

```python
from itertools import combinations
import numpy as np

names = list(graphs.keys())
n = len(names)
dist_matrix = np.zeros((n, n))

# Fill distance matrix
for i, name_i in enumerate(names):
    for j, name_j in enumerate(names):
        if i < j:
            d = jensen_shannon_divergence(graphs[name_i], graphs[name_j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

# Optionally visualize with matplotlib/seaborn
# import seaborn as sns; import matplotlib.pyplot as plt
# sns.heatmap(dist_matrix, xticklabels=names, yticklabels=names, annot=True)
# plt.title("Repertoire Pairwise JSD"); plt.show()
```

## Comparing Diversity & Complexity

Beyond JSD, you can compare repertoires by their intrinsic graph metrics.

### Diversity Profiles (Hill Numbers)

Compare how diversity scales across different orders of \(\alpha\):

```python
orders = [0, 1, 2, 3]
hills_a = graph_a.hill_numbers(orders)
hills_b = graph_b.hill_numbers(orders)

for a, v_a, v_b in zip(orders, hills_a, hills_b):
    print(f"D({a}): Sample A = {v_a:.2e}, Sample B = {v_b:.2e}")
```

### Generation Probability (PGEN) Distribution

Compare the mean and standard deviation of sequence generation probabilities:

```python
moments_a = graph_a.pgen_moments()
moments_b = graph_b.pgen_moments()

print(f"Sample A Mean Log-Pgen: {moments_a['mean']:.2f}")
print(f"Sample B Mean Log-Pgen: {moments_b['mean']:.2f}")
```

## Cross-Repertoire Probability

Check how likely sequences from one repertoire are in another. This is useful for identifying "public" sequences or repertoire-specific motifs.

```python
# Sample 100 sequences from Repertoire A
sample_seqs = sequences_a[:100]

# Score them against Repertoire B
log_probs_in_b = graph_b.lzpgen(sample_seqs)

print(f"Avg log-Pgen of A sequences in B: {log_probs_in_b.mean():.2f}")
```

## Gene Usage Comparison

If the graphs were built with V/J gene data, you can compare their marginal distributions:

```python
v_a = graph_a.v_marginals
v_b = graph_b.v_marginals

# Find genes with largest frequency difference
diff = {g: v_a.get(g, 0) - v_b.get(g, 0) for g in set(v_a) | set(v_b)}
sorted_diff = sorted(diff.items(), key=lambda x: abs(x[1]), reverse=True)

print("Top V gene differences:")
for gene, d in sorted_diff[:5]:
    print(f"{gene}: {d:+.4f}")
```

## Next Steps

- [Personalize Graphs](posterior-personalization.md) — Compare individuals to a population baseline
- [Diversity Metrics](../tutorials/diversity-metrics.md) — Hill numbers and diversity profiles
- [API: LZGraph](../api/lzgraph.md) — Complete method reference
