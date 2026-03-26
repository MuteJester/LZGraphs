---
tags:
  - Comparison
  - Construction
---

# Graph Algebra

LZGraphs supports set-theoretic operations on graphs: **union**, **intersection**, **difference**, and **weighted merge**. These let you combine, compare, and decompose repertoires at the structural level — far more expressive than just computing a scalar divergence.

---

## When to use graph algebra

| Scenario | Operation | What it gives you |
|----------|-----------|-------------------|
| Merge two timepoint samples | Union (`a \| b`) | A combined graph with summed edge counts |
| Find shared repertoire structure | Intersection (`a & b`) | Only edges present in both graphs |
| Find what's unique to a sample | Difference (`a - b`) | Edges in A that aren't in B |
| Combine cohort + individual with weights | Weighted merge | Custom linear combination of two graphs |
| Adapt a population model to one patient | Posterior | Bayesian update (see [Personalize Graphs](posterior-personalization.md)) |

All operations produce a **new `LZGraph`** — the original graphs are never modified.

---

## Union: combining repertoires

**Union** sums the edge counts from both graphs. If an edge exists in both, its count is the sum; if it's in only one, it's kept as-is.

```python
from LZGraphs import LZGraph

# Two time points from the same donor
graph_t0 = LZGraph(sequences_baseline, variant='aap')
graph_t1 = LZGraph(sequences_followup, variant='aap')

# Combine
combined = graph_t0 | graph_t1
# equivalent to: combined = graph_t0.union(graph_t1)

print(f"T0: {graph_t0.n_nodes} nodes, {graph_t0.n_edges} edges")
print(f"T1: {graph_t1.n_nodes} nodes, {graph_t1.n_edges} edges")
print(f"Combined: {combined.n_nodes} nodes, {combined.n_edges} edges")
```

The combined graph has:

- **All nodes** from either input (union of node sets)
- **All edges** from either input, with **summed counts**
- **Re-normalized edge weights** (so probabilities sum to 1 at each node)

!!! tip "Use case: pooling replicates"
    If you have technical replicates of the same sample, union gives you a single graph with more statistical power. The summed counts improve the probability estimates without introducing bias.

---

## Intersection: finding shared structure

**Intersection** keeps only the edges that exist in **both** graphs, using the **minimum** count for each shared edge.

```python
# What structure do two donors share?
shared = donor_a & donor_b
# equivalent to: shared = donor_a.intersection(donor_b)

print(f"Donor A:  {donor_a.n_edges} edges")
print(f"Donor B:  {donor_b.n_edges} edges")
print(f"Shared:   {shared.n_edges} edges")
print(f"Overlap:  {shared.n_edges / min(donor_a.n_edges, donor_b.n_edges):.1%}")
```

The intersection graph represents the **public structural core** — transitions that both repertoires use. You can simulate from it to generate sequences that are plausible in both repertoires.

```python
# Generate "public-like" sequences
public_seqs = shared.simulate(1000, seed=42)
```

---

## Difference: finding what's unique

**Difference** (`A - B`) subtracts B's edge counts from A. Edges where A's count exceeds B's are kept (with reduced count); edges where B's count is equal or greater are removed.

```python
# What's unique to the disease sample?
disease_specific = disease_graph - healthy_graph
# equivalent to: disease_specific = disease_graph.difference(healthy_graph)

print(f"Disease:    {disease_graph.n_edges} edges")
print(f"Healthy:    {healthy_graph.n_edges} edges")
print(f"Unique:     {disease_specific.n_edges} edges")
```

The difference graph highlights **repertoire-specific structure** — transitions that are enriched in one sample relative to another.

!!! warning "Direction matters"
    Difference is **not symmetric**: `A - B` is different from `B - A`. Think of it as "what's in A but not in B."

### Finding disease-associated motifs

```python
# Build difference graphs in both directions
disease_only = disease_graph - healthy_graph
healthy_only = healthy_graph - disease_graph

# Score a candidate sequence against each
seq = "CASSLGQAYEQYF"
print(f"Disease-enriched model score: {disease_only.lzpgen(seq):.2f}")
print(f"Health-enriched model score:  {healthy_only.lzpgen(seq):.2f}")
```

A sequence that scores high in `disease_only` but low in `healthy_only` uses transitions that are specifically enriched in the disease repertoire.

---

## Weighted merge: custom combinations

**Weighted merge** creates a linear combination: $\alpha \cdot A + \beta \cdot B$. Each graph's edge counts are scaled by its weight before summation.

```python
# Give disease sample twice the weight
merged = healthy_graph.weighted_merge(disease_graph, alpha=1.0, beta=2.0)
```

Useful scenarios:

- **Cohort averaging**: merge multiple samples with equal weight (`alpha=1/n`)
- **Emphasis weighting**: upweight a sample you trust more
- **Smoothing**: blend a small sample with a population reference

### Building a cohort-average graph

```python
# Start with the first graph
cohort = graphs[0]

# Merge in each subsequent graph with equal weight
for g in graphs[1:]:
    cohort = cohort.weighted_merge(g, alpha=1.0, beta=1.0)

print(f"Cohort graph: {cohort.n_nodes} nodes, {cohort.n_edges} edges")
```

!!! info "Weighted merge with alpha=1, beta=1 is equivalent to union"
    The only difference is that `weighted_merge` lets you scale the counts before combining.

---

## Chaining operations

Since every operation returns a new `LZGraph`, you can chain them:

```python
# (A union B) minus C
result = (graph_a | graph_b) - graph_c

# Public core across three donors
shared_ab = donor_a & donor_b
shared_abc = shared_ab & donor_c
```

---

## Operator summary

| Python operator | Method | Edge count rule |
|:---:|:---|:---|
| `a \| b` | `a.union(b)` | $c_e = a_e + b_e$ |
| `a & b` | `a.intersection(b)` | $c_e = \min(a_e, b_e)$, only if both > 0 |
| `a - b` | `a.difference(b)` | $c_e = \max(a_e - b_e, 0)$, drop if 0 |
| — | `a.weighted_merge(b, α, β)` | $c_e = \alpha \cdot a_e + \beta \cdot b_e$ |

All operations require both graphs to use the **same variant** (both `'aap'`, both `'ndp'`, etc.). Mixing variants raises an error.

---

## Complete example: longitudinal analysis

Track how a repertoire changes over time by quantifying what's gained and lost:

```python
from LZGraphs import LZGraph, jensen_shannon_divergence

# Three time points
g0 = LZGraph(seqs_week0, variant='aap')
g1 = LZGraph(seqs_week4, variant='aap')
g2 = LZGraph(seqs_week8, variant='aap')

# Overall divergence over time
print(f"JSD(week0, week4): {jensen_shannon_divergence(g0, g1):.4f}")
print(f"JSD(week0, week8): {jensen_shannon_divergence(g0, g2):.4f}")

# What's new at week 8 that wasn't there at baseline?
new_structure = g2 - g0
print(f"New edges at week 8: {new_structure.n_edges}")

# What was lost from baseline?
lost_structure = g0 - g2
print(f"Lost edges by week 8: {lost_structure.n_edges}")

# Stable core across all three time points
stable = g0 & g1 & g2
print(f"Stable core: {stable.n_edges} edges "
      f"({stable.n_edges / g0.n_edges:.0%} of baseline)")
```

---

## See Also

- [Compare Repertoires](repertoire-comparison.md) — JSD and cross-scoring workflows
- [Personalize Graphs](posterior-personalization.md) — Bayesian posterior (a more principled way to combine prior + data)
- [API: LZGraph](../api/lzgraph.md#graph-algebra) — method signatures
