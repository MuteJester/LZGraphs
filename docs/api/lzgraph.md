---
description: Complete API reference for the LZGraph class — construction, scoring, simulation, diversity, graph algebra, and IO.
search:
  boost: 2
---

# LZGraph

The primary class for building, analyzing, and using LZ76-based repertoire graphs.

## Constructor

```python
LZGraph(
    sequences,
    *,
    variant='aap',
    abundances=None,
    v_genes=None,
    j_genes=None,
    smoothing=0.0
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | CDR3 sequences (amino acid or nucleotide) |
| `variant` | `str` | `'aap'`, `'ndp'`, or `'naive'` (default: `'aap'`) |
| `abundances` | `list[int]` | Optional frequency weights per sequence |
| `v_genes` | `list[str]` | Optional V gene names |
| `j_genes` | `list[str]` | Optional J gene names |
| `smoothing` | `float` | Laplace smoothing alpha (default: 0.0) |

## Core Methods

### lzpgen

```python
lzpgen(sequences, log=True)
```
Calculate the generation probability of one or more sequences.

- **Parameters**: `sequences` (str or list[str]), `log` (bool)
- **Returns**: `float` (if single) or `np.ndarray` (if list)

### simulate

```python
simulate(n, *, v_gene=None, j_gene=None, sample_genes=False, seed=None)
```
Batch-generate `n` new sequences from the model.

- **Returns**: [SimulationResult](simulation-result.md)

### posterior

```python
posterior(sequences, *, abundances=None, kappa=1.0)
```
Create a personalized posterior graph using this graph as a prior.

- **Returns**: A new `LZGraph` instance.

## Diversity & Analytics

### effective_diversity

```python
effective_diversity()
```
Returns \(e^H\) where H is Shannon entropy. Equivalent to Hill number \(D(1)\).

### hill_number

```python
hill_number(alpha)
```
Returns Hill diversity number \(D(\alpha)\).

### hill_numbers

```python
hill_numbers(orders)
```
Returns an array of Hill numbers for multiple orders.

### hill_curve

```python
hill_curve(orders=None)
```
Returns a dict with `orders` and `values` for plotting a diversity profile.

### pgen_moments

```python
pgen_moments()
```
Returns a dict with `mean`, `variance`, `std`, `skewness`, `kurtosis` of the log-PGEN distribution.

### path_entropy_rate

```python
path_entropy_rate(sequences)
```
Estimated bits per subpattern step from a list of sequences.

## Occupancy & Prediction

### predicted_richness

```python
predicted_richness(depth)
```
Expected number of distinct sequences at a given sequencing depth.

### predicted_overlap

```python
predicted_overlap(d_i, d_j)
```
Expected number of shared sequences between two samples of depths \(d_i\) and \(d_j\).

### predict_sharing

```python
predict_sharing(draw_counts, max_k=None)
```
Predict the sharing spectrum across a cohort of donors.

## Graph Algebra

| Operation | Method | Result |
|-----------|--------|--------|
| Union | `a | b` or `a.union(b)` | Sum edge counts |
| Intersection | `a & b` or `a.intersection(b)` | Shared structure, min counts |
| Difference | `a - b` or `a.difference(b)` | Subtract edge counts |
| Weighted Merge | `a.weighted_merge(b, α, β)` | Linear combination \( \alpha A + \beta B \) |

## Features & Adjacency

### feature_aligned

```python
feature_aligned(query)
```
Project a query graph into this graph's node space. Returns a `np.ndarray`.

### feature_stats

```python
feature_stats()
```
Returns a 15-element statistical vector describing the graph.

### feature_mass_profile

```python
feature_mass_profile(max_pos=30)
```
Position-based mass distribution profile.

### adjacency_csr

```python
adjacency_csr()
```
Returns a dictionary of numpy arrays representing the graph in CSR format.

### successors

```python
successors(node_label)
```
Returns a list of `(target, weight, count)` for a given node.

## IO

### save

```python
save(path)
```
Save to `.lzg` binary format.

### load (classmethod)

```python
LZGraph.load(path)
```
Load from a `.lzg` binary file.

## Attributes

### Basic
| Attribute | Type | Description |
|-----------|------|-------------|
| `n_nodes` | `int` | Total number of nodes (including sentinels) |
| `n_edges` | `int` | Total number of edges (including sentinels) |
| `n_sequences` | `int` | Number of input sequences |
| `variant` | `str` | `'aap'`, `'ndp'`, or `'naive'` |
| `is_dag` | `bool` | True if the graph is a Directed Acyclic Graph |
| `path_count` | `int` | Combinatorial size (total unique paths) |

### Structure
| Attribute | Type | Description |
|-----------|------|-------------|
| `nodes` | `list[str]` | Node labels (excluding sentinels) |
| `all_nodes` | `list[str]` | All node labels (including @ and $) |
| `edges` | `list[tuple]` | `(src, dst, weight, count)` tuples (no sentinels) |
| `n_initial` | `int` | Number of initial states |
| `n_terminal` | `int` | Number of terminal nodes |
| `density` | `float` | Graph density (0 to 1) |
| `out_degrees` | `np.ndarray`| Out-degree of each node |
| `in_degrees` | `np.ndarray` | In-degree of each node |
| `length_distribution`| `dict` | `{length: count}` mapping |

### Genes
| Attribute | Type | Description |
|-----------|------|-------------|
| `has_gene_data`| `bool` | True if gene annotations are available |
| `v_genes` | `list[str]` | List of V gene names |
| `j_genes` | `list[str]` | List of J gene names |
| `v_marginals` | `dict` | V gene marginal distribution |
| `j_marginals` | `dict` | J gene marginal distribution |
| `vj_distribution`| `list` | Joint VJ distribution |
