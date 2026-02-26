# Utilities

Helper functions and classes for LZGraphs analysis.

## Core Utilities

### lempel_ziv_decomposition

Decompose a sequence using LZ76 algorithm.

```python
from LZGraphs.utilities import lempel_ziv_decomposition

patterns = lempel_ziv_decomposition("CASSLEPSGGTDTQYF")
print(patterns)
# ['C', 'A', 'S', 'SL', 'E', 'P', 'SG', 'G', 'T', 'D', 'TQ', 'Y', 'F']
```

### generate_kmer_dictionary

Generate all possible DNA k-mer patterns up to length k.

```python
from LZGraphs.utilities import generate_kmer_dictionary

# All DNA k-mers up to length 6
dictionary = generate_kmer_dictionary(6)
print(f"Patterns: {len(dictionary)}")  # 5460
```

!!! note
    Uses a hardcoded DNA alphabet (`A`, `T`, `G`, `C`). Designed for use with `NaiveLZGraph`.

---

## LZBOW (Bag of Words)

Vectorize sequences using LZ76 patterns.

```python
from LZGraphs import LZBOW, NDPLZGraph

# Create vectorizer
vectorizer = LZBOW(encoding_function=NDPLZGraph.encode_sequence)

# Fit on repertoire
vectorizer.fit(sequences)

# Transform new sequences
bow_vector = vectorizer.transform(new_sequences)
```

### Methods

| Method | Description |
|--------|-------------|
| `fit(sequences)` | Build dictionary from sequences |
| `transform(sequences)` | Convert to BOW vectors |
| `fit_transform(sequences)` | Fit and transform |

### Combining BOW Objects

```python
# Combine dictionaries from multiple repertoires
bow1 = LZBOW(encoding_function=NDPLZGraph.encode_sequence)
bow1.fit(sequences1)

bow2 = LZBOW(encoding_function=NDPLZGraph.encode_sequence)
bow2.fit(sequences2)

combined = bow1 + bow2
```

!!! note "Class: LZBOW"
    Bag-of-Words encoder using LZ76 decomposition patterns as vocabulary.

    **Constructor:** `LZBOW(encoding_function=lempel_ziv_decomposition)`

    **Key Methods:**

    - `fit(sequences)` - Build vocabulary from sequences
    - `transform(sequences, normalize=False)` - Convert to BOW vectors
    - `fit_transform(sequences)` - Fit and transform in one step
    - `load_from(other_bow)` - Load vocabulary from another LZBOW

---

## NodeEdgeSaturationProbe

Analyze how diversity grows with sample size.

```python
from LZGraphs import NodeEdgeSaturationProbe

# Create probe with encoding type
probe = NodeEdgeSaturationProbe(node_function='aap')

# Generate saturation curve
curve = probe.saturation_curve(sequences, log_every=100)
print(curve.head())
# Columns: n_sequences, nodes, edges
```

### Constructor

```python
NodeEdgeSaturationProbe(node_function='naive', log_level=1, verbose=False)
```

`node_function` can be `'naive'`, `'ndp'`, `'aap'`, or a custom callable.

### Methods

| Method | Description |
|--------|-------------|
| `saturation_curve(sequence_list, log_every=100)` | Generate node/edge counts vs sample size |
| `half_saturation_point(sequence_list, log_every=50, metric='nodes')` | Find 50% saturation point |
| `area_under_saturation_curve(sequence_list, log_every=100, normalize=True, metric='nodes')` | Calculate AUSC metric |
| `diversity_profile(sequence_list, log_every=100)` | Full diversity profile |

### Example

```python
from LZGraphs import NodeEdgeSaturationProbe
import matplotlib.pyplot as plt

probe = NodeEdgeSaturationProbe(node_function='aap')
curve = probe.saturation_curve(sequences, log_every=100)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(curve['n_sequences'], curve['nodes'], label='Nodes')
plt.plot(curve['n_sequences'], curve['edges'], label='Edges')
plt.xlabel('Number of Sequences')
plt.ylabel('Count')
plt.legend()
plt.savefig('saturation.png')
```

---

## Visualization Functions

### plot_graph

Visualize graph structure.

```python
from LZGraphs.visualization import plot_graph

plot_graph(graph, file_name='graph.png')
```

### plot_ancestor_descendant_curves

Plot ancestors and descendants along a sequence.

```python
from LZGraphs.visualization import plot_ancestor_descendant_curves

plot_ancestor_descendant_curves(graph, "CASSLEPSGGTDTQYF")
```

### plot_possible_paths

Plot branching factor at each position.

```python
from LZGraphs.visualization import plot_possible_paths

plot_possible_paths(graph, "CASSLEPSGGTDTQYF")
```

### plot_gene_node_variability

Plot V/J gene diversity per node.

```python
from LZGraphs.visualization import plot_gene_node_variability

plot_gene_node_variability(graph, "CASSLEPSGGTDTQYF")
```

### plot_gene_edge_variability

Plot V/J gene associations per edge.

```python
from LZGraphs.visualization import plot_gene_edge_variability

plot_gene_edge_variability(graph, "CASSLEPSGGTDTQYF")
```

---

## Graph Operations

### graph_summary (method)

Get summary statistics for a graph. This is a **method** on all graph classes, not an importable function.

```python
summary = graph.graph_summary()
print(summary)
# Returns dict with: Chromatic Number, Number of Isolates,
# Max In Deg, Max Out Deg, Number of Edges
```

### graph_union

Combine two graphs.

```python
from LZGraphs import graph_union

combined = graph_union(graph1, graph2)
```

---

## See Also

- [Tutorials: Visualization](../tutorials/visualization.md)
- [Concepts: LZ76 Algorithm](../concepts/lz76-algorithm.md)
