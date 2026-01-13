# Utilities

Helper functions and classes for LZGraphs analysis.

## Core Utilities

### lempel_ziv_decomposition

Decompose a sequence using LZ76 algorithm.

```python
from LZGraphs.Utilities import lempel_ziv_decomposition

patterns = lempel_ziv_decomposition("CASSLEPSGGTDTQYF")
print(patterns)
# ['C', 'A', 'S', 'SL', 'E', 'P', 'SG', 'G', 'T', 'D', 'TQ', 'Y', 'F']
```

### generate_kmer_dictionary

Generate all possible patterns up to length k.

```python
from LZGraphs.Utilities import generate_kmer_dictionary

# For nucleotides
dictionary = generate_kmer_dictionary(6)
print(f"Patterns: {len(dictionary)}")  # 5460

# For amino acids (custom alphabet)
aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
aa_dict = generate_kmer_dictionary(4, alphabet=aa_alphabet)
```

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
from LZGraphs import NodeEdgeSaturationProbe, AAPLZGraph

probe = NodeEdgeSaturationProbe()

# Generate saturation curve
curve = probe.saturation_curve(
    sequences,
    encoding_function=AAPLZGraph.encode_sequence,
    steps=50
)

print(curve.head())
```

### Methods

| Method | Description |
|--------|-------------|
| `saturation_curve()` | Generate node/edge counts vs sample size |
| `half_saturation_point()` | Find 50% saturation point |
| `area_under_curve()` | Calculate AUSC metric |
| `diversity_profile()` | Full diversity profile |

### Example

```python
from LZGraphs import NodeEdgeSaturationProbe
import matplotlib.pyplot as plt

probe = NodeEdgeSaturationProbe()
curve = probe.saturation_curve(sequences, AAPLZGraph.encode_sequence)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(curve['sequences'], curve['nodes'], label='Nodes')
plt.plot(curve['sequences'], curve['edges'], label='Edges')
plt.xlabel('Number of Sequences')
plt.ylabel('Count')
plt.legend()
plt.savefig('saturation.png')
```

!!! note "Class: NodeEdgeSaturationProbe"
    Analyzes how graph complexity grows with sample size.

    **Methods:**

    - `saturation_curve(sequences, encoding_function, steps)` - Generate saturation data
    - `half_saturation_point(curve)` - Find 50% saturation
    - `area_under_curve(curve)` - Calculate AUSC metric

---

## Visualization Functions

### draw_graph

Visualize graph structure.

```python
from LZGraphs.Visualization import draw_graph

draw_graph(graph, file_name='graph.png')
```

### ancestors_descendants_curves_plot

Plot ancestors and descendants along a sequence.

```python
from LZGraphs.Visualization import ancestors_descendants_curves_plot

ancestors_descendants_curves_plot(graph, "CASSLEPSGGTDTQYF")
```

### sequence_possible_paths_plot

Plot branching factor at each position.

```python
from LZGraphs.Visualization import sequence_possible_paths_plot

sequence_possible_paths_plot(graph, "CASSLEPSGGTDTQYF")
```

### sequence_genomic_node_variability_plot

Plot V/J gene diversity per node.

```python
from LZGraphs.Visualization import sequence_genomic_node_variability_plot

sequence_genomic_node_variability_plot(graph, "CASSLEPSGGTDTQYF")
```

### sequence_genomic_edges_variability_plot

Plot V/J gene associations per edge.

```python
from LZGraphs.Visualization import sequence_genomic_edges_variability_plot

sequence_genomic_edges_variability_plot(graph, "CASSLEPSGGTDTQYF")
```

---

## Graph Operations

### graph_summary

Get summary statistics for a graph.

```python
from LZGraphs import graph_summary

summary = graph_summary(graph)
print(summary)
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
