# Graph Variants

LZGraphs provides three graph variants, each optimized for different analysis scenarios. All are handled by the unified `LZGraph` class.

## Overview Comparison

| Feature | AAP Variant | NDP Variant | Naive Variant |
|---------|-------------|-------------|---------------|
| **Input** | Amino acids | Nucleotides | Any strings |
| **Position encoding** | Single (end) | Reading frame + position | None |
| **V/J gene support** | Yes | Yes | Yes |
| **Alphabet size** | 20 AA | 4 NT | Configurable |
| **Graph complexity** | Medium | High | Low |
| **Memory usage** | Medium | High | Low |
| **Best for** | Most TCR analysis | Nucleotide-level | Motif discovery |

## AAP Variant

**Amino Acid Positional**

### When to Use

- Analyzing amino acid CDR3 sequences
- Need V/J gene annotations
- Standard repertoire analysis
- Moderate-sized repertoires

### Node Format

```
{subpattern}_{position}
```

Example: `SL_6` means pattern "SL" ending at position 6 in the sequence (position 1 is the internal `@` sentinel).

### Example

```python
from LZGraphs import LZGraph

# Build from list of sequences
graph = LZGraph(sequences, variant='aap')
```

### Key Features

- **Position-aware**: Distinguishes the same pattern at different sequence positions.
- **Gene-aware**: Edges carry V/J gene transition statistics.
- **Compact**: The 20-letter amino acid alphabet results in manageable graph sizes for most repertoires.

---

## NDP Variant

**Nucleotide Double Positional**

### When to Use

- Analyzing nucleotide sequences
- Need fine-grained positional information
- Studying codon usage or reading frames
- Have memory for larger graphs

### Node Format

```
{subpattern}{reading_frame}_{position}
```

Example: `TG0_4` means pattern "TG" starting at reading frame 0 and ending at position 4.

### Example

```python
from LZGraphs import LZGraph

graph = LZGraph(nt_sequences, variant='ndp')
```

### Key Features

- **Reading frame + position**: Captures codon context and pattern boundaries.
- **Higher resolution**: Provides more detailed structural information than AAP.
- **Large graphs**: Due to the 4-letter nucleotide alphabet and frame encoding, NDP graphs grow faster than other variants.

---

## Naive Variant

**Position-free**

### When to Use

- General motif discovery
- Simple sequence complexity metrics
- Repertoires where position is less critical
- Memory-constrained environments

### Node Format

```
{subpattern}
```

Just the raw LZ76 subpattern without position information.

### Example

```python
from LZGraphs import LZGraph

graph = LZGraph(sequences, variant='naive')
```

### Key Features

- **Position-free**: Merges the same subpattern across all positions into a single node.
- **Simplest structure**: Smallest number of nodes and edges.
- **Feature alignment**: Use `reference_graph.feature_aligned(query_graph)` to project any graph into a shared feature space for machine learning.

---

## Feature Alignment for ML

In older versions, `NaiveLZGraph` used a "fixed dictionary" for consistent features. In the current version, you can project **any** graph into the space of a **reference graph** to get consistent, high-dimensional feature vectors.

```python
# Build a reference from a large dataset
ref = LZGraph(large_repertoire, variant='aap')

# Project a new sample into the reference node space
# Returns a numpy array of shape (ref.n_nodes,)
vector = ref.feature_aligned(LZGraph(new_sample))
```

---

## Memory and Performance

### Graph Size Estimates

For a repertoire of N sequences with average length L:

| Variant | Nodes | Edges |
|---------|-------|-------|
| **AAP** | O(20 × L) | O(N × L) |
| **NDP** | O(4 × 3 × L) | O(N × L) |
| **Naive** | O(Unique subpatterns) | O(Unique transitions) |

*Note: In practice, node counts are much lower than these upper bounds due to shared patterns.*

### Practical Recommendations

```python
# Check your graph size
print(f"Nodes: {graph.n_nodes}")
print(f"Edges: {graph.n_edges}")
```

## Next Steps

- [Probability Model](probability-model.md) - How graphs calculate probabilities
- [LZ76 Algorithm](lz76-algorithm.md) - Understand the encoding
- [How-To: Repertoire Comparison](../how-to/repertoire-comparison.md) - Compare different graphs
