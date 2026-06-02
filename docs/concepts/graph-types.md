---
tags:
  - LZGraph
  - FlashBackGraph
---

# Graph Variants

LZGraphs provides **two graph families**, built on different representations of the same LZ76 idea:

- **The `LZGraph` family**: three variants (`aap`, `ndp`, `naive`) of a single unified class, using a coarsened LZ76 dictionary. Analytics that depend on the full sequence distribution (diversity, simulation) are estimated efficiently, with simulation used where a closed form is intractable.
- **`FlashBackGraph`**: a separate class that builds a strictly Markovian DAG from the FlashBack decomposition. Because it is Markovian, diversity, entropy, Hill numbers, path counts, and PGEN are all computed **exactly** by forward dynamic programming.

Most of this page covers the three `LZGraph` variants; the [FlashBackGraph](#flashbackgraph) section below covers the Markovian alternative.

## Overview Comparison (LZGraph variants)

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

---

## FlashBackGraph

**The Markovian alternative**

`FlashBackGraph` is a **separate class** (not a `variant=` of `LZGraph`). Instead of a coarsened LZ76 dictionary, it builds a strictly Markovian DAG from the **FlashBack decomposition**: the algorithm recursively peels matching character runs from both ends of the sentinel-wrapped sequence, and each resulting token becomes a node. Edges are transitions between consecutive tokens.

Because the graph is Markovian, the full sequence distribution factorises over edges, so diversity, Shannon entropy, Hill numbers, path counts, and generation probability are all computed **exactly** by forward dynamic programming. No Monte Carlo simulation is needed.

### When to Use

- You need **exact**, reproducible diversity and entropy (no sampling variance)
- You want self-calibrated per-sequence **anomaly scoring** (SCALE)
- You want exact PGEN, dynamic range, and the top-K most/least probable sequences
- You are scoring sequences against a large reference repertoire

### Node Format

FlashBack tokens (plus `@` and `$` sentinels marking sequence start/end). The `variant` property is always `'flashback'`.

### Example

```python
from LZGraphs import FlashBackGraph

fb = FlashBackGraph(sequences)

# Exact analytics, computed by forward DP, not simulation
print(f"Effective diversity: {fb.effective_diversity():.2f}")
print(f"Inverse Simpson D(2): {fb.hill_number(2):.2f}")

# Exact generation probability, and the self-calibrated SCALE anomaly score
log_p = fb.pgen('CASSLEPSGGTDTQYF')
cal = fb.calibrate_scale(seed=0)                # self-calibrate once
score = fb.scale_score('CASSLEPSGGTDTQYF', cal) # SCALE: higher = more anomalous
```

### Key Features

- **Exact analytics**: Hill numbers, entropy, path counts, and PGEN via forward DP, no approximation.
- **Anomaly scoring**: SCALE (`calibrate_scale()` + `scale_score()`) gives a self-calibrated, length-invariant surprise score for sequences against the repertoire model.
- **Graph algebra & Bayesian updates**: `union` / `intersection` / `difference` / `weighted_merge`, plus `posterior()` and `without()` for personalization and leave-donor-out construction.
- **Streaming construction**: `FlashBackStream` builds incrementally from open-ended sources with instant running counts.

## LZGraph family vs FlashBackGraph

| | `LZGraph` family (`aap`/`ndp`/`naive`) | `FlashBackGraph` |
|---|---|---|
| **Class** | `LZGraph(..., variant=...)` | `FlashBackGraph(...)` |
| **Representation** | Coarsened LZ76 dictionary | Strictly Markovian DAG of FlashBack tokens |
| **Diversity / entropy** | Estimated | **Exact** (forward DP) |
| **Sequence generation** | Random walk simulation | Random walk + exact top-K |
| **Anomaly scoring** | - | SCALE (`scale_score()`) |
| **Gene (V/J) annotations** | Yes | No |
| **Best for** | General repertoire analysis, gene-aware modelling, ML features | Exact diversity/PGEN, anomaly detection, reference scoring |

See the [FlashBackGraph API reference](../api/flashback-graph.md) for the full method list.

## Next Steps

- [Probability Model](probability-model.md) - How graphs calculate probabilities
- [LZ76 Algorithm](lz76-algorithm.md) - Understand the encoding
- [FlashBackGraph API](../api/flashback-graph.md) - Full reference for the Markovian variant
- [How-To: Repertoire Comparison](../how-to/repertoire-comparison.md) - Compare different graphs
