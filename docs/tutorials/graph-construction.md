# Graph Construction

This tutorial covers building the three types of LZGraphs from TCR repertoire data.

## Overview

LZGraphs transforms sequences into directed graph representations where:

- **Nodes** represent subpatterns from LZ76 decomposition
- **Edges** represent observed transitions between subpatterns
- **Edge weights** encode transition probabilities

## Amino Acid Positional Graph (AAPLZGraph)

The `AAPLZGraph` is optimized for amino acid CDR3 sequences with positional encoding.

### Basic Construction

```python
from LZGraphs import AAPLZGraph
import pandas as pd

# Load your data
data = pd.read_csv("Examples/ExampleData1.csv")

# Build the graph
graph = AAPLZGraph(data, verbose=True)
```

**Expected output:**
```
Gene Information Loaded.. |  0.01  Seconds
Graph Constructed.. |  0.94  Seconds
Graph Metadata Derived.. |  0.94  Seconds
Individual Subpattern Empirical Probability Derived.. |  0.98  Seconds
Graph Edge Weight Normalized.. |  1.0  Seconds
Graph Edge Gene Weights Normalized.. |  1.13  Seconds
Terminal State Map Derived.. |  1.2  Seconds
LZGraph Created Successfully.. |  1.37  Seconds
```

### Required Data Format

Your DataFrame must have a `cdr3_amino_acid` column:

| cdr3_amino_acid | V | J |
|-----------------|---|---|
| CASSLEPSGGTDTQYF | TRBV16-1*01 | TRBJ1-2*01 |
| CASSDTSGGTDTQYF | TRBV1-1*01 | TRBJ1-5*01 |

!!! info "V and J columns"
    V and J gene annotation columns are optional but enable gene-aware features like `genomic_random_walk()`.

### Exploring Graph Structure

```python
# Get all nodes and edges
nodes = list(graph.nodes)
edges = list(graph.edges)

print(f"Nodes: {len(nodes)}")
print(f"Edges: {len(edges)}")
print(f"First 10 nodes: {nodes[:10]}")
print(f"First 10 edges: {edges[:10]}")
```

**Output:**
```
Nodes: 1523
Edges: 8492
First 10 nodes: ['C_1', 'A_2', 'S_3', 'SQ_5', 'Q_6', 'G_7', 'R_8', 'D_9', 'T_10', 'QY_12']
First 10 edges: [('C_1', 'A_2'), ('C_1', 'T_2'), ('C_1', 'V_2'), ...]
```

### Length Distribution

```python
# Sequence length distribution
print(graph.lengths)
```

**Output:**
```python
{13: 2973, 15: 5075, 14: 4412, 16: 2862, 12: 1147, ...}
```

### Initial and Terminal States

```python
# Initial states (first subpattern of sequences)
print("Initial states:")
print(graph.initial_states)

# Terminal states (last subpattern of sequences)
print("\nTerminal states (top 5):")
print(graph.terminal_states.head())
```

### V/J Gene Distributions

```python
# Marginal V gene probabilities
print("V gene distribution:")
print(graph.marginal_vgenes)

# Marginal J gene probabilities
print("\nJ gene distribution:")
print(graph.marginal_jgenes)
```

---

## Nucleotide Double Positional Graph (NDPLZGraph)

The `NDPLZGraph` is designed for nucleotide sequences with double positional encoding.

### Construction

```python
from LZGraphs import NDPLZGraph

# Data must have 'cdr3_rearrangement' column
data = pd.read_csv("nucleotide_repertoire.csv")
graph = NDPLZGraph(data, verbose=True)
```

### Data Format

| cdr3_rearrangement | V | J |
|-------------------|---|---|
| TGTGCCAGCAGTTTAGAG... | TRBV16-1*01 | TRBJ1-2*01 |

### Encoding Example

```python
sequence = "TGTGCCAGC"
encoded = NDPLZGraph.encode_sequence(sequence)
print(encoded)
# ['T_1_1', 'G_2_2', 'T_3_3', 'G_4_4', 'C_5_5', 'C_6_6', 'A_7_7', 'G_8_8', 'C_9_9']
```

The double position encoding (`_start_end`) captures subpattern boundaries precisely.

---

## Naive LZGraph

The `NaiveLZGraph` uses pure LZ76 decomposition without positional encoding, making it ideal for cross-repertoire comparisons.

### Construction

```python
from LZGraphs import NaiveLZGraph
from LZGraphs.Utilities import generate_kmer_dictionary

# Generate a shared dictionary
dictionary = generate_kmer_dictionary(6)
print(f"Dictionary size: {len(dictionary)}")  # 5460 patterns

# Build graph from sequence list
sequences = data['cdr3_rearrangement'].tolist()
graph = NaiveLZGraph(sequences, dictionary, verbose=True)
```

### Why Use a Shared Dictionary?

Using the same dictionary across multiple repertoires ensures:

1. **Consistent feature dimensions** for machine learning
2. **Comparable graphs** for cross-repertoire analysis
3. **Fixed node set** regardless of repertoire content

### Feature Extraction

```python
# Extract eigenvector centrality features
features = graph.eigenvector_centrality()
print(f"Feature vector length: {len(features)}")
print(pd.Series(features).head(10))
```

**Output:**
```
A         3.009520e-01
T         1.183398e-01
G         1.186366e-01
C         2.461758e-01
AA        1.252643e-01
...
```

---

## Construction Options

### Verbose Mode

Control output verbosity:

```python
# Silent construction
graph = AAPLZGraph(data, verbose=False)

# With progress output
graph = AAPLZGraph(data, verbose=True)
```

### Graph Summary

Get a quick overview of your graph:

```python
from LZGraphs import graph_summary

summary = graph_summary(graph)
print(summary)
```

---

## Saving and Loading Graphs

Persist your graphs for later use:

```python
# Save
graph.save("my_graph.pkl")

# Load
loaded_graph = AAPLZGraph.load("my_graph.pkl")
```

See [How-To: Serialization](../how-to/serialization.md) for more details.

---

## Next Steps

- [Sequence Analysis Tutorial](sequence-analysis.md) - Work with your constructed graph
- [Concepts: Graph Types](../concepts/graph-types.md) - Deep dive into graph differences
- [API: AAPLZGraph](../api/aaplzgraph.md) - Complete class reference
