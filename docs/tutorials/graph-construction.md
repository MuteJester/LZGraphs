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

# Load your data (ExampleData3 has amino acid + gene columns)
data = pd.read_csv("Examples/ExampleData3.csv")

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
print(graph.initial_state_counts)

# Terminal states (last subpattern of sequences)
print("\nTerminal states (top 5):")
print(graph.terminal_state_counts.head())
```

### V/J Gene Distributions

```python
# Marginal V gene probabilities
print("V gene distribution:")
print(graph.marginal_v_genes)

# Marginal J gene probabilities
print("\nJ gene distribution:")
print(graph.marginal_j_genes)
```

---

## Nucleotide Reading Frame Positional Graph (NDPLZGraph)

The `NDPLZGraph` is designed for nucleotide sequences with reading frame + position encoding.

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
# ['T0_1', 'G1_2', 'TG2_4', 'C1_5', 'CA2_7', 'GC1_9']
```

Each node has the format `{subpattern}{reading_frame}_{position}`, where the reading frame
(0, 1, or 2) indicates the codon position and the suffix is the cumulative sequence position.

---

## Naive LZGraph

The `NaiveLZGraph` uses pure LZ76 decomposition without positional encoding, making it ideal for cross-repertoire comparisons.

### Construction

```python
from LZGraphs import NaiveLZGraph
from LZGraphs.utilities import generate_kmer_dictionary

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

## Sequence Abundance Weighting

All three graph types support **sequence abundance weighting**, which allows you to incorporate clonotype frequency information into the graph construction. Instead of treating each unique sequence equally, abundance weighting ensures that sequences observed more frequently contribute proportionally more to edge weights and transition probabilities.

### Why Use Abundance Weighting?

- **More accurate probability estimates** that reflect true clonal frequencies
- **Better representation of clonal expansion patterns** in the repertoire
- **More realistic sequence generation** via `simulate()` -- generated sequences follow the abundance-weighted distribution
- **Probability models that account for how frequently each sequence was observed**, not just whether it was observed

### AAPLZGraph and NDPLZGraph

For `AAPLZGraph` and `NDPLZGraph`, include an `abundance` column in your input DataFrame. Each sequence will be weighted by its abundance count during graph construction.

```python
import pandas as pd
from LZGraphs import AAPLZGraph

# DataFrame with abundance counts
df = pd.DataFrame({
    'cdr3_amino_acid': ['CASSLAPGATNEKLFF', 'CASSLGQAYEQYF', 'CASSQETQYF'],
    'V': ['TRBV5-1*01', 'TRBV7-2*01', 'TRBV3-1*01'],
    'J': ['TRBJ1-4*01', 'TRBJ2-7*01', 'TRBJ2-5*01'],
    'abundance': [15, 3, 42]  # clonotype counts
})

graph = AAPLZGraph(df)
# Edge weights now reflect abundance-weighted frequencies
```

The same approach works for `NDPLZGraph`:

```python
from LZGraphs import NDPLZGraph

df = pd.DataFrame({
    'cdr3_rearrangement': ['TGTGCCAGCAGTTTAGAG...', 'TGTGCCAGCAGTGACACT...'],
    'V': ['TRBV16-1*01', 'TRBV1-1*01'],
    'J': ['TRBJ1-2*01', 'TRBJ1-5*01'],
    'abundance': [10, 5]
})

graph = NDPLZGraph(df)
```

### NaiveLZGraph

For `NaiveLZGraph`, pass the `abundances` parameter as a list of integers (one per sequence):

```python
from LZGraphs import NaiveLZGraph
from LZGraphs.utilities import generate_kmer_dictionary

sequences = ['ACGTACGT', 'TGCATGCA', 'GGCCTTAA']
abundances = [10, 5, 20]
dictionary = generate_kmer_dictionary(4)

graph = NaiveLZGraph(sequences, dictionary, abundances=abundances)
```

### Effect on Downstream Analysis

When abundance weighting is used:

- **Edge weights** reflect the total abundance-weighted transitions, not just the number of unique sequences sharing an edge
- **`walk_probability()`** returns probabilities that account for clonal expansion
- **`simulate()`** generates sequences with frequencies matching the abundance-weighted model
- **Diversity metrics** computed on the graph reflect the true clonal distribution

!!! tip "When to use abundance weighting"
    Use abundance weighting when your dataset includes clonotype counts and you want the graph to model the **observed repertoire** (including clonal expansions). Omit it when you want to model the **unique sequence diversity** regardless of how many times each sequence was observed.

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
summary = graph.graph_summary()
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
