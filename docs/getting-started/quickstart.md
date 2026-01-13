# Quick Start

Build your first LZGraph in 5 minutes. This guide shows the essential workflow for TCR repertoire analysis.

## Step 1: Import and Load Data

```python
import pandas as pd
from LZGraphs import AAPLZGraph

# Load your repertoire data
# Your CSV should have a 'cdr3_amino_acid' column (or 'cdr3_rearrangement' for nucleotides)
data = pd.read_csv("your_repertoire.csv")

# Preview the data
print(data.head())
```

Example data format:

| cdr3_amino_acid | V | J |
|-----------------|---|---|
| CASSLEPSGGTDTQYF | TRBV16-1*01 | TRBJ1-2*01 |
| CASSDTSGGTDTQYF | TRBV1-1*01 | TRBJ1-5*01 |
| CASSLEPQTFTDTFFF | TRBV16-1*01 | TRBJ2-7*01 |

!!! tip "V and J columns are optional"
    Gene annotation columns (`V` and `J`) enable gene-aware analysis but aren't required for basic usage.

## Step 2: Build a Graph

```python
# Build the graph (with gene annotation)
graph = AAPLZGraph(data, verbose=True)
```

Output:
```
Gene Information Loaded.. |  0.01  Seconds
Graph Constructed.. |  0.94  Seconds
Graph Metadata Derived.. |  0.94  Seconds
...
LZGraph Created Successfully.. |  1.37  Seconds
```

## Step 3: Explore the Graph

```python
# Basic graph statistics
print(f"Nodes: {graph.graph.number_of_nodes()}")
print(f"Edges: {graph.graph.number_of_edges()}")
print(f"Sequences: {sum(graph.lengths.values())}")

# View length distribution
print("\nSequence length distribution:")
print(graph.lengths)
```

## Step 4: Calculate Sequence Probability

Every sequence has a generation probability based on the repertoire:

```python
# Encode a sequence
sequence = "CASSLEPSGGTDTQYF"
encoded = AAPLZGraph.encode_sequence(sequence)
print(f"Encoded: {encoded}")

# Calculate probability
pgen = graph.walk_probability(encoded)
print(f"P(gen) = {pgen:.2e}")

# Use log probability for very small values
log_pgen = graph.walk_probability(encoded, use_log=True)
print(f"log P(gen) = {log_pgen:.2f}")
```

## Step 5: Generate New Sequences

Generate sequences that follow the statistical patterns of your repertoire:

```python
# Generate a sequence with gene constraints
walk, v_gene, j_gene = graph.genomic_random_walk()
print(f"Generated walk: {walk}")
print(f"V gene: {v_gene}, J gene: {j_gene}")

# Convert back to sequence
sequence = ''.join([AAPLZGraph.clean_node(node) for node in walk])
print(f"Generated sequence: {sequence}")
```

## Step 6: Calculate Diversity

```python
from LZGraphs import K1000_Diversity

# Calculate K1000 diversity index
sequences = data['cdr3_amino_acid'].tolist()
k1000 = K1000_Diversity(sequences, AAPLZGraph.encode_sequence, draws=30)
print(f"K1000 Diversity: {k1000:.1f}")
```

## Complete Example

Here's everything together:

```python
import pandas as pd
from LZGraphs import AAPLZGraph, K1000_Diversity

# Load data
data = pd.read_csv("repertoire.csv")

# Build graph
graph = AAPLZGraph(data, verbose=True)

# Analyze a sequence
seq = "CASSLEPSGGTDTQYF"
pgen = graph.walk_probability(AAPLZGraph.encode_sequence(seq))
print(f"\n{seq}: P(gen) = {pgen:.2e}")

# Generate new sequence
walk, v, j = graph.genomic_random_walk()
new_seq = ''.join([AAPLZGraph.clean_node(n) for n in walk])
print(f"Generated: {new_seq} ({v}, {j})")

# Calculate diversity
k1000 = K1000_Diversity(data['cdr3_amino_acid'].tolist(),
                        AAPLZGraph.encode_sequence, draws=30)
print(f"K1000: {k1000:.1f}")
```

## What's Next?

Now that you've built your first graph, explore:

- [First Steps](first-steps.md) - Learn which graph type to use
- [Graph Construction Tutorial](../tutorials/graph-construction.md) - Detailed construction options
- [Concepts: Graph Types](../concepts/graph-types.md) - Understand the different graph types
