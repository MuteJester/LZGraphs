# Quick Start

Build your first LZGraph in 5 minutes. This guide shows the essential workflow for TCR repertoire analysis.

## Step 1: Import and Load Data

```python
from LZGraphs import AAPLZGraph

# Option 1: Plain list of sequences (no pandas needed)
sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]

# Option 2: Load from CSV with pandas
import pandas as pd
data = pd.read_csv("your_repertoire.csv")
sequences = data['cdr3_amino_acid'].tolist()
```

!!! note "pandas is optional"
    LZGraphs does not require pandas. You can pass a plain `list[str]` to any graph constructor. pandas is only needed if you want to load data from CSV files or pass DataFrames directly.

Example data format:

| cdr3_amino_acid | V | J |
|-----------------|---|---|
| CASSLEPSGGTDTQYF | TRBV16-1*01 | TRBJ1-2*01 |
| CASSDTSGGTDTQYF | TRBV1-1*01 | TRBJ1-5*01 |
| CASSLEPQTFTDTFFF | TRBV16-1*01 | TRBJ2-7*01 |

!!! tip "V and J columns are optional"
    Gene annotation columns (`V` and `J`) enable gene-aware analysis but aren't required for basic usage.

## Step 2: Build a Graph

=== "List of sequences"

    ```python
    # Pass a plain list — no DataFrame needed
    sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]
    graph = AAPLZGraph(sequences, verbose=True)
    ```

=== "List + abundance"

    ```python
    # Weight each sequence by its clonotype count
    sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]
    counts    = [150, 42, ...]
    graph = AAPLZGraph(sequences, abundances=counts, verbose=True)
    ```

=== "List + V/J genes"

    ```python
    # Provide gene annotations alongside sequences
    sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]
    v_genes   = ["TRBV16-1*01", "TRBV1-1*01", ...]
    j_genes   = ["TRBJ1-2*01", "TRBJ1-5*01", ...]
    graph = AAPLZGraph(sequences, v_genes=v_genes, j_genes=j_genes, verbose=True)
    print(f"Gene data available: {graph.has_gene_data}")  # True
    ```

=== "pandas DataFrame"

    ```python
    # Traditional DataFrame input (all columns in one table)
    data = pd.read_csv("repertoire.csv")
    # Expected columns: cdr3_amino_acid, (optional) V, J, abundance
    graph = AAPLZGraph(data, verbose=True)
    ```

!!! info "When do you need V/J genes?"
    Gene columns are only required for gene-aware methods like `genomic_random_walk()` and `vj_combination_random_walk()`. All core features — probability, generation, diversity — work without them.

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
# Calculate probability (accepts a raw sequence string)
pgen = graph.walk_probability("CASSLEPSGGTDTQYF")
print(f"P(gen) = {pgen:.2e}")

# Use log probability for very small values
log_pgen = graph.walk_probability("CASSLEPSGGTDTQYF", use_log=True)
print(f"log P(gen) = {log_pgen:.2f}")
```

!!! tip "Encoding under the hood"
    `walk_probability` accepts either a raw sequence string or a pre-encoded list of nodes
    (from `AAPLZGraph.encode_sequence()`). Passing a string is simpler for most use cases.

## Step 5: Generate New Sequences

Generate sequences that follow the statistical patterns of your repertoire:

```python
# Generate a sequence with gene constraints
walk, v_gene, j_gene = graph.genomic_random_walk()
print(f"Generated walk: {walk}")
print(f"V gene: {v_gene}, J gene: {j_gene}")

# Convert back to sequence
sequence = ''.join([AAPLZGraph.extract_subpattern(node) for node in walk])
print(f"Generated sequence: {sequence}")
```

## Step 6: Calculate Diversity

```python
from LZGraphs import k1000_diversity

# Calculate K1000 diversity index
sequences = data['cdr3_amino_acid'].tolist()
k1000 = k1000_diversity(sequences, AAPLZGraph.encode_sequence, draws=30)
print(f"K1000 Diversity: {k1000:.1f}")
```

## Complete Example

Here's everything together:

```python
import pandas as pd
from LZGraphs import AAPLZGraph, k1000_diversity

# Load data
data = pd.read_csv("repertoire.csv")

# Build graph
graph = AAPLZGraph(data, verbose=True)

# Analyze a sequence
seq = "CASSLEPSGGTDTQYF"
pgen = graph.walk_probability(seq)
print(f"\n{seq}: P(gen) = {pgen:.2e}")

# Generate new sequence
walk, v, j = graph.genomic_random_walk()
new_seq = ''.join([AAPLZGraph.extract_subpattern(n) for n in walk])
print(f"Generated: {new_seq} ({v}, {j})")

# Calculate diversity
k1000 = k1000_diversity(data['cdr3_amino_acid'].tolist(),
                        AAPLZGraph.encode_sequence, draws=30)
print(f"K1000: {k1000:.1f}")
```

## What's Next?

Now that you've built your first graph, explore:

- [First Steps](first-steps.md) - Learn which graph type to use
- [Graph Construction Tutorial](../tutorials/graph-construction.md) - Detailed construction options
- [Concepts: Graph Types](../concepts/graph-types.md) - Understand the different graph types
