---
description: Build your first LZGraph in 5 minutes — install, construct a graph, score sequences, simulate, and measure diversity.
search:
  boost: 2
---

# Quick Start

Build your first LZGraph in 5 minutes. This guide shows the essential workflow for TCR repertoire analysis.

## Step 1: Import and Load Data

```python
from LZGraphs import LZGraph

# Option 1: Plain list of sequences
sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", "CASSLEPQTFTDTFFF"]

# Option 2: Load from CSV
import csv
with open("your_repertoire.csv") as f:
    sequences = [row['cdr3_amino_acid'] for row in csv.DictReader(f)]
```

!!! note "Input Format"
    LZGraphs expects a list of strings for sequences. If you have gene annotations or abundances, they should also be provided as parallel lists.

## Step 2: Build a Graph

=== "Basic Graph"

    ```python
    from LZGraphs import LZGraph

    sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", "CASSLEPQTFTDTFFF"]
    graph = LZGraph(sequences, variant='aap') # (1)
    ```

    1. `variant='aap'` uses amino acid positional encoding — the best choice for CDR3 amino acid sequences. See [Graph Variants](../concepts/graph-types.md) for alternatives.

=== "With Abundances"

    ```python
    # Weight each sequence by its clonotype count
    counts = [150, 42, 10]
    graph = LZGraph(sequences, abundances=counts, variant='aap')
    ```

=== "With V/J Genes"

    ```python
    # Provide gene annotations alongside sequences
    v_genes = ["TRBV16-1*01", "TRBV1-1*01", "TRBV16-1*01"]
    j_genes = ["TRBJ1-2*01", "TRBJ1-5*01", "TRBJ2-7*01"]
    
    graph = LZGraph(sequences, v_genes=v_genes, j_genes=j_genes, variant='aap')
    print(f"Gene data available: {graph.has_gene_data}")  # True
    ```

## Step 3: Explore the Graph

```python
# Basic graph statistics
print(f"Nodes: {graph.n_nodes}")
print(f"Edges: {graph.n_edges}")
print(f"Total sequences: {graph.n_sequences}")

# View length distribution {length: count}
print(graph.length_distribution)
```

## Step 4: Sequence Generation Probability (LZPGEN)

Every sequence has a generation probability under the LZ-constrained model:

```python
# Calculate log probability
log_p = graph.lzpgen("CASSLEPSGGTDTQYF")
print(f"log P(gen) = {log_p:.2f}")

# Multiple sequences at once (returns numpy array)
log_ps = graph.lzpgen(["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF"])
```

## Step 5: Simulate New Sequences

Generate sequences that follow the statistical patterns of your repertoire:

```python
# Generate 1000 sequences
results = graph.simulate(1000, seed=42)

# results is iterable and contains the generated strings
for seq in results[:5]:
    print(seq)

# If the graph has gene data, you can constrain the simulation
custom_results = graph.simulate(10, v_gene="TRBV16-1*01", j_gene="TRBJ1-2*01")
```

## Step 6: Diversity & Richness

```python
# Hill diversity number D(1) (Effective Diversity)
print(f"Effective Diversity: {graph.effective_diversity():.1f}")

# Predicted richness at depth 100,000
print(f"Predicted richness: {graph.predicted_richness(100000):.1f}")
```

## Complete Example

```python
from LZGraphs import LZGraph

# Sample data
sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", "CASSLEPQTFTDTFFF"]
v_genes   = ["TRBV16-1*01", "TRBV1-1*01", "TRBV16-1*01"]
j_genes   = ["TRBJ1-2*01", "TRBJ1-5*01", "TRBJ2-7*01"]

# 1. Build graph
graph = LZGraph(sequences, v_genes=v_genes, j_genes=j_genes, variant='aap')

# 2. Probability
seq = "CASSLEPSGGTDTQYF"
print(f"{seq} log P: {graph.lzpgen(seq):.2f}")

# 3. Simulate
simulated = graph.simulate(5, seed=42)
print(f"Simulated: {list(simulated)}")

# 4. Diversity
print(f"D(2): {graph.hill_number(2.0):.1f}")
```

## What's Next?

- [First Steps](first-steps.md) - Learn which graph variant to choose
- [How-To Guides](../how-to/index.md) - Task-specific recipes
- [Concepts](../concepts/index.md) - Theory behind LZ76 graphs
