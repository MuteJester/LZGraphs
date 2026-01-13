# Diversity Metrics

This tutorial covers measuring repertoire diversity using LZGraphs' novel metrics.

## Overview

LZGraphs provides several approaches to quantify repertoire diversity:

| Metric | What it Measures | Use Case |
|--------|------------------|----------|
| **K1000** | Unique patterns in 1000 sequences | Overall diversity |
| **LZCentrality** | Sequence position in repertoire | Sequence rarity |
| **Entropy** | Information content | Graph complexity |
| **Perplexity** | Prediction uncertainty | Model quality |

---

## K-Diversity Metrics

K-diversity measures how many unique LZ76 patterns appear when sampling K sequences.

### K1000 Diversity

The most common diversity measure:

```python
from LZGraphs import K1000_Diversity, AAPLZGraph
import pandas as pd

# Load data
data = pd.read_csv("Examples/ExampleData1.csv")
sequences = data['cdr3_amino_acid'].tolist()

# Calculate K1000
k1000 = K1000_Diversity(
    sequences,
    encoding_function=AAPLZGraph.encode_sequence,
    draws=30  # Number of resampling rounds
)
print(f"K1000 Diversity: {k1000:.1f}")
```

!!! info "Interpretation"
    Higher K1000 values indicate more diverse repertoires. A repertoire with many unique patterns will have a higher K1000 than one dominated by repeated motifs.

### Other K-Diversity Variants

```python
from LZGraphs import K100_Diversity, K500_Diversity, K5000_Diversity

# For smaller repertoires
k100 = K100_Diversity(sequences, AAPLZGraph.encode_sequence)

# For larger repertoires
k5000 = K5000_Diversity(sequences, AAPLZGraph.encode_sequence)

print(f"K100:  {k100:.1f}")
print(f"K5000: {k5000:.1f}")
```

### Adaptive K-Diversity

Automatically select K based on repertoire size:

```python
from LZGraphs import adaptive_K_Diversity

k_adaptive = adaptive_K_Diversity(sequences, AAPLZGraph.encode_sequence)
print(f"Adaptive K-Diversity: {k_adaptive:.1f}")
```

### K-Diversity with Statistics

Get confidence intervals:

```python
from LZGraphs import K_Diversity

result = K_Diversity(
    sequences,
    encoding_function=AAPLZGraph.encode_sequence,
    sample_size=1000,
    draws=100,
    return_stats=True
)

print(f"Mean: {result['mean']:.1f}")
print(f"Std:  {result['std']:.2f}")
print(f"CI:   [{result['ci_low']:.1f}, {result['ci_high']:.1f}]")
```

---

## LZCentrality

LZCentrality measures how central a sequence is within the repertoire structure.

```python
from LZGraphs import LZCentrality, AAPLZGraph
import pandas as pd

# Build graph
data = pd.read_csv("Examples/ExampleData1.csv")
graph = AAPLZGraph(data, verbose=False)

# Calculate centrality for a sequence
sequence = "CASSLEPSGGTDTQYF"
centrality = LZCentrality(graph, sequence)
print(f"LZCentrality of {sequence}: {centrality:.4f}")
```

### Comparing Sequence Centrality

```python
sequences = [
    "CASSLEPSGGTDTQYF",
    "CASSLGQGSTEAFF",
    "CASSXYZRARESEQ"
]

for seq in sequences:
    try:
        cent = LZCentrality(graph, seq)
        print(f"{seq}: {cent:.4f}")
    except:
        print(f"{seq}: Not in graph")
```

!!! tip "LZCentrality interpretation"
    - Higher values = more central (common patterns)
    - Lower values = more peripheral (rare patterns)
    - Zero = sequence not representable in graph

---

## Entropy Metrics

Entropy quantifies the information content of your graph.

### Node Entropy

Measures uncertainty in node (subpattern) distribution:

```python
from LZGraphs import node_entropy

h_nodes = node_entropy(graph)
print(f"Node entropy: {h_nodes:.2f} bits")
```

### Edge Entropy

Measures uncertainty in transitions:

```python
from LZGraphs import edge_entropy

h_edges = edge_entropy(graph)
print(f"Edge entropy: {h_edges:.2f} bits")
```

### Graph Entropy

Combined measure of graph complexity:

```python
from LZGraphs import graph_entropy, normalized_graph_entropy

h_graph = graph_entropy(graph)
h_norm = normalized_graph_entropy(graph)

print(f"Graph entropy: {h_graph:.2f} bits")
print(f"Normalized:    {h_norm:.4f}")
```

---

## Perplexity

Perplexity measures how "surprised" the model is by sequences.

### Sequence Perplexity

```python
from LZGraphs import sequence_perplexity

sequence = "CASSLEPSGGTDTQYF"
perp = sequence_perplexity(graph, sequence)
print(f"Perplexity of {sequence}: {perp:.2f}")
```

!!! info "Perplexity interpretation"
    - Lower perplexity = sequence fits model well
    - Higher perplexity = sequence is unexpected

### Repertoire Perplexity

Average perplexity across all sequences:

```python
from LZGraphs import repertoire_perplexity

sequences = data['cdr3_amino_acid'].tolist()
avg_perp = repertoire_perplexity(graph, sequences)
print(f"Average repertoire perplexity: {avg_perp:.2f}")
```

---

## Comparing Repertoires

### Jensen-Shannon Divergence

Measure similarity between two repertoires:

```python
from LZGraphs import jensen_shannon_divergence

# Build two graphs from different repertoires
data1 = pd.read_csv("Examples/ExampleData1.csv")
data2 = pd.read_csv("Examples/ExampleData2.csv")

graph1 = AAPLZGraph(data1, verbose=False)
graph2 = AAPLZGraph(data2, verbose=False)

# Calculate JS divergence
jsd = jensen_shannon_divergence(graph1, graph2)
print(f"JS Divergence: {jsd:.4f}")
```

!!! info "JS Divergence interpretation"
    - 0 = identical distributions
    - 1 = completely different
    - Symmetric: JSD(A,B) = JSD(B,A)

### Mutual Information for Genes

Measure association between genes and subpatterns:

```python
from LZGraphs import mutual_information_genes

mi_v = mutual_information_genes(graph, gene_type='V')
mi_j = mutual_information_genes(graph, gene_type='J')

print(f"MI (V genes): {mi_v:.4f}")
print(f"MI (J genes): {mi_j:.4f}")
```

---

## Saturation Analysis

Analyze how diversity grows with sample size.

```python
from LZGraphs import NodeEdgeSaturationProbe

# Create saturation curve
probe = NodeEdgeSaturationProbe()
curve = probe.saturation_curve(
    sequences,
    encoding_function=AAPLZGraph.encode_sequence,
    steps=50
)

print(curve.head())
```

### Key Saturation Metrics

```python
# Half-saturation point (sequences needed for 50% of nodes)
half_sat = probe.half_saturation_point(curve)
print(f"Half-saturation: {half_sat} sequences")

# Area under saturation curve
ausc = probe.area_under_curve(curve)
print(f"AUSC: {ausc:.2f}")
```

---

## Complete Example

Compare diversity between two repertoires:

```python
from LZGraphs import (
    AAPLZGraph, K1000_Diversity,
    node_entropy, jensen_shannon_divergence
)
import pandas as pd

# Load two repertoires
rep1 = pd.read_csv("Examples/ExampleData1.csv")
rep2 = pd.read_csv("Examples/ExampleData2.csv")

# Build graphs
g1 = AAPLZGraph(rep1, verbose=False)
g2 = AAPLZGraph(rep2, verbose=False)

# Compare metrics
seq1 = rep1['cdr3_amino_acid'].tolist()
seq2 = rep2['cdr3_amino_acid'].tolist()

print("Repertoire 1 vs Repertoire 2")
print("-" * 40)
print(f"K1000:    {K1000_Diversity(seq1, AAPLZGraph.encode_sequence):.0f} vs "
      f"{K1000_Diversity(seq2, AAPLZGraph.encode_sequence):.0f}")
print(f"Entropy:  {node_entropy(g1):.2f} vs {node_entropy(g2):.2f}")
print(f"JS Div:   {jensen_shannon_divergence(g1, g2):.4f}")
```

---

## Next Steps

- [Visualization Tutorial](visualization.md) - Plot diversity metrics
- [Concepts: Diversity Indices](../concepts/probability-model.md) - Theory behind metrics
- [How-To: Compare Repertoires](../how-to/repertoire-comparison.md) - Detailed comparison workflows
