# Visualization

This tutorial covers creating publication-ready plots for TCR repertoire analysis.

## Overview

LZGraphs provides specialized visualization functions:

| Function | Purpose |
|----------|---------|
| `plot_graph` | Visualize graph structure |
| `plot_ancestor_descendant_curves` | Trace sequence path through graph |
| `plot_possible_paths` | Show branching at each position |
| `plot_gene_node_variability` | V/J gene diversity per node |
| `plot_gene_edge_variability` | V/J gene diversity per edge |

---

## Setup

```python
import pandas as pd
from LZGraphs import AAPLZGraph
from LZGraphs.visualization import (
    plot_graph,
    plot_ancestor_descendant_curves,
    plot_possible_paths,
    plot_gene_node_variability,
    plot_gene_edge_variability
)

# Build a graph
data = pd.read_csv("Examples/ExampleData3.csv")
graph = AAPLZGraph(data, verbose=False)
```

---

## Drawing the Graph

Visualize the graph structure:

```python
plot_graph(graph, file_name='my_lzgraph.png')
```

This generates a PNG image showing the graph structure with nodes representing LZ76 patterns and edges showing observed transitions.

!!! warning "Large graphs"
    For large repertoires, the graph may be too complex to visualize effectively. Consider filtering to a subset of nodes.

---

## Ancestors and Descendants Curves

This plot shows how the number of ancestors (predecessors) and descendants (successors) changes along a sequence path.

```python
sequence = 'CASTPGTASGYTF'
plot_ancestor_descendant_curves(graph, sequence)
```

![Ancestors Descendants Curve](../images/ad_curve_example.png)

### Interpretation

- **Descendants curve** (blue): Number of reachable nodes from each position
- **Ancestors curve** (orange): Number of paths leading to each position
- **Intersection point**: Where the sequence transitions from "common start" to "specific ending"

### Use Cases

- Compare rare vs. common sequences
- Identify motifs that constrain downstream options
- Study sequence "funneling" patterns

---

## Sequence Possible Paths

Shows the number of alternative paths (branching factor) at each position:

```python
sequence = 'CASTPGTASGYTF'
plot_possible_paths(graph, sequence)
```

![Possible Paths Plot](../images/sequence_path_number_example.png)

### Interpretation

- **High values**: Many alternatives at that position (common patterns)
- **Low values**: Few alternatives (rare patterns)
- **Value of 1**: Only one observed continuation

### Correlation with Rarity

Sequences with consistently low path counts are rare in the repertoire and tend to have:
- Lower generation probability
- Higher Levenshtein distance from repertoire mean
- Lower lz_centrality

---

## Genomic Node Variability

Shows V and J gene diversity at each node in a sequence:

```python
sequence = 'CASTPGTASGYTF'
plot_gene_node_variability(graph, sequence)
```

![Node Variability Plot](../images/number_of_vj_at_nodes_example.png)

### Interpretation

- **Bar height**: Number of distinct V/J genes observed at that node
- **High V diversity early**: Expected for V-gene derived regions
- **High J diversity late**: Expected for J-gene derived regions

### Requirements

This function requires gene annotation data (`V` and `J` columns) in your original DataFrame.

---

## Genomic Edge Variability

Shows V and J gene associations for each edge transition:

```python
sequence = 'CASTPGTASGYTF'
plot_gene_edge_variability(graph, sequence)
```

![Edge Variability Plot](../images/number_of_vj_at_edges_example.png)

### Reading the Heatmap

- **Rows**: Gene names (V or J)
- **Columns**: Edge transitions
- **Color intensity**: Probability of that edge given the gene
- **Red gene names**: Gene appears in ALL edges
- **Black cells**: Gene not observed at that edge

### Use Cases

- Identify gene-specific sequence motifs
- Compare gene usage between sequences
- Study CDR3 structure by gene

---

## Customizing Plots

### Saving Figures

```python
import matplotlib.pyplot as plt

# Create the plot
fig = plot_possible_paths(graph, sequence)

# Customize and save
plt.title("Path Variability Analysis")
plt.tight_layout()
plt.savefig("my_analysis.png", dpi=300, bbox_inches='tight')
plt.close()
```

### Batch Processing

```python
sequences = [
    'CASTPGTASGYTF',
    'CASSLEPSGGTDTQYF',
    'CASSLGQGSTEAFF'
]

for i, seq in enumerate(sequences):
    plot_ancestor_descendant_curves(graph, seq)
    plt.savefig(f"ad_curve_{i}.png", dpi=150)
    plt.close()
```

---

## Comparing Sequences

Visualize differences between sequences:

```python
import matplotlib.pyplot as plt

sequences = ['CASTPGTASGYTF', 'CASSLEPSGGTDTQYF']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, seq in zip(axes, sequences):
    plt.sca(ax)
    plot_possible_paths(graph, seq)
    ax.set_title(seq)

plt.tight_layout()
plt.savefig("comparison.png", dpi=300)
```

---

## Saturation Curves

Visualize how diversity grows with sample size:

```python
from LZGraphs import NodeEdgeSaturationProbe
import matplotlib.pyplot as plt

sequences = data['cdr3_amino_acid'].tolist()
probe = NodeEdgeSaturationProbe(node_function='aap')

# Generate curve
curve = probe.saturation_curve(sequences, log_every=100)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(curve['n_sequences'], curve['nodes'], label='Nodes')
plt.plot(curve['n_sequences'], curve['edges'], label='Edges')
plt.xlabel('Number of Sequences')
plt.ylabel('Count')
plt.title('Node/Edge Saturation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("saturation_curve.png", dpi=300)
```

---

## Complete Example

```python
import pandas as pd
import matplotlib.pyplot as plt
from LZGraphs import AAPLZGraph
from LZGraphs.visualization import (
    plot_ancestor_descendant_curves,
    plot_possible_paths,
    plot_gene_node_variability
)

# Load and build
data = pd.read_csv("Examples/ExampleData3.csv")
graph = AAPLZGraph(data, verbose=False)

# Analyze a sequence
sequence = 'CASTPGTASGYTF'

# Create multi-panel figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Ancestors/Descendants
plt.sca(axes[0])
plot_ancestor_descendant_curves(graph, sequence)
axes[0].set_title("Ancestors & Descendants")

# Panel 2: Possible Paths
plt.sca(axes[1])
plot_possible_paths(graph, sequence)
axes[1].set_title("Path Variability")

# Panel 3: Gene Variability
plt.sca(axes[2])
plot_gene_node_variability(graph, sequence)
axes[2].set_title("V/J Gene Diversity")

plt.suptitle(f"Analysis of {sequence}", fontsize=14)
plt.tight_layout()
plt.savefig("complete_analysis.png", dpi=300)
plt.show()
```

---

## Next Steps

- [Examples Gallery](../examples/index.md) - See complete notebooks
- [API: Visualization](../api/utilities.md) - Full function reference
- [How-To: Compare Repertoires](../how-to/repertoire-comparison.md) - Visual comparison workflows
