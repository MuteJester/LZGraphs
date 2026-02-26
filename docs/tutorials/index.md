# Tutorials

Welcome to the LZGraphs tutorials. These step-by-step guides will help you master TCR repertoire analysis using LZGraphs.

## Learning Path

We recommend following these tutorials in order:

<div class="grid" markdown>

<div class="card" markdown>
### 1. [Graph Construction](graph-construction.md)
**Beginner** · 15 minutes

Learn to build AAPLZGraph, NDPLZGraph, and NaiveLZGraph from your data.
</div>

<div class="card" markdown>
### 2. [Sequence Analysis](sequence-analysis.md)
**Beginner** · 20 minutes

Calculate sequence probabilities, explore graph structure, and generate new sequences.
</div>

<div class="card" markdown>
### 3. [Diversity Metrics](diversity-metrics.md)
**Intermediate** · 15 minutes

Measure repertoire diversity using k1000_diversity, lz_centrality, and entropy metrics.
</div>

<div class="card" markdown>
### 4. [Visualization](visualization.md)
**Intermediate** · 20 minutes

Create publication-ready plots for sequence and repertoire analysis.
</div>

</div>

## Prerequisites

Before starting, ensure you have:

- [x] Installed LZGraphs ([Installation Guide](../getting-started/installation.md))
- [x] Basic Python knowledge
- [x] Sample data to work with (or use our [example datasets](../examples/index.md))

## Sample Data

All tutorials use example data included with LZGraphs:

```python
import pandas as pd

# Amino acid sequences with gene annotations (for AAPLZGraph)
data = pd.read_csv("Examples/ExampleData3.csv")

# Nucleotide sequences (for NDPLZGraph)
data_nt = pd.read_csv("Examples/ExampleData2.csv")
```

## Quick Reference

| Tutorial | Topics Covered |
|----------|----------------|
| [Graph Construction](graph-construction.md) | AAPLZGraph, NDPLZGraph, NaiveLZGraph, gene annotation |
| [Sequence Analysis](sequence-analysis.md) | walk_probability, random_walk, encode_sequence, extract_subpattern |
| [Diversity Metrics](diversity-metrics.md) | k1000_diversity, lz_centrality, node_entropy, edge_entropy |
| [Visualization](visualization.md) | Ancestors/descendants plots, path variability, genomic heatmaps |

## Next Steps

After completing the tutorials:

- Explore [Concepts](../concepts/index.md) for deeper understanding
- Check [How-To Guides](../how-to/index.md) for specific tasks
- Browse [Examples](../examples/index.md) for complete notebooks
