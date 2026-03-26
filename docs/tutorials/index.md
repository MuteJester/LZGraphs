# Tutorials

Welcome to the LZGraphs tutorials. These step-by-step guides will help you master TCR repertoire analysis using LZGraphs.

## Learning Path

We recommend following these tutorials in order:

<div class="grid" markdown>

<div class="card" markdown>
### 1. [Graph Construction](graph-construction.md)
**Beginner** · 15 minutes

Learn to build AAP, NDP, and Naive graph variants from your data.
</div>

<div class="card" markdown>
### 2. [Sequence Analysis](sequence-analysis.md)
**Beginner** · 20 minutes

Calculate sequence probabilities, explore graph structure, and simulate new sequences.
</div>

<div class="card" markdown>
### 3. [Diversity Metrics](diversity-metrics.md)
**Intermediate** · 15 minutes

Measure repertoire complexity using k-diversity, Hill numbers, and occupancy models.
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
import csv

# Load sample data
with open("examples/ExampleData1.csv") as f:
    sequences = [row['cdr3_amino_acid'] for row in csv.DictReader(f)]
```

## Quick Reference

| Tutorial | Topics Covered |
|----------|----------------|
| [Graph Construction](graph-construction.md) | LZGraph variants, gene annotations, abundance weighting |
| [Sequence Analysis](sequence-analysis.md) | lzpgen, simulate, lz76_decompose |
| [Diversity Metrics](diversity-metrics.md) | k_diversity, hill_numbers, predicted_richness |

## Next Steps

After completing the tutorials:

- Explore [Concepts](../concepts/index.md) for deeper understanding
- Check [How-To Guides](../how-to/index.md) for specific tasks
- Browse [Examples](../examples/index.md) for complete notebooks
