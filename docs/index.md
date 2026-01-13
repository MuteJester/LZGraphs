# LZGraphs

<div class="badges" markdown>
[![PyPI version](https://badge.fury.io/py/LZGraphs.svg)](https://badge.fury.io/py/LZGraphs)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://mutejester.github.io/LZGraphs/)
</div>

<p align="center">
  <img src="images/lzglogo2.png" alt="LZGraphs Logo" width="300">
</p>

**LZGraphs** is a Python library for analyzing T-cell receptor (TCR) repertoires using Lempel-Ziv 76 compression-based graph representations. It provides a novel approach to sequence analysis that doesn't rely on alignment or genotype references.

---

## Why LZGraphs?

Traditional TCR repertoire analysis methods often struggle with:

- **Alignment dependencies** - requiring reference sequences
- **Computational complexity** - O(n²) pairwise comparisons
- **Loss of positional information** - treating sequences as bags of k-mers

LZGraphs solves these problems by encoding sequences as walks through directed graphs, capturing both the **content** and **structure** of repertoires in a computationally efficient way.

---

## Key Features

<div class="grid" markdown>

<div class="card" markdown>
### :material-graph: Graph Representations
Three specialized graph types for different analysis needs: **AAPLZGraph** for amino acids, **NDPLZGraph** for nucleotides, and **NaiveLZGraph** for general sequences.
</div>

<div class="card" markdown>
### :material-chart-line: Diversity Metrics
Novel diversity indices including **K1000** and **LZCentrality** that capture repertoire complexity through graph topology.
</div>

<div class="card" markdown>
### :material-dna: Gene Analysis
Built-in V/J gene annotation support for genomic-aware sequence generation and gene usage analysis.
</div>

<div class="card" markdown>
### :material-chart-scatter-plot: Visualization
Publication-ready plots for sequence analysis, including path variability, genomic heatmaps, and saturation curves.
</div>

</div>

---

## Quick Start

### Installation

```bash
pip install LZGraphs
```

**Requirements:** Python 3.9 or higher

### Your First Graph

```python
import pandas as pd
from LZGraphs import AAPLZGraph

# Load your TCR repertoire data
data = pd.DataFrame({
    'cdr3_amino_acid': ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF', 'CASSLEPQTFTDTFFF'],
    'V': ['TRBV16-1*01', 'TRBV1-1*01', 'TRBV16-1*01'],
    'J': ['TRBJ1-2*01', 'TRBJ1-5*01', 'TRBJ2-7*01']
})

# Build the graph
graph = AAPLZGraph(data, verbose=True)

# Calculate sequence probability
sequence = "CASSLEPSGGTDTQYF"
pgen = graph.walk_probability(AAPLZGraph.encode_sequence(sequence))
print(f"P(gen) = {pgen:.2e}")
```

---

## Documentation Overview

<div class="quick-links" markdown>

[:material-rocket-launch: **Getting Started**](getting-started/index.md)
New to LZGraphs? Start here for installation and basic usage.

[:material-school: **Tutorials**](tutorials/index.md)
Step-by-step guides for common analysis tasks.

[:material-lightbulb: **Concepts**](concepts/index.md)
Understand the theory behind LZGraphs.

[:material-wrench: **How-To Guides**](how-to/index.md)
Task-oriented guides for specific operations.

[:material-notebook: **Examples**](examples/index.md)
Interactive Jupyter notebooks with real data.

[:material-api: **API Reference**](api/index.md)
Complete reference for all classes and functions.

</div>

---

## Citation

If you use LZGraphs in your research, please cite our paper:

```bibtex
@article{lzgraphs2024,
  title={LZGraphs: A Novel Approach for T-Cell Receptor Repertoire Analysis},
  author={Konstantinovsky, Thomas and others},
  journal={...},
  year={2024}
}
```

See the [Citation page](resources/citation.md) for more details.

---

## Connect With Us

- :fontawesome-brands-github: [GitHub Repository](https://github.com/MuteJester/LZGraphs)
- :material-bug: [Report Issues](https://github.com/MuteJester/LZGraphs/issues)
- :material-email: [Contact](mailto:thomaskon90@gmail.com)
