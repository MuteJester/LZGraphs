<p align="center">

[![PyPI version](https://img.shields.io/pypi/v/LZGraphs.svg)](https://pypi.org/project/LZGraphs/)
[![Python versions](https://img.shields.io/pypi/pyversions/LZGraphs.svg)](https://pypi.org/project/LZGraphs/)
[![CI/CD](https://github.com/MuteJester/LZGraphs/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/MuteJester/LZGraphs/actions/workflows/ci-cd.yml)
[![License](https://img.shields.io/github/license/MuteJester/LZGraphs.svg)](https://github.com/MuteJester/LZGraphs/blob/master/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/LZGraphs.svg)](https://pypi.org/project/LZGraphs/)

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

</p>


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/MuteJester/LZGraphs">
    <img src="https://github.com/MuteJester/LZGraphs/blob/master/misc/lzglogo2.png" alt="Logo" width="480" height="330">
  </a>

  <h2 align="center">LZGraphs</h2>

  <p align="center">
    LZ76 Graphs and Applications in Immunology
    <br />
    <a href="https://MuteJester.github.io/LZGraphs/"><strong>Explore the docs &raquo;</strong></a>
    <br />
    <br />
    <a href="https://github.com/MuteJester/LZGraphs/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/MuteJester/LZGraphs/issues">Request Feature</a>
  </p>
</p>

---

> **:dna: New to LZGraphs?** Head over to the **[full documentation and tutorials](https://MuteJester.github.io/LZGraphs/)** for comprehensive guides, API reference, and worked examples covering every feature of the library.

---

## Table of Contents

* [About the Project](#about-the-project)
* [Key Features](#key-features)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [Graph Types](#graph-types)
* [Sequence Abundance Weighting](#sequence-abundance-weighting)
* [Core Capabilities](#core-capabilities)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)


<!-- ABOUT THE PROJECT -->
## About The Project

LZGraphs :dna: is a Python library for immune receptor repertoire analysis based on the Lempel-Ziv 76 (LZ-76) compression algorithm. It builds directed graph models from TCR and BCR CDR3 sequences, capturing the sequential structure of repertoires without relying on alignment.

The methodology is presented in the research paper *"A Novel Approach to T-Cell Receptor Beta Chain (TCRB) Repertoire Encoding Using Lossless String Compression"*.

### Background

The diversity of T-cells and B-cells is crucial for producing receptors that recognize the wide range of pathogens encountered throughout life. V(D)J recombination generates this diversity through a stochastic process, making repertoire analysis challenging. LZGraphs addresses this by decomposing sequences into LZ-76 subpatterns and encoding them as graph transitions, providing a compact, information-rich representation of an entire repertoire.


## Key Features

- **Alignment-free analysis** -- no error-prone sequence alignment required
- **Generation probability inference** -- compute P(sequence) under the learned graph model
- **Sequence simulation** -- generate realistic synthetic sequences from the graph
- **Diversity estimation** -- LZ-based diversity indices (K-diversity family)
- **Information-theoretic metrics** -- entropy, perplexity, Jensen-Shannon divergence, mutual information, and more
- **Repertoire comparison** -- compare two repertoires via graph-level statistics
- **Analytical probability distributions** -- exact moments and scipy-like distribution objects for generation probabilities
- **Gene annotation support** -- optional V/J gene tracking on edges for gene usage analysis
- **Abundance weighting** -- weight sequences by clonal abundance for more realistic models
- **Serialization** -- save and load graphs in JSON format


## Installation

Install from PyPI:

```bash
pip install LZGraphs
```

LZGraphs requires Python 3.9 or later. To verify the installation:

```python
import LZGraphs
print(LZGraphs.__version__)
```


## Quick Start

Build an amino acid positional graph from CDR3 sequences and compute sequence probabilities:

```python
import pandas as pd
from LZGraphs import AAPLZGraph

# Prepare data as a DataFrame with a 'cdr3_amino_acid' column
data = pd.DataFrame({
    'cdr3_amino_acid': [
        'CASSLAPGATNEKLFF',
        'CASSLGQAYEQYF',
        'CASSFSTCSANYGYTF',
        'CASSQEGTEAFF',
        'CASSLGQGNIQYF',
        # ... your CDR3 amino acid sequences
    ]
})

# Construct the graph
graph = AAPLZGraph(data, verbose=True)

# Compute the log-probability of a sequence under the model
log_prob = graph.walk_log_probability('CASSLAPGATNEKLFF')
print(f"Log P(seq): {log_prob:.4f}")

# Simulate 100 new sequences from the graph
generated = graph.simulate(100, seed=42)
print(f"Generated {len(generated)} sequences")

# Access graph properties
print(f"Nodes: {graph.num_subpatterns}, Edges: {graph.num_transitions}")
print(f"Length distribution: {graph.length_probabilities}")
```


## Graph Types

LZGraphs provides three graph variants, each suited to different sequence types and analysis goals:

### AAPLZGraph -- Amino Acid Positional

Best for **CDR3 amino acid sequences**. Each LZ-76 subpattern is annotated with its position in the sequence, creating a directed acyclic graph (DAG). This enables exact analytical computations including `lzpgen_moments()` and `lzpgen_analytical_distribution()`.

```python
from LZGraphs import AAPLZGraph

graph = AAPLZGraph(data, verbose=True)  # data has 'cdr3_amino_acid' column
```

### NDPLZGraph -- Nucleotide Double Positional

Best for **CDR3 nucleotide sequences** where reading frame matters. Encodes both the subpattern and a double positional index derived from nucleotide positions. Also a DAG, supporting exact analytical methods.

```python
from LZGraphs import NDPLZGraph

graph = NDPLZGraph(data, verbose=True)  # data has 'cdr3_rearrangement' column
```

### NaiveLZGraph -- Basic Nucleotide

A simpler model for **nucleotide sequences** that uses raw LZ-76 subpatterns without positional annotation. The resulting graph may contain cycles. Use Monte Carlo methods (`lzpgen_distribution()`) rather than exact analytics for this graph type.

```python
from LZGraphs import NaiveLZGraph
from LZGraphs import generate_kmer_dictionary

cdr3_list = ['TGTGCCAGCAGC...', 'TGTGCCAGCAGT...', ...]
dictionary = generate_kmer_dictionary(cdr3_list)
graph = NaiveLZGraph(cdr3_list, dictionary, verbose=True)
```

### Gene Annotation

All three graph types support optional V and J gene annotation. Include `V` and `J` columns in your DataFrame (or pass them separately for NaiveLZGraph) to track gene usage on graph edges:

```python
data = pd.DataFrame({
    'cdr3_amino_acid': sequences,
    'V': v_genes,
    'J': j_genes,
})
graph = AAPLZGraph(data, verbose=True)

# Gene data is now available
print(graph.has_gene_data)           # True
print(graph.marginal_v_genes)        # V gene usage distribution
print(graph.marginal_j_genes)        # J gene usage distribution
```


## Sequence Abundance Weighting

Immune repertoire datasets often include clonal abundance information -- the number of times each unique clonotype was observed. LZGraphs supports abundance-weighted graph construction, where each sequence contributes proportionally to its observed count rather than being treated as a single observation.

This is particularly important for:

- **More accurate probability estimates** -- highly expanded clones exert greater influence on transition probabilities, reflecting the true distribution of the repertoire
- **Better representation of clonal expansion** -- dominant clones shape the graph structure proportionally to their prevalence
- **More realistic sequence generation** -- simulated sequences reflect the abundance-weighted landscape, not just the unique sequence set

To use abundance weighting, include an `abundance` column in your DataFrame:

```python
data = pd.DataFrame({
    'cdr3_amino_acid': ['CASSLAPGATNEKLFF', 'CASSLGQAYEQYF', 'CASSFSTCSANYGYTF'],
    'abundance': [150, 42, 7],
})

# Each sequence is weighted by its abundance during graph construction
graph = AAPLZGraph(data, verbose=True)
```

For `NaiveLZGraph`, pass abundances as a separate parameter:

```python
graph = NaiveLZGraph(
    cdr3_list,
    dictionary,
    verbose=True,
    abundances=[150, 42, 7, ...],
)
```

When no abundance information is provided, every sequence is implicitly weighted as 1.


## Core Capabilities

### Probability Inference

Compute the probability of a sequence under the learned Markov model:

```python
prob = graph.walk_probability('CASSLAPGATNEKLFF')
log_prob = graph.walk_log_probability('CASSLAPGATNEKLFF')
```

### Sequence Simulation

Generate new sequences by sampling random walks through the graph:

```python
sequences = graph.simulate(1000, seed=42)
```

### Generation Probability Distributions

Characterize the distribution of generation probabilities across the repertoire:

```python
# Monte Carlo empirical distribution (works on all graph types)
log_probs = graph.lzpgen_distribution(n=10000, seed=42)

# Exact analytical moments (DAG graphs only: AAPLZGraph, NDPLZGraph)
moments = graph.lzpgen_moments()
print(moments['mean'], moments['std'])

# Full scipy-like distribution object (DAG graphs only)
dist = graph.lzpgen_analytical_distribution()
print(dist.mean(), dist.std())
x = dist.ppf(0.05)  # 5th percentile
```

### Diversity Metrics

```python
from LZGraphs import lz_centrality, k_diversity

centrality = lz_centrality(graph, 'CASSLAPGATNEKLFF')
diversity = k_diversity(sequences, graph.encode_sequence, sample_size=1000)
```

### Information-Theoretic Analysis

```python
from LZGraphs import (
    node_entropy, edge_entropy, graph_entropy,
    jensen_shannon_divergence, compare_repertoires,
)

print(f"Graph entropy: {graph_entropy(graph):.4f}")

# Compare two repertoires
jsd = jensen_shannon_divergence(graph1, graph2)
comparison = compare_repertoires(graph1, graph2)
```

### Visualization

```python
from LZGraphs import plot_graph, plot_possible_paths

plot_graph(graph)
plot_possible_paths(graph, 'CASSLAPGATNEKLFF')
```

### Saturation Analysis

```python
from LZGraphs import NodeEdgeSaturationProbe

probe = NodeEdgeSaturationProbe()
# Feed sequences incrementally and track node/edge saturation curves
```

For detailed usage of every feature, see the **[documentation](https://MuteJester.github.io/LZGraphs/)**.


## Contributing

Contributions are what make the open-source community such a powerful place to create new ideas, inspire, and make progress. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the MIT license. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

[Thomas Konstantinovsky]() - thomaskon90@gmail.com

Project Link: [https://github.com/MuteJester/LZGraphs](https://github.com/MuteJester/LZGraphs)


<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/MuteJester/LZGraphs.svg?style=flat-square
[stars-url]: https://github.com/MuteJester/LZGraphs/stargazers
[issues-shield]: https://img.shields.io/github/issues/MuteJester/LZGraphs.svg?style=flat-square
[issues-url]: https://github.com/MuteJester/LZGraphs/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/thomas-konstantinovsky-56230117b/
