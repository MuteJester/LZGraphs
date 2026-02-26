# API Reference

Complete reference documentation for all LZGraphs classes and functions.

## Quick Navigation

<div class="grid" markdown>

<div class="card" markdown>
### Graph Classes
- [AAPLZGraph](aaplzgraph.md) - Amino acid graphs
- [NDPLZGraph](ndplzgraph.md) - Nucleotide graphs
- [NaiveLZGraph](naivelzgraph.md) - Non-positional graphs
</div>

<div class="card" markdown>
### Analysis
- [Metrics](metrics.md) - Diversity and entropy
- [Utilities](utilities.md) - Helper functions
- [Exceptions](exceptions.md) - Error handling
</div>

</div>

## Import Patterns

### Core Classes

```python
from LZGraphs import AAPLZGraph, NDPLZGraph, NaiveLZGraph
```

### Metrics Functions

```python
from LZGraphs import (
    k1000_diversity,
    k_diversity,
    lz_centrality,
    node_entropy,
    edge_entropy,
    graph_entropy,
    jensen_shannon_divergence
)
```

### Utilities

```python
from LZGraphs import LZBOW, NodeEdgeSaturationProbe
from LZGraphs.utilities import lempel_ziv_decomposition, generate_kmer_dictionary
```

### Visualization

```python
from LZGraphs.visualization import (
    plot_graph,
    plot_ancestor_descendant_curves,
    plot_possible_paths,
    plot_gene_node_variability,
    plot_gene_edge_variability
)
```

## Class Hierarchy

```
LZGraphBase (abstract)
├── AAPLZGraph - Amino acid positional
├── NDPLZGraph - Nucleotide reading frame positional
└── NaiveLZGraph - No positional encoding

LZBOW - Bag of Words encoder
NodeEdgeSaturationProbe - Saturation analysis

Exceptions:
LZGraphError (base)
├── InputValidationError
│   ├── EmptyDataError
│   ├── MissingColumnError
│   └── InvalidSequenceError
├── GraphConstructionError
├── GeneDataError
│   ├── NoGeneDataError
│   └── GeneAnnotationError
├── WalkError
│   ├── NoValidPathError
│   └── MissingNodeError
└── ...
```

## Common Methods

All graph classes share these methods:

| Method | Description |
|--------|-------------|
| `walk_probability(walk)` | Calculate sequence probability (accepts raw string or encoded list) |
| `random_walk()` | Generate a random sequence |
| `simulate(n)` | Batch-generate sequences with pre-computed cache |
| `save(filepath)` | Save graph to disk |
| `load(filepath)` | Load graph from disk |
| `encode_sequence(seq)` | Encode sequence to walk (static) |
| `extract_subpattern(node)` | Extract pattern from node (static) |
| `get_posterior(seqs, kappa)` | Bayesian posterior personalization |
| `graph_summary()` | Summary statistics (nodes, edges, degree) |

## Version Information

```python
import LZGraphs
print(LZGraphs.__version__)
```

## Module Structure

```
LZGraphs/
├── __init__.py           # Main exports
├── graphs/
│   ├── amino_acid_positional.py
│   ├── nucleotide_double_positional.py
│   ├── naive.py
│   ├── lz_graph_base.py
│   └── graph_operations.py
├── metrics/
│   ├── diversity.py
│   ├── entropy.py
│   ├── saturation.py
│   ├── convenience.py
│   └── pgen_distribution.py
├── constants.py             # Shared numerical constants
├── mixins/
│   ├── gene_logic.py
│   ├── gene_prediction.py
│   ├── random_walk.py
│   ├── graph_topology.py
│   ├── lzpgen_distribution.py
│   ├── walk_analysis.py
│   ├── bayesian_posterior.py
│   └── serialization.py
├── utilities/
│   ├── decomposition.py
│   ├── helpers.py
│   └── misc.py
├── bag_of_words/
│   └── bow_encoder.py
├── visualization/
│   └── visualize.py
└── exceptions/
    └── __init__.py
```
