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
    K1000_Diversity,
    K_Diversity,
    LZCentrality,
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
    draw_graph,
    ancestors_descendants_curves_plot,
    sequence_possible_paths_plot,
    sequence_genomic_node_variability_plot,
    sequence_genomic_edges_variability_plot
)
```

## Class Hierarchy

```
LZGraphBase (abstract)
├── AAPLZGraph - Amino acid positional
├── NDPLZGraph - Nucleotide double positional
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
| `walk_probability(walk)` | Calculate sequence probability |
| `random_walk()` | Generate a random sequence |
| `save(filepath)` | Save graph to disk |
| `load(filepath)` | Load graph from disk |
| `encode_sequence(seq)` | Encode sequence to walk |
| `clean_node(node)` | Extract pattern from node |

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
│   └── naive.py
├── metrics/
│   ├── diversity.py
│   └── entropy.py
├── utilities/
│   ├── utilities.py
│   └── node_edge_saturation_probe.py
├── bag_of_words/
│   └── bow_encoder.py
├── visualization/
│   └── visualize.py
└── exceptions/
    └── __init__.py
```
