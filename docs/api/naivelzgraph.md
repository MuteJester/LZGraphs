# NaiveLZGraph

Non-positional LZGraph for consistent feature extraction and cross-repertoire analysis.

## Quick Example

```python
from LZGraphs import NaiveLZGraph
from LZGraphs.utilities import generate_kmer_dictionary

# Create shared dictionary
dictionary = generate_kmer_dictionary(6)

# Build graph
sequences = ['TGTGCCAGCAGT', 'TGTGCCAGCAGC']
graph = NaiveLZGraph(sequences, dictionary, verbose=True)

# Extract features
features = graph.eigenvector_centrality()
```

## Class Reference

::: LZGraphs.graphs.naive.NaiveLZGraph
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - walk_probability
        - eigenvector_centrality
        - save
        - load
        - clean_node
      heading_level: 3

## Constructor

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | List of sequences |
| `dictionary` | `list[str]` | List of allowed patterns |
| `verbose` | `bool` | Print progress (default: `True`) |

## Key Differences

Unlike AAPLZGraph and NDPLZGraph:

- **No positional encoding** - Nodes are just patterns
- **Fixed dictionary** - Consistent nodes across repertoires
- **No gene support** - No V/J annotation

## Primary Use Cases

### Machine Learning Features

```python
from LZGraphs import NaiveLZGraph
from LZGraphs.utilities import generate_kmer_dictionary

# Shared dictionary for all repertoires
dictionary = generate_kmer_dictionary(6)

# Build graphs for multiple repertoires
graphs = []
for sequences in repertoire_list:
    g = NaiveLZGraph(sequences, dictionary, verbose=False)
    graphs.append(g)

# Extract feature vectors (same dimensions!)
features = [g.eigenvector_centrality() for g in graphs]
```

### Cross-Repertoire Comparison

```python
# Same dictionary ensures comparable graphs
g1 = NaiveLZGraph(seqs1, dictionary)
g2 = NaiveLZGraph(seqs2, dictionary)

# Features are directly comparable
f1 = g1.eigenvector_centrality()
f2 = g2.eigenvector_centrality()
```

## Dictionary Generation

```python
from LZGraphs.utilities import generate_kmer_dictionary

# All patterns up to length k
dict_6 = generate_kmer_dictionary(6)  # 5460 patterns
dict_5 = generate_kmer_dictionary(5)  # 1364 patterns
dict_4 = generate_kmer_dictionary(4)  # 340 patterns

print(f"Length 6: {len(dict_6)} patterns")
```

## See Also

- [AAPLZGraph](aaplzgraph.md) - Positional amino acid version
- [NDPLZGraph](ndplzgraph.md) - Positional nucleotide version
- [Concepts: Graph Types](../concepts/graph-types.md)
