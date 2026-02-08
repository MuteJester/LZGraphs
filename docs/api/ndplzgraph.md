# NDPLZGraph

Nucleotide Double Positional LZGraph for analyzing nucleotide CDR3 sequences.

## Quick Example

```python
from LZGraphs import NDPLZGraph
import pandas as pd

# Build graph
data = pd.read_csv("repertoire.csv")
graph = NDPLZGraph(data, verbose=True)

# Calculate probability
sequence = "TGTGCCAGCAGT"
encoded = NDPLZGraph.encode_sequence(sequence)
pgen = graph.walk_probability(encoded)
```

## Class Reference

::: LZGraphs.graphs.nucleotide_double_positional.NDPLZGraph
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - walk_probability
        - random_walk
        - genomic_random_walk
        - encode_sequence
        - clean_node
        - save
        - load
        - eigenvector_centrality
      heading_level: 3

## Constructor

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `pd.DataFrame` | DataFrame with `cdr3_rearrangement` column |
| `verbose` | `bool` | Print progress messages (default: `True`) |

### Required Columns

- `cdr3_rearrangement` - Nucleotide CDR3 sequences

### Optional Columns

- `V` - V gene/allele annotations
- `J` - J gene/allele annotations

## Node Format

NDPLZGraph uses double positional encoding:

```
<pattern>_<start>_<end>
```

Example:
```python
encoded = NDPLZGraph.encode_sequence("TGTGCC")
# ['T_1_1', 'G_2_2', 'T_3_3', 'G_4_4', 'C_5_5', 'C_6_6']
```

## Key Methods

### walk_probability

```python
encoded = NDPLZGraph.encode_sequence("TGTGCCAGCAGT")
pgen = graph.walk_probability(encoded, use_log=True)
print(f"log P(gen) = {pgen:.2f}")
```

### encode_sequence (static)

```python
encoded = NDPLZGraph.encode_sequence("TGTGCC")
# Returns: ['T_1_1', 'G_2_2', 'T_3_3', 'G_4_4', 'C_5_5', 'C_6_6']
```

### clean_node (static)

```python
pattern = NDPLZGraph.clean_node("TG_3_4")
# Returns: "TG"
```

## Comparison with AAPLZGraph

| Feature | NDPLZGraph | AAPLZGraph |
|---------|------------|------------|
| Sequence type | Nucleotides | Amino acids |
| Position encoding | Double (start, end) | Single (end) |
| Alphabet size | 4 | 20 |
| Graph size | Larger | Smaller |
| Resolution | Higher | Lower |

## See Also

- [AAPLZGraph](aaplzgraph.md) - Amino acid version
- [NaiveLZGraph](naivelzgraph.md) - Non-positional version
- [Concepts: Graph Types](../concepts/graph-types.md)
