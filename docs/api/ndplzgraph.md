# NDPLZGraph

Nucleotide Reading Frame Positional LZGraph for analyzing nucleotide CDR3 sequences.

## Quick Example

```python
from LZGraphs import NDPLZGraph

# Build from a plain list of nucleotide sequences
sequences = ["TGTGCCAGCAGTTTCAAGAT", "TGTGCCAGCAGCCAAAGCAG", ...]
graph = NDPLZGraph(sequences, verbose=True)

# Calculate probability (accepts raw strings)
pgen = graph.walk_probability("TGTGCCAGCAGT")
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
        - simulate
        - encode_sequence
        - extract_subpattern
        - save
        - load
        - eigenvector_centrality
        - graph_summary
      heading_level: 3

## Constructor

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `list[str]`, `pd.Series`, or `pd.DataFrame` | Nucleotide CDR3 sequences |
| `abundances` | `list[int]` | Per-sequence abundance counts *(list/Series input only)* |
| `v_genes` | `list[str]` | V gene annotations *(list/Series input only)* |
| `j_genes` | `list[str]` | J gene annotations *(list/Series input only)* |
| `verbose` | `bool` | Print progress messages (default: `True`) |

When `data` is a **list** or **Series**, use the keyword arguments above to attach metadata.
When `data` is a **DataFrame**, metadata columns should be in the DataFrame itself (`cdr3_rearrangement`, and optionally `V`, `J`, `abundance`).

## Node Format

NDPLZGraph uses reading frame and position encoding:

```
<pattern><reading_frame>_<position>
```

Where `reading_frame` is 0, 1, or 2 (codon position) and `position` is the ending position.

Example:
```python
encoded = NDPLZGraph.encode_sequence("TGTGCC")
# ['T0_1', 'G1_2', 'TG2_4', 'C1_5', 'C2_6']
```

## Key Methods

### walk_probability

Accepts a raw sequence string or a pre-encoded walk list.

```python
pgen = graph.walk_probability("TGTGCCAGCAGT", use_log=True)
print(f"log P(gen) = {pgen:.2f}")
```

### encode_sequence (static)

```python
encoded = NDPLZGraph.encode_sequence("TGTGCC")
# Returns: ['T0_1', 'G1_2', 'TG2_4', 'C1_5', 'C2_6']
```

### extract_subpattern (static)

```python
pattern = NDPLZGraph.extract_subpattern("TG2_4")
# Returns: "TG"
```

### simulate

```python
sequences = graph.simulate(1000, seed=42)
```

## Comparison with AAPLZGraph

| Feature | NDPLZGraph | AAPLZGraph |
|---------|------------|------------|
| Sequence type | Nucleotides | Amino acids |
| Position encoding | Reading frame + position | Single (end position) |
| Alphabet size | 4 | 20 |
| Graph size | Larger | Smaller |
| Resolution | Higher | Lower |

## See Also

- [AAPLZGraph](aaplzgraph.md) - Amino acid version
- [NaiveLZGraph](naivelzgraph.md) - Non-positional version
- [Concepts: Graph Types](../concepts/graph-types.md)
