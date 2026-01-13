# AAPLZGraph

Amino Acid Positional LZGraph for analyzing amino acid CDR3 sequences.

## Quick Example

```python
from LZGraphs import AAPLZGraph
import pandas as pd

# Build graph
data = pd.read_csv("repertoire.csv")
graph = AAPLZGraph(data, verbose=True)

# Calculate probability
sequence = "CASSLEPSGGTDTQYF"
encoded = AAPLZGraph.encode_sequence(sequence)
pgen = graph.walk_probability(encoded)
```

## Class Reference

::: LZGraphs.Graphs.AminoAcidPositional.AAPLZGraph
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
| `data` | `pd.DataFrame` | DataFrame with `cdr3_amino_acid` column |
| `verbose` | `bool` | Print progress messages (default: `True`) |

### Required Columns

- `cdr3_amino_acid` - Amino acid CDR3 sequences

### Optional Columns

- `V` - V gene/allele annotations
- `J` - J gene/allele annotations

## Key Methods

### walk_probability

Calculate the generation probability of a sequence.

```python
encoded = AAPLZGraph.encode_sequence("CASSLEPSGGTDTQYF")
pgen = graph.walk_probability(encoded)
print(f"P(gen) = {pgen:.2e}")

# Use log probability for numerical stability
log_pgen = graph.walk_probability(encoded, use_log=True)
print(f"log P(gen) = {log_pgen:.2f}")
```

### random_walk

Generate a random sequence following edge probabilities.

```python
walk = graph.random_walk()
sequence = ''.join([AAPLZGraph.clean_node(n) for n in walk])
print(sequence)
```

### genomic_random_walk

Generate a sequence consistent with V/J gene usage.

```python
walk, v_gene, j_gene = graph.genomic_random_walk()
sequence = ''.join([AAPLZGraph.clean_node(n) for n in walk])
print(f"{sequence} ({v_gene}, {j_gene})")
```

### encode_sequence (static)

Convert a sequence to graph walk format.

```python
encoded = AAPLZGraph.encode_sequence("CASSLE")
# Returns: ['C_1', 'A_2', 'S_3', 'SL_5', 'E_6']
```

### clean_node (static)

Extract the pattern from a node name.

```python
pattern = AAPLZGraph.clean_node("SL_5")
# Returns: "SL"
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.DiGraph` | NetworkX directed graph |
| `nodes` | `NodeView` | All nodes in the graph |
| `edges` | `EdgeView` | All edges in the graph |
| `lengths` | `dict` | Sequence length distribution |
| `initial_states` | `pd.Series` | Initial state counts |
| `terminal_states` | `pd.Series` | Terminal state counts |
| `marginal_vgenes` | `pd.Series` | V gene probabilities |
| `marginal_jgenes` | `pd.Series` | J gene probabilities |
| `subpattern_individual_probability` | `pd.DataFrame` | Pattern probabilities |

## Examples

### Building with Gene Annotation

```python
data = pd.DataFrame({
    'cdr3_amino_acid': ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF'],
    'V': ['TRBV16-1*01', 'TRBV1-1*01'],
    'J': ['TRBJ1-2*01', 'TRBJ1-5*01']
})

graph = AAPLZGraph(data, verbose=True)
print(graph.marginal_vgenes)
```

### Batch Probability Calculation

```python
sequences = ['CASSLEPSGGTDTQYF', 'CASSLGQGSTEAFF', 'CASSXYZRARESEQ']

for seq in sequences:
    try:
        encoded = AAPLZGraph.encode_sequence(seq)
        log_p = graph.walk_probability(encoded, use_log=True)
        print(f"{seq}: {log_p:.2f}")
    except:
        print(f"{seq}: Not in graph")
```

## See Also

- [NDPLZGraph](ndplzgraph.md) - Nucleotide version
- [NaiveLZGraph](naivelzgraph.md) - Non-positional version
- [Tutorials: Graph Construction](../tutorials/graph-construction.md)
