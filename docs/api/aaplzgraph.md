# AAPLZGraph

Amino Acid Positional LZGraph for analyzing amino acid CDR3 sequences.

## Quick Example

```python
from LZGraphs import AAPLZGraph

# Build from a plain list of sequences
sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]
graph = AAPLZGraph(sequences, verbose=True)

# Calculate probability (accepts raw strings)
pgen = graph.walk_probability("CASSLEPSGGTDTQYF")
```

## Class Reference

::: LZGraphs.graphs.amino_acid_positional.AAPLZGraph
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
| `data` | `list[str]`, `pd.Series`, or `pd.DataFrame` | Amino acid CDR3 sequences |
| `abundances` | `list[int]` | Per-sequence abundance counts *(list/Series input only)* |
| `v_genes` | `list[str]` | V gene annotations *(list/Series input only)* |
| `j_genes` | `list[str]` | J gene annotations *(list/Series input only)* |
| `verbose` | `bool` | Print progress messages (default: `True`) |

When `data` is a **list** or **Series**, use the keyword arguments above to attach metadata.
When `data` is a **DataFrame**, metadata columns should be in the DataFrame itself (`cdr3_amino_acid`, and optionally `V`, `J`, `abundance`).

!!! tip "Simplest usage"
    ```python
    graph = AAPLZGraph(["CASSLE...", "CASSD...", ...])
    ```

## Key Methods

### walk_probability

Calculate the generation probability of a sequence. Accepts a raw sequence string or a pre-encoded walk list.

```python
# Raw string (recommended)
pgen = graph.walk_probability("CASSLEPSGGTDTQYF")
print(f"P(gen) = {pgen:.2e}")

# Use log probability for numerical stability
log_pgen = graph.walk_probability("CASSLEPSGGTDTQYF", use_log=True)
print(f"log P(gen) = {log_pgen:.2f}")
```

### random_walk

Generate a random sequence following edge probabilities.

```python
walk = graph.random_walk()
sequence = ''.join([AAPLZGraph.extract_subpattern(n) for n in walk])
print(sequence)
```

### genomic_random_walk

Generate a sequence consistent with V/J gene usage.

```python
walk, v_gene, j_gene = graph.genomic_random_walk()
sequence = ''.join([AAPLZGraph.extract_subpattern(n) for n in walk])
print(f"{sequence} ({v_gene}, {j_gene})")
```

### simulate

Batch-generate sequences using a pre-computed walk cache for maximum throughput.

```python
# Generate 1000 sequences
sequences = graph.simulate(1000)

# Reproducible generation
sequences = graph.simulate(1000, seed=42)

# Get walks and sequences
walks_and_seqs = graph.simulate(100, return_walks=True)
for walk, seq in walks_and_seqs[:3]:
    print(f"{seq} (walk: {len(walk)} steps)")
```

### encode_sequence (static)

Convert a sequence to graph walk format.

```python
encoded = AAPLZGraph.encode_sequence("CASSLE")
# Returns: ['C_1', 'A_2', 'S_3', 'SL_5', 'E_6']
```

### extract_subpattern (static)

Extract the pattern from a node name.

```python
pattern = AAPLZGraph.extract_subpattern("SL_5")
# Returns: "SL"
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `graph` | `nx.DiGraph` | NetworkX directed graph |
| `nodes` | `NodeView` | All nodes in the graph |
| `edges` | `EdgeView` | All edges in the graph |
| `lengths` | `dict` | Sequence length distribution |
| `initial_state_counts` | `pd.Series` | Initial state counts |
| `terminal_state_counts` | `pd.Series` | Terminal state counts |
| `marginal_v_genes` | `pd.Series` | V gene probabilities |
| `marginal_j_genes` | `pd.Series` | J gene probabilities |
| `node_probability` | `dict` | Pattern probabilities (node → float) |
| `has_gene_data` | `bool` | Whether V/J gene data was provided |
| `num_subpatterns` | `int` | Total unique nodes |
| `num_transitions` | `int` | Total transition count |

## Examples

### Building with Gene Annotation

```python
sequences = ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF']
v = ['TRBV16-1*01', 'TRBV1-1*01']
j = ['TRBJ1-2*01', 'TRBJ1-5*01']

graph = AAPLZGraph(sequences, v_genes=v, j_genes=j, verbose=True)
print(graph.marginal_v_genes)
```

### Building with Abundance Weighting

```python
sequences  = ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF']
abundances = [150, 42]

graph = AAPLZGraph(sequences, abundances=abundances, verbose=True)
```

### Batch Probability Calculation

```python
sequences = ['CASSLEPSGGTDTQYF', 'CASSLGQGSTEAFF', 'CASSXYZRARESEQ']

for seq in sequences:
    try:
        log_p = graph.walk_probability(seq, use_log=True)
        print(f"{seq}: {log_p:.2f}")
    except:
        print(f"{seq}: Not in graph")
```

## See Also

- [NDPLZGraph](ndplzgraph.md) - Nucleotide version
- [NaiveLZGraph](naivelzgraph.md) - Non-positional version
- [Tutorials: Graph Construction](../tutorials/graph-construction.md)
