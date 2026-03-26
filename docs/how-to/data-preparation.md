# Prepare Your Data

Learn how to format and clean your TCR/BCR repertoire data for use with LZGraphs.

## Quick Reference

```python
from LZGraphs import LZGraph

# 1. Plain list of sequences (simplest)
sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]
graph = LZGraph(sequences, variant='aap')

# 2. With abundance weighting
counts = [150, 42, 10, ...]
graph = LZGraph(sequences, abundances=counts, variant='aap')

# 3. With V/J gene annotations
v_genes = ["TRBV16-1*01", "TRBV1-1*01", ...]
j_genes = ["TRBJ1-2*01", "TRBJ1-5*01", ...]
graph = LZGraph(sequences, v_genes=v_genes, j_genes=j_genes, variant='aap')
```

---

## Accepted Input Format

The `LZGraph` constructor expects **parallel lists** for all input data.

| Argument | Type | Description |
|----------|------|-------------|
| `sequences` | `list[str]` | **Required**: Amino acid or nucleotide strings |
| `abundances` | `list[int]` | Optional: Frequency counts per sequence |
| `v_genes` | `list[str]` | Optional: V gene names |
| `j_genes` | `list[str]` | Optional: J gene names |

!!! tip "No pandas required"
    LZGraphs has no pandas dependency. All inputs are plain Python lists. Use Python's built-in `csv` module to load data from files.

---

## Loading from Common Formats

### From CSV

```python
import csv
from LZGraphs import LZGraph

seqs, v_genes, j_genes = [], [], []
with open("repertoire.csv") as f:
    for row in csv.DictReader(f):
        seqs.append(row['cdr3_amino_acid'])
        v_genes.append(row['v_call'])
        j_genes.append(row['j_call'])

graph = LZGraph(seqs, v_genes=v_genes, j_genes=j_genes, variant='aap')
```

### From TSV (AIRR-Standard Files)

If your data is in the [AIRR TSV format](https://docs.airr-community.org/):

```python
import csv
from LZGraphs import LZGraph

seqs, v_genes, j_genes = [], [], []
with open("repertoire.tsv") as f:
    for row in csv.DictReader(f, delimiter='\t'):
        seqs.append(row['junction_aa'])
        v_genes.append(row['v_call'])
        j_genes.append(row['j_call'])

graph = LZGraph(seqs, v_genes=v_genes, j_genes=j_genes, variant='aap')
```

---

## Cleaning and Filtering

### Invalid Characters
LZGraphs expects valid amino acid (standard 20 residues) or nucleotide (A, C, G, T) strings. Filter out entries with missing data or non-standard characters like `*` or `X`:

```python
import re

# Standard 20 amino acids
valid_aa = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')

# Keep only valid sequences
clean_seqs = [s for s in sequences if valid_aa.match(s)]
```

### Length Filtering
Extremely short or long sequences are often artifacts of sequencing or pipeline errors:

```python
# Keep typical CDR3 lengths (8 to 25 residues)
filtered = [s for s in sequences if 8 <= len(s) <= 25]
```

---

## Saturation Analysis

How do you know if you have enough data to build a representative graph? Use the `saturation_curve` function to see how the number of nodes and edges grows as you add more sequences.

```python
from LZGraphs import saturation_curve

# Compute saturation stats
# Returns a list of dicts: [{'n_sequences': 100, 'n_nodes': 450, 'n_edges': 600}, ...]
stats = saturation_curve(sequences, variant='aap', log_every=500)

for s in stats[:5]:
    print(f"Seqs: {s['n_sequences']:>6} | Nodes: {s['n_nodes']:>6}")
```

If the curve of `n_nodes` starts to flatten, it means you have enough data to capture the structural diversity of that repertoire.

---

## Next Steps

- [Graph Construction tutorial](../tutorials/graph-construction.md) — Detailed construction options
- [Choosing Graph Variants](../concepts/graph-types.md) — Choose between AAP, NDP, and Naive
- [Quick Start](../getting-started/quickstart.md) — Build your first graph
