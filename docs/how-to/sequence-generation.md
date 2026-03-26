---
tags:
  - Simulation
  - Genes
---

# Generate Sequences

This guide covers the practical tasks involved in generating synthetic sequences
from an LZGraph and working with the results.

---

## Quick reference

```python
from LZGraphs import LZGraph

graph = LZGraph(sequences, variant="aap")

results = graph.simulate(1000)                # basic generation
results = graph.simulate(1000, seed=42)       # reproducible
results = graph.simulate(1000,                # gene-constrained
                         sample_genes=True)
results = graph.simulate(100,                 # specific V/J pair
                         v_gene="TRBV7-2*01",
                         j_gene="TRBJ2-1*01")

# SimulationResult is iterable, indexable, and sliceable
subset = results[:5]  # returns a SimulationResult with aligned metadata
for i, seq in enumerate(subset):
    print(seq, subset.log_probs[i])
```

---

## Generate a batch of sequences

Call `simulate(n)` to produce `n` sequences in one shot. The returned
[`SimulationResult`](../api/simulation-result.md) holds sequences together with
their generation metadata.

```python
from LZGraphs import LZGraph

sequences = [
    "CASSLEPSGGTDTQYF",
    "CASSDTSGGTDTQYF",
    "CASSLEPQTFTDTFFF",
]
graph = LZGraph(sequences, variant="aap")

results = graph.simulate(100)

print(f"Generated {len(results)} sequences")
print(f"First sequence: {results[0]}")
```

!!! tip
    `simulate` uses a compiled C extension by default for maximum throughput.
    Generating one million sequences typically completes in under a second.

---

## Reproduce results with a seed

Pass the `seed` parameter to make generation deterministic. The same seed
always yields the same sequences, regardless of platform.

```python
run_a = graph.simulate(50, seed=42)
run_b = graph.simulate(50, seed=42)

assert list(run_a) == list(run_b)  # identical output
```

!!! note
    Seeds are local to each `simulate` call and do not affect any other
    random state in your program.

---

## Generate with gene constraints

These options require a graph built with V/J gene data (i.e., the
`v_genes` and `j_genes` lists were provided at construction time).
Check `graph.has_gene_data` to verify.

### Sample V/J pairs from the observed distribution

Set `sample_genes=True` to draw a (V, J) pair per sequence from the joint
gene distribution recorded during graph construction. Each sequence is then
generated through the sub-graph conditioned on that pair.

```python
results = graph.simulate(1000, sample_genes=True)

for i in range(3):
    print(
        f"{results[i]}  "
        f"V={results.v_genes[i]}  "
        f"J={results.j_genes[i]}"
    )
```

### Fix a specific V/J combination

To generate sequences for one particular gene pair, pass the gene names
directly.

```python
results = graph.simulate(100,
                         v_gene="TRBV7-2*01",
                         j_gene="TRBJ2-1*01")
```

You can also constrain only one gene and leave the other free:

```python
# All sequences will use TRBV7-2*01; J gene varies
results = graph.simulate(100, v_gene="TRBV7-2*01")
```

!!! warning
    If the requested gene does not exist in the graph, a `ValueError` is
    raised. Check available genes beforehand with `graph.v_genes` and
    `graph.j_genes`.

---

## Work with SimulationResult

`simulate` returns a [`SimulationResult`](../api/simulation-result.md) object.
It supports iteration, indexing, slicing, and `len`.

### Iterate over sequences

```python
for seq in results:
    process(seq)
```

### Index and slice

```python
first = results[0]           # single sequence (str)
batch = results[10:20]       # slice returns a new SimulationResult
```

### Access structured fields

| Field | Type | Always present | Description |
|---|---|---|---|
| `results.sequences` | `list[str]` | Yes | Generated amino-acid sequences |
| `results.log_probs` | `numpy.ndarray` (float64) | Yes | Exact log-probability of each sequence under the model |
| `results.n_tokens` | `numpy.ndarray` (uint32) | Yes | Number of LZ tokens per sequence |
| `results.v_genes` | `list[str]` or `None` | Only with gene constraints | V gene assigned to each sequence |
| `results.j_genes` | `list[str]` or `None` | Only with gene constraints | J gene assigned to each sequence |

```python
import numpy as np

results = graph.simulate(500, seed=7)

# Highest-probability sequence
best_idx = np.argmax(results.log_probs)
print(results[best_idx], results.log_probs[best_idx])

# Mean token count
print(f"Mean LZ tokens: {results.n_tokens.mean():.1f}")
```

---

## Score sequences with lzpgen

Use `graph.lzpgen(seq)` to compute the log-probability of any sequence --
generated or observed -- under the graph's transition model. This is useful for
ranking, filtering, or statistical testing.

```python
score = graph.lzpgen("CASSLEPSGGTDTQYF")
print(f"Log-probability: {score:.4f}")
```

You can score an entire batch by iterating:

```python
import numpy as np

observed = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF"]
scores = graph.lzpgen(observed)  # pass a list for batch scoring → returns np.ndarray
```

!!! info
    `log_probs` on a `SimulationResult` are computed during generation at
    zero extra cost. Use `lzpgen` only when you need to score sequences that
    were *not* produced by `simulate`.

---

## Filter generated sequences by length

`SimulationResult` fields are aligned by index, so NumPy boolean indexing works
naturally.

```python
import numpy as np

results = graph.simulate(5000, seed=0)
lengths = np.array([len(s) for s in results.sequences])

# Keep only sequences between 13 and 17 residues
mask = (lengths >= 13) & (lengths <= 17)
filtered_seqs = [s for s, keep in zip(results.sequences, mask) if keep]
filtered_lps  = results.log_probs[mask]

print(f"Kept {len(filtered_seqs)} / {len(results)} sequences")
```

---

## Data augmentation

Generate a large synthetic repertoire from a small observed sample to
supplement downstream machine-learning pipelines.

```python
from LZGraphs import LZGraph

# Build from a small clinical sample
graph = LZGraph(small_sample, variant="aap")

# Augment to 50,000 sequences
synthetic = graph.simulate(50_000, seed=123)

# Write to a plain-text file (one sequence per line)
with open("synthetic_repertoire.txt", "w") as fh:
    for seq in synthetic:
        fh.write(seq + "\n")
```

!!! tip "Preserving gene annotations"
    If gene labels matter for your downstream task, use `sample_genes=True`
    and write the gene columns alongside the sequences:

    ```python
    synthetic = graph.simulate(50_000, seed=123, sample_genes=True)

    import csv
    with open("synthetic_repertoire.tsv", "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["sequence", "v_gene", "j_gene", "log_prob"])
        for i, seq in enumerate(synthetic):
            writer.writerow([
                seq,
                synthetic.v_genes[i],
                synthetic.j_genes[i],
                f"{synthetic.log_probs[i]:.6f}",
            ])
    ```

---

## Null model generation

Build a null distribution of generation probabilities and use it to assess
whether a particular sequence is statistically unusual given the repertoire
structure.

```python
import numpy as np
from LZGraphs import LZGraph

graph = LZGraph(repertoire_sequences, variant="aap")

# 1. Generate null sequences and collect their log-probabilities
null = graph.simulate(10_000, seed=0)
null_lps = null.log_probs

# 2. Score the test sequence
test_seq = "CASSLEPSGGTDTQYF"
test_lp  = graph.lzpgen(test_seq)

# 3. Empirical p-value (fraction of null sequences at least as unlikely)
p_value = (null_lps <= test_lp).mean()
print(f"Test log-prob: {test_lp:.4f}  |  p-value: {p_value:.4f}")
```

!!! note
    A low p-value indicates the test sequence is less probable than most
    sequences the model generates -- it may represent a rare or atypical
    rearrangement.

---

## Troubleshooting

??? failure "\"No gene data\" error"
    Raised when you pass `v_gene`, `j_gene`, or `sample_genes=True` to a
    graph that was not built with gene annotations. Rebuild the graph with
    V/J columns or load from an AIRR file that includes them.

??? failure "\"Gene not found\" error"
    The gene name you requested does not appear in the graph. Inspect the
    available genes:

    ```python
    print(graph.v_genes)  # list of V gene names
    print(graph.j_genes)  # list of J gene names
    ```

---

## Next steps

- [SimulationResult API reference](../api/simulation-result.md) -- full field and method documentation
- [Distribution Analytics](../concepts/distribution-analytics.md) -- the mathematical model behind generation probabilities
- [Compare Repertoires](repertoire-comparison.md) -- compare synthetic and real distributions
- [Sequence Analysis tutorial](../tutorials/sequence-analysis.md) -- end-to-end walkthrough with real data
