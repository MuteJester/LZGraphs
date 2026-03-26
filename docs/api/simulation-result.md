# SimulationResult

Container returned by [`LZGraph.simulate()`](lzgraph.md#simulate). Holds the generated sequences along with their exact log-probabilities and token counts, and optionally V/J gene annotations.

## Quick Example

```python
result = graph.simulate(1000, seed=42)

# Iterate as strings
for seq in result:
    print(seq)

# Access metadata
print(result.log_probs[:5])   # exact log P(gen) per sequence
print(result.n_tokens[:5])    # LZ76 token count per sequence
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Generated CDR3 strings |
| `log_probs` | `np.ndarray[float64]` | Exact log-probability of each sequence under the LZ-constrained model |
| `n_tokens` | `np.ndarray[uint32]` | Number of LZ76 subpattern tokens in each sequence's walk |
| `v_genes` | `list[str]` or `None` | V gene used for each sequence (only with `sample_genes=True` or explicit `v_gene`) |
| `j_genes` | `list[str]` or `None` | J gene used for each sequence (only with gene-constrained simulation) |

!!! info "Exact probabilities"
    Unlike many generative models, each simulated sequence carries its **exact** generation probability — the precise product of all transition probabilities along the walk. This is not an approximation, and it enables unbiased importance-sampling estimators for diversity and entropy.

## Sequence Protocol

`SimulationResult` implements Python's sequence protocol, so you can use it like a list of strings:

```python
result = graph.simulate(100, seed=42)

# Length
print(len(result))          # 100

# Indexing — returns the sequence string
first = result[0]           # 'CASSLEPSGGTDTQYF'

# Slicing — returns a new SimulationResult with aligned metadata
subset = result[:10]
print(len(subset))          # 10
print(subset.log_probs)     # first 10 log-probs

# Iteration — yields strings
for seq in result:
    print(seq)

# Membership
print('CASSLGIRRT' in result.sequences)
```

## Working with Metadata

The `log_probs` and `n_tokens` arrays are aligned with `sequences` — index `i` in each corresponds to the same generated sequence:

```python
import numpy as np

result = graph.simulate(10000, seed=42)

# Find the most probable generated sequence
best_idx = np.argmax(result.log_probs)
print(f"Most probable: {result[best_idx]}")
print(f"  log P = {result.log_probs[best_idx]:.4f}")
print(f"  tokens = {result.n_tokens[best_idx]}")

# Summary statistics
print(f"Mean log P: {result.log_probs.mean():.2f}")
print(f"Std log P:  {result.log_probs.std():.2f}")
print(f"Mean length: {np.mean([len(s) for s in result.sequences]):.1f}")
print(f"Mean tokens: {result.n_tokens.mean():.1f}")
```

## Gene-Annotated Results

When simulation includes gene data (`sample_genes=True` or explicit `v_gene`/`j_gene`), the result includes gene annotations:

```python
result = graph.simulate(100, sample_genes=True, seed=42)

for i in range(3):
    print(f"{result[i]:25s}  V={result.v_genes[i]}  J={result.j_genes[i]}")
```

If the simulation was not gene-constrained, `v_genes` and `j_genes` are `None`:

```python
result = graph.simulate(100, seed=42)
print(result.v_genes)  # None
```

## Filtering Results

Since all arrays are aligned, you can use NumPy boolean indexing to filter:

```python
import numpy as np

result = graph.simulate(10000, seed=42)

# Keep only sequences with > median probability
median_lp = np.median(result.log_probs)
mask = result.log_probs > median_lp

high_prob_seqs = [s for s, m in zip(result.sequences, mask) if m]
high_prob_lps = result.log_probs[mask]
print(f"Kept {len(high_prob_seqs)} sequences above median log P")

# Keep only sequences of specific length
lengths = np.array([len(s) for s in result.sequences])
mask_15 = lengths == 15
seqs_15 = [s for s, m in zip(result.sequences, mask_15) if m]
print(f"{len(seqs_15)} sequences of length 15")
```

## See Also

- [`LZGraph.simulate()`](lzgraph.md#simulate) — how to generate results
- [Sequence Analysis tutorial](../tutorials/sequence-analysis.md) — working with simulation output
- [Generate Sequences how-to](../how-to/sequence-generation.md) — recipes for filtering, augmentation, and null models
