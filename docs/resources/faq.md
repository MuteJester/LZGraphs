# Frequently Asked Questions

Common questions about using LZGraphs.

## General

### Which graph type should I use?

**Short answer:**

- **Amino acid sequences** → `AAPLZGraph`
- **Nucleotide sequences** → `NDPLZGraph`
- **ML feature extraction** → `NaiveLZGraph`

**Detailed guide:** See [Concepts: Graph Types](../concepts/graph-types.md)

### How much data do I need?

LZGraphs works with any dataset size, but:

- **Minimum:** ~100 sequences for basic analysis
- **Recommended:** 1,000+ sequences for reliable diversity metrics
- **K1000 requirement:** At least 1,000 unique sequences

For small datasets, consider `k100_diversity` instead of `k1000_diversity`.

### Can I use LZGraphs for non-TCR sequences?

Yes! LZGraphs works with any string sequences. The library is optimized for TCR/CDR3 analysis, but the core algorithms are sequence-agnostic.

```python
# Works with any strings
from LZGraphs import NaiveLZGraph
from LZGraphs.utilities import generate_kmer_dictionary

dictionary = generate_kmer_dictionary(6)
graph = NaiveLZGraph(my_custom_sequences, dictionary)
```

---

## Probability and Analysis

### Why is my sequence probability zero?

A probability of zero (or a `MissingNodeError`/`MissingEdgeError`) means the sequence contains patterns or transitions not observed in the training data:

```python
# Debug: check which nodes/edges are missing
encoded = AAPLZGraph.encode_sequence(sequence)
for node in encoded:
    if not graph.graph.has_node(node):
        print(f"Missing node: {node}")

for i in range(len(encoded) - 1):
    if not graph.graph.has_edge(encoded[i], encoded[i+1]):
        print(f"Missing edge: {encoded[i]} -> {encoded[i+1]}")
```

!!! tip
    `walk_probability` accepts raw strings directly — no need to call `encode_sequence` yourself:
    ```python
    pgen = graph.walk_probability("CASSLEPSGGTDTQYF", use_log=True)
    ```

### How do I interpret K1000?

K1000 measures the number of unique LZ76 patterns in a sample of 1,000 sequences:

- **Higher values** = more diverse repertoire
- **Lower values** = more repetitive patterns
- **Typical range** = 500-3000 depending on repertoire

### What's the difference between perplexity and probability?

- **Probability (Pgen)**: How likely is this exact sequence?
- **Perplexity**: How "surprised" is the model by this sequence?

Lower perplexity = sequence fits the model better.

---

## Technical Issues

### "ModuleNotFoundError: No module named 'LZGraphs'"

Ensure LZGraphs is installed:

```bash
pip install LZGraphs
```

Or check you're in the correct Python environment.

### "MissingColumnError: Required column 'cdr3_amino_acid' not found"

Your DataFrame needs the correct column names:

- `cdr3_amino_acid` for `AAPLZGraph`
- `cdr3_rearrangement` for `NDPLZGraph`

```python
# Check your columns
print(data.columns.tolist())

# Rename if needed
data = data.rename(columns={'CDR3': 'cdr3_amino_acid'})
```

### "NoGeneDataError: This operation requires gene annotation data"

Some functions require V/J gene columns:

```python
# Build with gene data
data = pd.DataFrame({
    'cdr3_amino_acid': sequences,
    'V': v_genes,  # Required for genomic functions
    'J': j_genes
})
graph = AAPLZGraph(data)
```

### Memory issues with large repertoires

For very large datasets:

1. **Subsample first:**
   ```python
   data_sample = data.sample(n=50000)
   graph = AAPLZGraph(data_sample)
   ```

2. **Use NaiveLZGraph:** Smaller graphs with fixed dictionary

3. **Save and reload:**
   ```python
   graph.save("large_graph.pkl")
   # Load when needed
   graph = AAPLZGraph.load("large_graph.pkl")
   ```

---

## Performance

### How can I speed up graph construction?

- Use `verbose=False` to skip progress output
- Subsample large datasets for exploration
- Build once and save for repeated use

### How long should K1000 take?

With 30 draws on 10,000 sequences: ~10-30 seconds

Increase `draws` for more accurate results (slower).

---

## Best Practices

### Should I normalize sequence lengths?

No, LZGraphs handles variable-length sequences naturally. The length distribution is captured in `graph.lengths`.

### How do I compare repertoires of different sizes?

Use normalized metrics:

- `normalized_graph_entropy()` - Entropy normalized by graph size
- `jensen_shannon_divergence()` - Inherently normalized (0 to 1)
- K-diversity with same sample size

### How do I handle special characters?

Remove or replace them before building the graph:

```python
# Remove non-standard amino acids
data = data[data['cdr3_amino_acid'].str.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$')]
```

---

## Still Have Questions?

- [Open an issue](https://github.com/MuteJester/LZGraphs/issues) on GitHub
- Email: [thomaskon90@gmail.com](mailto:thomaskon90@gmail.com)
- Check the [API Reference](../api/index.md) for detailed documentation
