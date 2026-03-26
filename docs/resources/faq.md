# Frequently Asked Questions

Common questions about using LZGraphs.

## General

### Which graph variant should I use?

All variants are handled by the unified `LZGraph` class:

- **Amino acid sequences** → `variant='aap'` (most common)
- **Nucleotide sequences** → `variant='ndp'`
- **Position-free motif discovery** → `variant='naive'`

See [Concepts: Graph Variants](../concepts/graph-types.md) for a detailed comparison.

### How much data do I need?

LZGraphs works with any dataset size, but:

- **Minimum:** ~100 sequences for basic exploration.
- **Recommended:** 10,000+ sequences for stable probability estimates.
- **Diversity analysis:** At least 1,000 unique sequences are recommended for reliable `k_diversity` results.

### Can I use LZGraphs for non-TCR sequences?

Yes! While optimized for TCR/CDR3 analysis, the core LZ76 algorithms are sequence-agnostic. You can use the `naive` variant for any string sequences (e.g., proteins, DNA, or even natural language).

```python
from LZGraphs import LZGraph
graph = LZGraph(my_custom_sequences, variant='naive')
```

---

## Probability and Analysis

### Why is my sequence probability zero?

A probability of zero means the sequence contains patterns or transitions never observed in the training data. Because LZGraphs 3.0+ strictly enforces LZ76 dictionary constraints, if a transition doesn't exist in the graph, the walk is impossible.

!!! tip
    `lzpgen` accepts raw strings directly:
    ```python
    log_p = graph.lzpgen("CASSLEPSGGTDTQYF")
    ```

### How do I interpret Hill Numbers?

Hill numbers provide a diversity profile:
- **D(0)**: Total number of unique sequences the graph can generate (Richness).
- **D(1)**: Effective diversity (based on Shannon entropy).
- **D(2)**: Collision diversity (reciprocal of the probability that two random sequences are identical).

### What's the difference between Pgen and Perplexity?

- **Generation Probability (Pgen)**: How likely is this exact sequence to be produced by the model?
- **Perplexity**: How "surprised" is the model by this sequence? (Lower is better).

---

## Technical Issues

### "ModuleNotFoundError: No module named 'LZGraphs'"

Ensure LZGraphs is installed in your current environment:
```bash
pip install LZGraphs
```

### "Missing column error"

The `LZGraph` constructor in version 3.0+ expects **plain lists** of strings, not DataFrames. If you are using pandas, extract the columns first:

```python
# Instead of LZGraph(df), use:
graph = LZGraph(df['cdr3_amino_acid'].tolist(), variant='aap')
```

### "NoGeneDataError"

This occurs when you call gene-aware methods (like `sample_genes=True` in `simulate()`) on a graph that was built without gene annotations. To fix this, provide the `v_genes` and `j_genes` lists during construction.

---

## Performance

### Is LZGraphs fast?

Yes. Version 3.0+ features a high-performance C backend. Constructing a graph from 1 million sequences typically takes only a few seconds on a modern laptop. Simulation and probability scoring are equally fast, handling millions of operations per second.

### Memory usage

The C core is extremely memory-efficient. Even graphs representing millions of sequences typically fit within 100-500 MB of RAM.

---

## Best Practices

### Should I normalize sequence lengths?

No. LZGraphs handles variable-length sequences naturally. The positional encoding in AAP and NDP variants ensures that motifs are analyzed in their correct structural context.

### How do I compare repertoires of different sizes?

Use **Jensen-Shannon Divergence (JSD)**. It is inherently normalized between 0 and 1 and is robust to differences in sample size. For diversity, use `k_diversity` with a fixed `k` (e.g., 1000) for all samples.

### How do I handle special characters?

Remove non-standard characters (like `*`, `X`, or spaces) before building the graph to ensure the LZ76 decomposition is clean.

---

## Reproducibility

### How do I get identical results every time?

**Graph construction** is deterministic — the same input always produces the same graph.

**Simulation** supports seeding:
```python
results = graph.simulate(1000, seed=42)
```

**Save and load** using the `.lzg` format to ensure you are working with the exact same model across sessions.

---

## Limitations

### What can't LZGraphs do?

- **Mechanistic Modeling**: LZGraphs is a statistical model of *observed* sequences. It does not model the mechanistic process of V(D)J recombination (insertions/deletions).
- **Paired Chains**: It currently models single chains (e.g., TRB) and does not support alpha-beta pairing.
- **Zero-shot Generalization**: It cannot assign probability to motifs it has never seen.

---

## Still Have Questions?

- [Open an issue](https://github.com/MuteJester/LZGraphs/issues) on GitHub
- Email: [thomaskon90@gmail.com](mailto:thomaskon90@gmail.com)
- Check the [API Reference](../api/index.md)
