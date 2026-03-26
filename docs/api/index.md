# API Reference

Complete reference documentation for all LZGraphs classes and functions.

## Quick Navigation

<div class="grid" markdown>

<div class="card" markdown>
### Core
- [LZGraph](lzgraph.md) — Main graph class (all variants)
- [SimulationResult](simulation-result.md) — Output of `simulate()`
- [PgenDistribution](pgen-distribution.md) — Analytical PGEN distribution
</div>

<div class="card" markdown>
### Analysis & Utilities
- [Module Functions](functions.md) — `jensen_shannon_divergence`, `k_diversity`, etc.
- [CLI Tool](cli.md) — `lzg` command reference
- [Exceptions](exceptions.md) — Error handling
</div>

</div>

## Import Patterns

### Classes

```python
from LZGraphs import LZGraph, SimulationResult, PgenDistribution
```

### Functions

```python
from LZGraphs import (
    jensen_shannon_divergence,
    k_diversity,
    saturation_curve,
    lz76_decompose,
    set_log_level,
)
```

### Exceptions

```python
from LZGraphs import LZGraphError, NoGeneDataError, ConvergenceError, CorruptFileError
```

## The `LZGraph` Class

There is a single graph class with a `variant` parameter:

```python
from LZGraphs import LZGraph

# Amino acid positional (most common)
graph = LZGraph(sequences, variant='aap')

# Nucleotide double positional
graph = LZGraph(sequences, variant='ndp')

# Naive (no positional encoding)
graph = LZGraph(sequences, variant='naive')
```

All variants share the same methods. See the [LZGraph reference](lzgraph.md) for the full API.

## Quick Method Reference

| Category | Method | Description |
|----------|--------|-------------|
| **Scoring** | `lzpgen(seq)` | Log-probability of sequence(s) |
| **Simulation** | `simulate(n)` | Generate n sequences |
| **Diversity** | `effective_diversity()` | exp(Shannon entropy) = D(1) |
| | `hill_number(alpha)` | Hill diversity number D(α) |
| | `hill_numbers(orders)` | Multiple Hill numbers |
| | `hill_curve()` | Full diversity curve |
| | `diversity_profile()` | Entropy, diversity, uniformity |
| **Occupancy** | `predicted_richness(depth)` | Expected unique seqs at depth |
| | `predicted_overlap(d_i, d_j)` | Expected shared sequences |
| | `richness_curve(depths)` | Richness at multiple depths |
| | `predict_sharing(draws)` | Sharing spectrum across donors |
| **Distribution** | `pgen_moments()` | Mean/var of log-PGEN |
| | `pgen_distribution()` | Analytical Gaussian mixture |
| | `pgen_diagnostics()` | Check proper distribution |
| | `pgen_dynamic_range()` | Dynamic range in orders of mag |
| **Perplexity** | `sequence_perplexity(seq)` | Single sequence perplexity |
| | `repertoire_perplexity(seqs)` | Average repertoire perplexity |
| | `path_entropy_rate(seqs)` | Entropy rate (bits/token) |
| **Graph Ops** | `union(other)` / `a \| b` | Sum edge counts |
| | `intersection(other)` / `a & b` | Shared edges, min counts |
| | `difference(other)` / `a - b` | Subtract edge counts |
| | `weighted_merge(other, α, β)` | Linear combination |
| | `posterior(seqs)` | Bayesian posterior update |
| **Features** | `feature_aligned(query)` | Aligned feature vector |
| | `feature_mass_profile()` | Position-based mass profile |
| | `feature_stats()` | 15-element stats vector |
| **IO** | `save(path)` | Save to `.lzg` binary format |
| | `LZGraph.load(path)` | Load from `.lzg` file |
| **Info** | `summary()` | Structural summary dict |
| | `n_nodes`, `n_edges` | Graph size |
| | `variant`, `is_dag` | Graph properties |
| | `has_gene_data` | Whether gene data is available |

## Version Information

```python
import LZGraphs
print(LZGraphs.__version__)  # 3.0.0
```
