# Performance Benchmarks

All benchmarks were run on a single core (Intel/AMD x86_64 Linux, Python 3.12, LZGraphs 3.0.1) using a dataset of 5,000 CDR3 amino acid sequences (mean length 14.7 characters). The resulting graph has 1,721 nodes and 9,644 edges.

Times are wall-clock averages across multiple runs. Your results will vary depending on hardware, dataset size, and sequence diversity.

---

## Graph Construction

Building a graph from raw sequences — includes LZ76 decomposition, CSR packing, edge weight normalization, and topological sort.

| Sequences | Nodes | Edges | Time |
|----------:|------:|------:|-----:|
| 100 | 342 | 654 | **0.5 ms** |
| 500 | 753 | 2,269 | **6 ms** |
| 1,000 | 997 | 3,585 | **10 ms** |
| 2,000 | 1,262 | 5,529 | **24 ms** |
| 5,000 | 1,721 | 9,644 | **82 ms** |

Construction scales roughly linearly with the number of input sequences. A graph from 5,000 sequences builds in under 100 ms.

---

## LZPGEN Scoring

Computing the exact log-generation-probability of sequences against a graph (5,000-sequence graph).

| Sequences scored | Time | Throughput |
|-----------------:|-----:|-----------:|
| 1 | 0.2 ms | ~5,000 /sec |
| 10 | 2 ms | ~5,000 /sec |
| 100 | 22 ms | ~4,600 /sec |
| 1,000 | 205 ms | ~4,900 /sec |
| 5,000 | 1.0 s | ~4,800 /sec |

Throughput is constant at ~5,000 sequences/second regardless of batch size. Each score requires tracing the full LZ76 walk through the graph with per-step dictionary constraint checking.

---

## Simulation

Generating new sequences via LZ-constrained random walks (5,000-sequence graph).

| Sequences | Time | Throughput |
|----------:|-----:|-----------:|
| 100 | 21 ms | ~4,800 /sec |
| 1,000 | 208 ms | ~4,800 /sec |
| 10,000 | 2.1 s | ~4,800 /sec |
| 100,000 | 20.6 s | ~4,800 /sec |

Simulation throughput is constant at ~4,800 sequences/second. Each walk maintains a per-walk LZ dictionary and includes backtracking for dead-end recovery.

---

## Analytics

Diversity and distribution metrics on the 5,000-sequence graph.

| Operation | Time | Notes |
|:----------|-----:|:------|
| `effective_diversity()` | **2.1 s** | Monte Carlo with 10K walks |
| `hill_numbers([0,1,2,5])` | **2.1 s** | Single MC run, all orders computed together |
| `pgen_moments()` | **0.2 ms** | Forward DP — no simulation needed |
| `pgen_distribution()` | **2.1 s** | Simulation-based Gaussian mixture fitting |
| `predicted_richness(1M)` | **106 s** | Occupancy model with series acceleration |
| `pgen_diagnostics()` | **~200 ms** | 1K validation walks |
| `diversity_profile()` | **2.1 s** | Same MC as effective_diversity |

!!! info "Why are Hill numbers fast but richness is slow?"
    **Hill numbers** use Monte Carlo with 10K walks — each walk provides its exact probability, enabling unbiased importance-sampling estimation in about 2 seconds.

    **Predicted richness** uses the Poisson occupancy formula $F(d) = \sum_i (1 - (1-p_i)^d)$, which requires evaluating a series over the full PGEN distribution with Gauss-Hermite quadrature and Wynn epsilon acceleration. At very large depths (1M+), the series converges slowly. For moderate depths (up to ~10K), it's much faster.

---

## Comparison

| Operation | Time |
|:----------|-----:|
| `jensen_shannon_divergence(g1, g2)` | **0.1 ms** |

JSD is computed directly from the edge weight vectors — no simulation needed.

---

## Graph Operations

Set-algebraic operations on two graphs (~1,200 nodes each).

| Operation | Time |
|:----------|-----:|
| `union` | **102 ms** |
| `intersection` | **91 ms** |
| `difference` | **69 ms** |

These operations rebuild a new CSR graph from the merged edge data, including re-normalization and topological sort.

---

## Serialization (IO)

Save/load the 5,000-sequence graph (285 KB on disk).

| Operation | Time |
|:----------|-----:|
| `save()` | **1.2 ms** |
| `load()` | **0.7 ms** |

The `.lzg` binary format is extremely fast — loading a graph is ~100x faster than rebuilding it from sequences.

---

## Feature Extraction

| Operation | Dimension | Time |
|:----------|----------:|-----:|
| `feature_aligned(query)` | 1,721 | **0.4 ms** |
| `feature_stats()` | 15 | **8.4 s** |
| `feature_mass_profile()` | 31 | **2.1 s** |

!!! note "`feature_stats()` is slow because it computes Hill numbers internally"
    The 15-element statistics vector includes D(0), D(0.5), D(1), D(2), D(5), entropy, and dynamic range — each requiring Monte Carlo simulation. If you only need the `feature_aligned()` vector for ML, that's sub-millisecond.

---

## Scaling characteristics

| Operation | Complexity | Scales with |
|:----------|:-----------|:------------|
| Graph construction | $O(n \cdot L)$ | n = sequences, L = mean length |
| LZPGEN scoring | $O(L)$ per sequence | L = sequence length |
| Simulation | $O(L)$ per sequence | L = generated length |
| Hill numbers (MC) | $O(N \cdot L)$ | N = MC samples (10K default) |
| JSD | $O(\|E\|)$ | E = edges in larger graph |
| Save/Load | $O(\|V\| + \|E\|)$ | V = nodes, E = edges |
| `feature_aligned()` | $O(\|V\|)$ | V = reference nodes |

---

## Tips for performance

1. **Build once, reuse.** Graph construction is the most expensive step per-sequence. Save to `.lzg` and load for repeated analysis.

2. **Batch LZPGEN calls.** `graph.lzpgen(list_of_seqs)` is no faster per-sequence than a loop, but avoids Python overhead.

3. **Use `pgen_moments()` for quick summaries.** It runs in microseconds via forward DP, while `pgen_distribution()` takes seconds.

4. **`feature_aligned()` is the fastest ML feature.** Sub-millisecond per sample, versus seconds for `feature_stats()`.

5. **JSD is nearly free.** At 0.1 ms, you can compute thousands of pairwise comparisons in seconds.
