---
tags:
  - FlashBackGraph
description: Complete API reference for the FlashBackGraph class, a strictly Markovian DAG with exact, closed-form diversity, scoring, and anomaly metrics.
search:
  boost: 2
---

# FlashBackGraph

A strictly Markovian graph built from the **FlashBack decomposition**, where nodes are FlashBack tokens and edges are transitions between consecutive tokens. Because the graph is a Markovian DAG, every analytic (path count, diversity, entropy, Hill numbers, PGEN) is computed **exactly** via forward dynamic programming, with no Monte Carlo approximation.

For the conceptual distinction between `FlashBackGraph` and the [`LZGraph`](lzgraph.md) family (`aap` / `ndp` / `naive`), see [Graph Variants](../concepts/graph-types.md).

## Constructor

```python
FlashBackGraph(
    sequences,
    *,
    abundances=None,
    smoothing=0.0
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Sequences to build from (must be non-empty) |
| `abundances` | `list[int]` | Optional per-sequence counts (default: all 1) |
| `smoothing` | `float` | Laplace smoothing alpha for edge weights (default: 0.0) |

### from_file (classmethod)

```python
FlashBackGraph.from_file(path, *, smoothing=0.0)
```
Build directly from a plain-text file with constant memory (streaming). Accepts one sequence per line, or `sequence<TAB>abundance`.

## Core Methods

### pgen

```python
pgen(sequence, *, log=True)
```
Exact generation probability of one or more sequences under the FlashBack model.

- **Parameters**: `sequence` (str or list[str]), `log` (bool)
- **Returns**: `float` (single) or `np.ndarray` (list)

### simulate

```python
simulate(n, *, seed=None)
```
Generate `n` sequences by Markov random walk.

- **Returns**: [SimulationResult](simulation-result.md)

### top_k_sequences

```python
top_k_sequences(k=100, *, most_probable=True)
```
Find the K most (or least) probable sequences exactly, via forward DP over the DAG's topological order, with no simulation. Set `most_probable=False` for the lowest-probability tail.

- **Returns**: [SimulationResult](simulation-result.md)

### posterior

```python
posterior(sequences, *, abundances=None, kappa=1.0)
```
Bayesian posterior graph using this graph as a Dirichlet-Multinomial prior. Keeps the prior topology and reweights edges toward the new individual. `kappa=0` → pure individual, `kappa→∞` → pure prior. Edges absent from the prior are ignored.

- **Returns**: A new `FlashBackGraph` instance.

### without

```python
without(sequences, *, abundances=None)
```
Return a new graph with the contribution of `sequences` removed: each walk's edge counts are decremented (clamped at zero, zero-count edges pruned, weights renormalised). Enables leave-donor-out construction from an existing graph in seconds rather than rebuilding from source.

- **Returns**: A new `FlashBackGraph` instance.

## Exact Diversity & Analytics

All values below are computed exactly by forward DP over the DAG.

### effective_diversity

```python
effective_diversity()
```
Exact \(e^H\) (Shannon entropy), equivalent to Hill number \(D(1)\).

### diversity_profile

```python
diversity_profile()
```
Full Shannon diversity breakdown (cached per instance).

### hill_number

```python
hill_number(alpha)
```
Exact Hill diversity number \(D(\alpha)\).

### hill_numbers

```python
hill_numbers(orders)
```
Exact Hill numbers for multiple orders. Returns an `np.ndarray`.

### hill_curve

```python
hill_curve(orders=None)
```
Returns a dict with `orders` and `values` for plotting a diversity profile (default orders span 0–10).

### power_sum

```python
power_sum(alpha)
```
Exact power sum \(M(\alpha)\).

### pgen_diagnostics

```python
pgen_diagnostics(atol=1e-6)
```
Exact absorbed vs leaked probability mass.

### pgen_dynamic_range

```python
pgen_dynamic_range()
```
Exact dynamic range of generation probabilities, in orders of magnitude. Use `pgen_dynamic_range_detail()` for the full breakdown.

### pgen_moments

```python
pgen_moments()
```
Moments (`mean`, `variance`, `std`, `skewness`, `kurtosis`) of the forward-DP log-PGEN distribution.

### pgen_distribution

```python
pgen_distribution()
```
Legacy Gaussian-mixture approximation of the log-PGEN distribution. The
global moments are propagated analytically, but the current per-component fit
uses sampled walks. For a sampling-free analysis, use `pseq_analysis()`.

### pseq_analysis

```python
analysis = graph.pseq_analysis()
```

Create a sampling-free analytical p-sequence spectrum for the FlashBack DAG.
The exact core is the Mellin transform

\[
M(q)=\sum_sP(s)^q,
\]

which connects directly to Hill diversity. The returned
`FlashBackPseqAnalysis` supports:

```python
analysis.mellin(q)                 # exact M(q)
analysis.log_mellin(q)             # stable exact log M(q)
analysis.derivatives(q, order=4)   # exact transform derivatives
analysis.tilted_moments(q, 4)      # normalized moments under P(sequence)**q
analysis.attribution(q)            # exact tilted node/edge usage and sensitivity
analysis.diversity_under_edge_thresholds(thresholds)  # conditioned D0/D1/D2
analysis.moments()                 # exact surprisal moments
analysis.cumulants()               # exact surprisal cumulants
graph.path_count_by_length()       # generated richness by literal AA length
analysis.length_marginals()        # fused richness/probability by AA length
analysis.length_derivatives(q, 4)  # transform derivatives by AA length
analysis.length_profile()          # exact moments by sequence length
analysis.exact_atoms()             # exact enumeration for small supports
analysis.histogram()               # deterministic large-support grid
analysis.histogram_pair()          # fused global counting/generated grids
analysis.histograms_by_length()    # joint length-conditioned grids
analysis.saddlepoint().pdf_cdf(x)  # batched smooth PDF/CDF approximation
analysis.position(sequence)         # individual repertoire position
analysis.expected_richness(n)       # finite-depth occupancy prediction
analysis.discovery_curve(draws)     # batched richness and novelty curves
analysis.publicness_distribution(depths)  # predicted repertoire occupancy
```

`exact_atoms()` is necessarily support-limited because an explicit PMF can
contain exponentially many atoms. `histogram()` is deterministic and reports
its grid spacing and rounding-error bound; it has no Monte Carlo variance.
When both global measures are needed, `histogram_pair()` returns a dictionary
containing `counting` and `generated` histograms after one native graph
traversal. Exact-length reconstruction remains available through
`histogram(length=...)`.
When several lengths are needed for one measure,
`histograms_by_length(max_length=...)` transports the length and grid states
together and returns one `PseqHistogram` per reachable amino-acid length.

Here, sequence length is the number of amino-acid characters in the sequence
reconstructed by a path, excluding the `@` and `$` sentinels and all token
metadata. It is not the number of nodes or edges in the walk. See
[FlashBack P-Sequence Analytics](../concepts/flashback-pseq-analytics.md) for
the transform equations, stable tilted moments, structural attribution,
edge-pruning curves, saddlepoint inversion, length-conditioned interpretation,
histogram measures, fused global reconstruction, batched discovery/novelty
curves, and guidance on choosing
an interface.

### publicness_distribution

```python
result = analysis.publicness_distribution(depths, levels=None)
```

Predicted number of sequences at each publicness level, where publicness is
repertoire occupancy: in how many of a cohort's repertoires a sequence is
present. `depths` gives one sampling depth per repertoire, in distinct
sequences contributed. A sequence of probability \(p\) is present in a
repertoire of depth \(N\) with probability \(1-(1-p)^N\); the depths differ, so
the occupancy count is Poisson-binomial rather than binomial, and its PMF comes
from the generating function

\[
G(z)=\prod_b\left(1-\pi_b+\pi_b z\right)^{m_b}
\]

sampled at the \(K\)-th roots of unity and inverted by a forward DFT.

This is not the same quantity as `expected_frequency_spectrum(n, max_count)`,
which counts how many times a sequence appears inside *one* pool of \(n\)
draws. Publicness counts how many *separate* repertoires contain it at all.

Returns a dict with `edges` (half-open integer bin edges), `expected_counts`,
`expected_total`, the per-atom `atom_mean` and `atom_variance`, the diagnostic
`min_retained_mass`, and the cohort description. `levels` selects the binning:
`None` for one bin per occupancy level, an `int` for that many equal-width
bins, or an explicit sequence of integer edges.

The inversion's roundoff, amplified by spectrum multiplicities that reach
\(10^{29}\), would otherwise rectify into a spurious floor in every publicness
bin. Each atom is truncated to the support its closed-form moments place the
mass in before negatives are clipped, which is what removes that floor. The
`tail_sigma` and `tail_floor` arguments control that window; the reasoning is
in the `LZGraphs._publicness` module docstring and should be read before
changing them.

## Anomaly Scoring (SCALE)

**SCALE** is the recommended anomaly/error score: a self-calibrated,
length-invariant transform of `-log pgen`. You calibrate once against the
graph, then score sequences (higher = more anomalous). See the
[anomaly-detection tutorial](../tutorials/flashback-anomaly.md).

### calibrate_scale

```python
calibrate_scale(*, n_sim=200_000, seed=None, min_count=50)
```
Self-calibrate SCALE: simulate `n_sim` sequences from the graph and record the
per-length median and IQR of `-log pgen`. Returns a `ScaleCalibration` (the reusable cache, documented below).

- **Parameters**: `n_sim` (int), `seed` (int or None), `min_count` (int, minimum simulated sequences at a length to get its own median/IQR; sparser lengths fall back to global).
- **Returns**: `ScaleCalibration`.

### scale_score

```python
scale_score(sequence, calibration)
```
Length-calibrated anomaly score: `(-log pgen(s) - median[len]) / IQR[len]`,
using a `ScaleCalibration`. Higher means more anomalous.

- **Parameters**: `sequence` (str or list[str]), `calibration` (`ScaleCalibration`).
- **Returns**: `float` (single) or `np.ndarray` (list).

### ScaleCalibration

The calibration cache returned by `calibrate_scale`. Holds `median_by_length`,
`iqr_by_length`, `global_median`, `global_iqr`, `n_sim`, and `seed`. Persist and
reuse it:

```python
calibration.save('scale_calibration.json')
from LZGraphs import ScaleCalibration
calibration = ScaleCalibration.load('scale_calibration.json')
```

## Graph Algebra

| Operation | Method | Result |
|-----------|--------|--------|
| Union | `a.union(b)` | Sum edge counts |
| Intersection | `a.intersection(b)` | Shared structure, min counts |
| Difference | `a.difference(b)` | Subtract edge counts |
| Weighted Merge | `a.weighted_merge(b, α, β)` | Linear combination \( \alpha A + \beta B \) |

All operations return a new `FlashBackGraph`.

## Features & Adjacency

### feature_stats

```python
feature_stats()
```
15-element statistical vector describing the graph, for ML pipelines.

### feature_mass_profile

```python
feature_mass_profile(max_pos=30)
```
Position-based mass distribution profile.

### adjacency_csr

```python
adjacency_csr()
```
CSR (Compressed Sparse Row) adjacency as a dict of numpy arrays (`row_offsets`, `col_indices`, `weights`, `counts`).

## IO

### save

```python
save(path)
```
Save to `.lzg` binary format.

### load (classmethod)

```python
FlashBackGraph.load(path)
```
Load from a `.lzg` binary file.

## Attributes

### Basic
| Attribute | Type | Description |
|-----------|------|-------------|
| `n_nodes` | `int` | Total number of nodes (including @ and $ sentinels) |
| `n_edges` | `int` | Total number of directed edges |
| `n_sequences` | `int` | Number of input sequences (abundance-weighted) |
| `variant` | `str` | Always `'flashback'` |
| `is_dag` | `bool` | Always `True` |
| `path_count` | `int` | Exact number of distinct walks, in arbitrary precision |

### Structure
| Attribute | Type | Description |
|-----------|------|-------------|
| `nodes` | `list[str]` | Node labels (excluding sentinels) |
| `all_nodes` | `list[str]` | All node labels (including @ and $) |
| `edges` | `list[tuple]` | `(src, dst, weight, count)` tuples (no sentinels) |
| `all_edges` | `list[tuple]` | All `(src, dst, weight, count)` tuples |
| `n_initial` | `int` | Number of initial states |
| `n_terminal` | `int` | Number of terminal nodes |
| `max_out_degree` | `int` | Maximum out-degree |
| `max_in_degree` | `int` | Maximum in-degree |
| `density` | `float` | Graph density (0 to 1) |
| `out_degrees` | `np.ndarray` | Out-degree of each node |
| `in_degrees` | `np.ndarray` | In-degree of each node |
| `length_distribution` | `dict` | `{length: count}` mapping |

## Streaming Builder: FlashBackStream

For sequences that arrive incrementally (a generator, a network stream, an open-ended simulator), `FlashBackStream` accumulates them and lets you monitor running node/edge counts before deciding to stop. The finalized graph is byte-identical to `FlashBackGraph(all_sequences)`.

```python
from LZGraphs import FlashBackStream

with FlashBackStream(smoothing=0.0) as stream:
    for batch in source:
        stream.add_sequences(batch)            # appends; cheap
        print(stream.n_nodes, stream.n_edges)  # peek; instant
        if some_stop_condition:
            break
    graph = stream.finalize()                  # one-time CSR build
    graph.save('foundation.lzg')
```

| Member | Description |
|--------|-------------|
| `FlashBackStream(smoothing=0.0)` | Open a new streaming builder. |
| `add_sequences(sequences, abundances=None)` | Append sequences; raises `RuntimeError` after finalize/abort. |
| `n_nodes` / `n_edges` | Current running counts (instant peek). |
| `peek()` | `{'n_nodes': int, 'n_edges': int}` of the running accumulator. |
| `snapshot()` | Build a finalized graph from the current state **without** consuming the stream (for checkpointing). |
| `finalize()` | Convert the accumulator into a `FlashBackGraph`; consumes the stream. |
| `abort()` | Discard the partial build without paying the finalize cost (idempotent). |

Used as a context manager, the stream auto-aborts on exit if it was never finalized.
