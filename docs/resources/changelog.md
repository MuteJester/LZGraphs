# Changelog

All notable changes to LZGraphs are documented here.

This project follows [Semantic Versioning](https://semver.org/).

---

## [3.0.2] - 2026

### Fixed
- Restored scalable public `simulate()` and `lzpgen()` semantics on large graphs while preserving sequence/log-prob consistency.
- Improved probability diagnostics, classical Hill-number estimation, and zero-probability repertoire perplexity handling.
- Added a standalone C benchmark harness for graph loading, simulation, scoring, analytics, and I/O throughput measurements.

### Changed
- Refactored the C core into smaller internal modules across simulation, graph finalization/build ingest, analytics, PGEN distribution, I/O, graph operations, occupancy, and diversity.
- Clarified public documentation around constrained simulation, approximate accepted-walk normalization, and the unconstrained forward-DP `pgen_dist` approximation.

---

## [3.0.1] - 2026

### Fixed
- Improved large-file graph construction with streaming plain-text ingestion, safer capacity handling, and `uint64` count support.
- Fixed plain `sequence<TAB>count` parsing and added stricter input-validation pathways in the CLI and Python APIs.
- Improved save metadata to record the correct library version in `.lzg` files.

### Changed
- Significantly accelerated Foundation-graph query paths for `lzpgen()` and `simulate()` without changing graph or traversal semantics.
- Improved long-running build logging with clearer progress, phase reporting, and operational safety checks.

---

## [3.0.0] - 2026

**Major Rewrite: High-Performance C-Core**

This version is a complete re-implementation of the LZGraphs engine in C, providing 100x-1000x faster construction and analysis while significantly reducing memory overhead.

### Added
- **C-Core Engine**: Core graph operations, LZ76 decomposition, and generative modeling now run in a high-performance C backend.
- **Unified `LZGraph` Class**: Replaced `AAPLZGraph`, `NDPLZGraph`, and `NaiveLZGraph` with a single `LZGraph` class using a `variant` parameter (`'aap'`, `'ndp'`, `'naive'`).
- **Binary Format (`.lzg`)**: New custom binary format for saving/loading graphs that is faster and more compact than `pickle`.
- **LZ-Constrained Model**: Simulation (`simulate()`) and probability scoring (`lzpgen()`) now strictly enforce LZ76 dictionary constraints at every step.
- **Analytical Moments**: Exact computation of log-PGEN mean, variance, skewness, and kurtosis via topological forward propagation (O(V+E)).
- **Occupancy Predictions**: Advanced `predicted_richness` and `predicted_overlap` algorithms using splitting + Taylor series + Wynn epsilon acceleration for machine-precision results at any depth.
- **Feature Alignment**: New `feature_aligned()` method to project any repertoire into the node space of a reference graph for consistent ML features.
- **Logging System**: New `set_log_level()` and `set_log_callback()` for controlling C-core diagnostic output.

### Changed
- **Dependencies**: Removed `networkx`, `scipy`, `tqdm`, and `matplotlib` from core dependencies. `numpy` is now the only required dependency.
- **Python API**: Simplified and modernized API:
    - `walk_probability()` → `lzpgen()`
    - `random_walk()` / `genomic_random_walk()` → `simulate()`
    - `get_posterior()` → `posterior()`
    - `k1000_diversity()` → `k_diversity()`
- **Attributes**: Renamed for consistency: `lengths` → `length_distribution`, `marginal_v_genes` → `v_marginals`, `marginal_j_genes` → `j_marginals`.

### Removed
- **Visualization Module**: The `LZGraphs.visualization` module has been removed to eliminate heavy dependencies.
- **LZBOW Vectorizer**: Replaced by the more robust `feature_aligned()` projection.
- **Legacy Metrics**: `lz_centrality`, `node_entropy`, `edge_entropy`, and `transition_jsd` have been removed in favor of more principled analytical moments and JSD.

---

## [2.5.0] - 2026

### Added
- **Distribution analytics** — characterization of the generative probability distribution.
- `simulation_potential_size()`: count of unique producible sequences.
- `pgen_diagnostics()`: mass conservation check.
- `effective_diversity()`: Shannon entropy and N_eff.
- `predict_sharing_spectrum(draw_counts)`: sharing spectrum via analytical quadrature.

---

## [2.2.0] - 2026

### Added
- **Bayesian posterior personalization** via `get_posterior()`.
- Dirichlet-Multinomial conjugacy for updating population priors.

### Changed
- **`pandas` is no longer a required dependency**.
- Internal attributes moved from `pd.Series` to plain `dict`.

---

## Migration Guide: Upgrading to 3.0.0

Version 3.0.0 is a **breaking change** from the 2.x series.

1. **Replace class names**: Change `AAPLZGraph(...)` to `LZGraph(..., variant='aap')`.
2. **Update method calls**:
    - `graph.walk_probability(seq)` → `graph.lzpgen(seq)`
    - `graph.random_walk()` → `graph.simulate(1)`
    - `graph.get_posterior(...)` → `graph.posterior(...)`
3. **Change file extensions**: Update saved graphs from `.pkl` or `.json` to `.lzg` by rebuilding them with the new version.
4. **Remove visualization**: If you relied on `LZGraphs.visualization`, you will now need to use `matplotlib` or `seaborn` directly with the data returned by `graph.nodes`, `graph.edges`, or `graph.hill_curve()`.
5. **Update data input**: Ensure sequences are passed as plain lists. If using pandas, use `df['col'].tolist()`.
