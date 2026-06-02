# Changelog

All notable changes to LZGraphs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026

### Added
- **FlashBack graph family**: a new graph representation alongside the `LZGraph` variants:
  - `FlashBackGraph`: a strictly Markovian DAG built from the FlashBack decomposition, with **exact** diversity, Shannon entropy, Hill numbers, path counts, and PGEN computed by forward dynamic programming (no Monte Carlo).
  - `FlashBackStream`: incremental builder for open-ended sources, with instant running node/edge counts and checkpoint snapshots.
  - `FlashBackGrammar`: FlashBack decomposition/reconstruction utilities.
- **SCALE anomaly score**: `FlashBackGraph.calibrate_scale()` + `scale_score()` (with the `ScaleCalibration` cache and the `lzg flashback scale` command), a self-calibrated, length-invariant `-log Pgen` score for flagging error/noise sequences.
- `FlashBackGraph.top_k_sequences()`: exact enumeration of the most/least probable sequences via forward DP.
- `FlashBackGraph.without()`: remove the contribution of given sequences for leave-donor-out construction in seconds; plus `posterior()` Bayesian updates and graph algebra (`union` / `intersection` / `difference` / `weighted_merge`).
- Foundation FlashBack graph is now published as a downloadable GitHub release asset (`make publish-foundation`).
- Documentation: `FlashBackGraph` API reference, and a "two graph families" rewrite of the Graph Variants concept page.

### Changed
- Reorganized the C library and added the FlashBack subsystem; improved the Python API surface.

## [3.0.2] - 2026

### Fixed
- Restored scalable public `simulate()` and `pgen()` semantics on large graphs while preserving sequence/log-prob consistency.
- Improved probability diagnostics, classical Hill-number estimation, and zero-probability repertoire perplexity handling.
- Added a standalone C benchmark harness for graph loading, simulation, scoring, analytics, and I/O throughput measurements.

### Changed
- Refactored the C core into smaller internal modules across simulation, graph finalization/build ingest, analytics, PGEN distribution, I/O, graph operations, occupancy, and diversity.
- Clarified public documentation around constrained simulation, approximate accepted-walk normalization, and the unconstrained forward-DP `pgen_dist` approximation.

## [3.0.1] - 2026

### Fixed
- Improved large-file graph construction with streaming plain-text ingestion, safer capacity handling, and `uint64` count support.
- Fixed plain `sequence<TAB>count` parsing and added stricter input-validation pathways in the CLI and Python APIs.
- Improved save metadata to record the correct library version in `.lzg` files.

### Changed
- Significantly accelerated Foundation-graph query paths for `pgen()` and `simulate()` without changing graph or traversal semantics.
- Improved long-running build logging with clearer progress, phase reporting, and operational safety checks.

## [3.0.0] - 2026

**Major Rewrite: High-Performance C-Core**

This version is a complete re-implementation of the LZGraphs engine in C, providing 100x-1000x faster construction and analysis while significantly reducing memory overhead.

### Added
- **C-Core Engine**: Core graph operations, LZ76 decomposition, and generative modeling now run in a high-performance C backend.
- **Unified `LZGraph` Class**: Replaced `AAPLZGraph`, `NDPLZGraph`, and `NaiveLZGraph` with a single `LZGraph` class using a `variant` parameter (`'aap'`, `'ndp'`, `'naive'`).
- **Binary Format (`.lzg`)**: New custom binary format for saving/loading graphs that is faster and more compact than `pickle`.
- **LZ-Constrained Model**: Simulation (`simulate()`) and probability scoring (`pgen()`) now strictly enforce LZ76 dictionary constraints at every step.
- **Analytical Moments**: Exact computation of log-PGEN mean, variance, skewness, and kurtosis via topological forward propagation (O(V+E)).
- **Occupancy Predictions**: Advanced `predicted_richness` and `predicted_overlap` algorithms using splitting + Taylor series + Wynn epsilon acceleration for machine-precision results at any depth.
- **Feature Alignment**: New `feature_aligned()` method to project any repertoire into the node space of a reference graph for consistent ML features.
- **Logging System**: New `set_log_level()` and `set_log_callback()` for controlling C-core diagnostic output.

### Changed
- **Dependencies**: Removed `networkx`, `scipy`, `tqdm`, and `matplotlib` from core dependencies. `numpy` is now the only required dependency.
- **Python API**: Simplified and modernized API:
    - `walk_probability()` → `pgen()`
    - `random_walk()` / `genomic_random_walk()` → `simulate()`
    - `get_posterior()` → `posterior()`
    - `k1000_diversity()` → `k_diversity()`
- **Attributes**: Renamed for consistency: `lengths` → `length_distribution`, `marginal_v_genes` → `v_marginals`, `marginal_j_genes` → `j_marginals`.

### Removed
- **Visualization Module**: The `LZGraphs.visualization` module has been removed to eliminate heavy dependencies.
- **LZBOW Vectorizer**: Replaced by the more robust `feature_aligned()` projection.
- **Legacy Metrics**: `lz_centrality`, `node_entropy`, `edge_entropy`, and `transition_jsd` have been removed in favor of more principled analytical moments and JSD.

## [2.5.0] - 2026

### Added
- **Distribution analytics**: characterization of the generative probability distribution.
- `simulation_potential_size()`: count of unique producible sequences.
- `pgen_diagnostics()`: mass conservation check.
- `effective_diversity()`: Shannon entropy and N_eff.
- `predict_sharing_spectrum(draw_counts)`: sharing spectrum via analytical quadrature.

## [2.2.0] - 2026

### Added
- **Bayesian posterior personalization** via `get_posterior()`.
- Dirichlet-Multinomial conjugacy for updating population priors.

### Changed
- **`pandas` is no longer a required dependency**.
- Internal attributes moved from `pd.Series` to plain `dict`.

## [1.1.1] - 2024-01-01

### Fixed
- Compatibility patches for newer Python and Pandas versions
- Fixed deprecated pandas operations

## [1.1.0] - 2023-12-01

### Added
- Major structure update for improved readability and efficiency
- Faster graph creation runtime
- New Metrics submodule with K1000 and LZCentrality functions

### Changed
- Restructured imports for cleaner organization
- Improved runtime performance for graph operations

## [1.0.0] - 2023-06-01

### Added
- Initial stable release
- AAPLZGraph (Amino Acid Positional LZ Graph)
- NDPLZGraph (Nucleotide Double Positional LZ Graph)
- NaiveLZGraph (Simple LZ Graph without position encoding)
- LZBOW (Bag of Words encoder using LZ decomposition)
- Graph visualization utilities
- Sequence generation via random walks
- V/J gene prediction capabilities
- Node and edge saturation analysis
- Graph serialization (JSON, pickle)
- Example notebooks and sample data

### Changed
- Beta refinements from 0.x versions

## [0.26] - 2023-03-01

### Added
- K1000 metric function
- LZCentrality metric function
- New Metrics submodule

### Changed
- Updated documentation

## [0.25] - 2023-02-01

### Changed
- Removed redundant imports
- Updated requirements

## [0.24] - 2023-01-01

### Added
- Example notebooks
- Sample data files

### Changed
- Code and documentation updates

[3.1.0]: https://github.com/MuteJester/LZGraphs/compare/v3.0.2...v3.1.0
[3.0.2]: https://github.com/MuteJester/LZGraphs/compare/v3.0.1...v3.0.2
[3.0.1]: https://github.com/MuteJester/LZGraphs/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/MuteJester/LZGraphs/compare/v2.5.0...v3.0.0
[2.5.0]: https://github.com/MuteJester/LZGraphs/compare/v2.2.0...v2.5.0
[2.2.0]: https://github.com/MuteJester/LZGraphs/compare/v1.1.1...v2.2.0
[1.1.1]: https://github.com/MuteJester/LZGraphs/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/MuteJester/LZGraphs/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MuteJester/LZGraphs/releases/tag/v1.0.0
