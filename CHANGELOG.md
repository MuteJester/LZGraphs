# Changelog

All notable changes to LZGraphs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026

### Fixed

Two silent-corruption defects in file input, both of which produced a wrong graph with exit code 0 and no warning:

- A FASTA build ingested `>seq10` header lines as sequences. Simulating from the resulting graph emitted `>seq10` as a "sequence".
- A CSV build ingested whole comma-joined rows. Simulating emitted values spliced across fields that never existed in the input.

Further input defects found while closing those:

- A single-column file whose header spells a known sequence column (`junction`, `sequence`, `aminoAcid`, ...) ingested that header as a sequence. `cdr3` and `junction_aa` escaped only incidentally, because a digit and an underscore made the well-formedness check reject them.
- A duplicate column name silently built the entire graph from the wrong column, since `csv.DictReader` keeps only the last occurrence. Now refused with a message naming the column.
- A tabular header was parsed two different ways, so `junction_aa, duplicate_count` with a single space silently lost every abundance, and a quoted duplicate read from the wrong column.
- A UTF-8 BOM merged a FASTA header into the first sequence and corrupted the first TSV column name.
- Lone carriage-return line endings collapsed a whole file into one sequence.
- Abundance `"3.0"`, the shape pandas and R emit whenever a count column is promoted to float, was read as 1. Counts above 2^53 lost precision.
- In the C reader, a count of `0` silently dropped the record and a negative count wrapped to ~1.8e19, reaching the graph as an edge weight.
- `lzg build` on a bzip2 or xz file streamed raw compressed bytes into the builder.
- `LZGraph.from_file` and `FlashBackGraph.from_file`, both documented public API, bypassed the input pipeline entirely, so a user writing Python rather than using the CLI still hit the FASTA and CSV corruption.
- `lzg validate-input` contradicted `lzg build`, reporting a FASTA as `plain` and counting its header lines as records.
- Records dropped as malformed or non-productive were invisible; an all-dropped file failed with the unhelpful `sequences must be a non-empty list`.

### Added

- **Input layer** (`LZGraphs._io`): content-based format detection for FASTA, FASTQ, AIRR TSV/CSV, plain, and `sequence<TAB>count`, under transparent gzip, bzip2 and xz. Formats are detected by content, not by file extension, so a misnamed file still works. zstd is recognised and reports an install hint.
- `read_sequences` returns a `RecordStats` under a new `stats` key, counting total, kept, malformed and non-productive records.
- AIRR `productive` filtering, with `keep_nonproductive=True` to retain them.
- **Terminal layer** (`LZGraphs._term`): a zero-dependency renderer with a live panelled display on a TTY and a scrolling, greppable `key=value` log in CI and pipes. New global flags `--ui {auto,rich,plain,quiet}` and `--no-color`. `NO_COLOR`, `TERM=dumb` and `CI` are honoured.
- `lzg build` now reports dropped-record counts and warns when the input alphabet does not match the chosen engine.

### Changed

- **`--expect-format` now accepts `fasta` and `fastq`**, which it previously rejected outright, and is an assertion in every path rather than a coercion in some.
- Input that cannot be interpreted now fails loudly rather than being ingested: duplicate column names, an empty or binary file, and a declared format that disagrees with the content.
- `from_file` no longer streams in constant memory for compressed or non-plain input; it buffers, which is the cost of routing those formats through the correct reader.
- stdout carries data only. All presentation goes to stderr, so pipes and redirects are unaffected by rendering.

### Removed

- `LZGraphs._io` is now a package; the former single-module implementation and its 197 unreachable lines are gone.

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

[3.2.0]: https://github.com/MuteJester/LZGraphs/compare/v3.1.0...v3.2.0
[3.1.0]: https://github.com/MuteJester/LZGraphs/compare/v3.0.2...v3.1.0
[3.0.2]: https://github.com/MuteJester/LZGraphs/compare/v3.0.1...v3.0.2
[3.0.1]: https://github.com/MuteJester/LZGraphs/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/MuteJester/LZGraphs/compare/v2.5.0...v3.0.0
[2.5.0]: https://github.com/MuteJester/LZGraphs/compare/v2.2.0...v2.5.0
[2.2.0]: https://github.com/MuteJester/LZGraphs/compare/v1.1.1...v2.2.0
[1.1.1]: https://github.com/MuteJester/LZGraphs/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/MuteJester/LZGraphs/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MuteJester/LZGraphs/releases/tag/v1.0.0
