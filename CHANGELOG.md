# Changelog

All notable changes to LZGraphs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `FlashBackPseqAnalysis.attribution(q)` for exact, sampling-free node and edge
  marginals under path weights proportional to `P(sequence)**q`, including
  edge sensitivities, surprisal contributions, and ranked edge summaries. The
  native implementation uses stable log-domain forward-backward passes and
  exposes lifetime-safe, read-only zero-copy arrays.

- **Analytical publicness prediction** (C implementation in `lib/lzgraph/occupancy/publicness.c`, Python API in `LZGraphs._publicness`): the Poisson-binomial repertoire-occupancy PMF. Publicness is in how many of a cohort's repertoires a sequence is present. Detection in a repertoire of depth `N` has probability `1 - (1-p)^N`, and depths differ, so the occupancy count is Poisson-binomial, not binomial; its PMF is recovered from the probability generating function `G(z) = prod_b (1 - pi_b + pi_b z)^{m_b}` sampled at the K-th roots of unity and inverted by a forward DFT. New public entry points:
  - `FlashBackPseqAnalysis.publicness_distribution(depths, levels=...)`: predicted sequences per publicness bin, aggregated over the graph's counting spectrum. Also available on `NaiveGraph` and `FlattenedFlashBackGraph`, which share the same analysis object.
  - New public type `PublicnessModel`, the cohort model on its own: `pmf(p)`, `moments(p)`, `bin_mass(p, edges)` and `expected_counts(probabilities, multiplicities)` for any explicit probability spectrum, with no graph involved.
  - C entry points `lzg_publicness_moments()`, `lzg_publicness_pgf()` and `lzg_publicness_accumulate()` in the new `include/lzgraph/publicness.h`. The generating-function product uses complex exponentiation by squaring rather than an accumulated argument, which holds the relative error near `2*log2(m)*eps` instead of the 1e-11 an angle of 1e5 radians would carry. The DFT itself stays in numpy: the library has no C FFT and adding one is not worth a third-party dependency, and the transform is not the expensive part.

  This is not `expected_frequency_spectrum(n, max_count)`, which counts how many times a sequence appears inside *one* pool of `n` draws, nor `predict_sharing()`, which integrates the Gaussian-mixture PGEN approximation against a Poisson detection model. Publicness counts how many *separate* repertoires contain a sequence at all, and is driven by the exact p-sequence spectrum.

  **The truncation in `lzg_publicness_accumulate()` is load-bearing and must not be simplified away.** Every atom's PMF is scaled by its multiplicity in the counting spectrum, which on a foundation-scale graph reaches 1.2e29 and sums to a D0 of 1.9e32. Atoms deep in the tail have `p` around 1e-80, so their true PMF is a delta at zero, but the DFT returns that delta with relative roundoff of order 1e-16 smeared across all 65,536 output bins, and 1e-16 times 1e32 is 1e16. Clipping at zero, the natural response to the tiny negatives that appear, keeps the positive half of that noise and rectifies it into a floor: in the case that prompted this work it produced roughly 2,000 spurious sequences in *every* publicness bin, predicting 2,114 in the top bin where 0 were observed and where only 135 atoms, carrying 108 sequences between them, could physically reach at all. Both moments of a Poisson-binomial are available in closed form without touching the PMF, so each atom is truncated to `[mu - 40*sigma, mu + 40*sigma]`, widened by an absolute floor of 256 levels for the degenerate `mu ~ 0` case, and zeroed outside that window *before* negatives are clipped. At 40 sigma the Chernoff bound on the discarded mass is below 1e-300. Each atom's retained mass is checked against 1 so that a truncation mistake raises instead of quietly reshaping a tail.
- **`NaiveGraph`** (C implementation in `lib/naive/`, variant string `naive_positional`): a positional-encoding graph where each node is one residue plus its 1-based index, so `CASS` walks `@ -> C_1 -> A_2 -> S_3 -> S_4 -> $`. It shares the CSR engine, MLE edge weights, and exact forward-DP analytics with `FlashBackGraph` and differs only in what a node is, which makes it the controlled baseline for measuring what the FlashBack decomposition itself contributes. Supports `pgen`, `simulate`, `path_count` (exact, arbitrary precision), `entropy`, `hill_number(s)`, `power_sum`, `pgen_dynamic_range`, `pgen_diagnostics`, set operations, and `save`/`load`. `max_length` (default 27) bounds the node set and, more importantly, fixes what a comparison against another model is trained on. Intentionally lightly documented: it exists as a control, not as a recommended analysis tool. `top_k_sequences`, `posterior`, `without`, and streaming construction are not implemented for this variant.
- **`FlattenedFlashBackGraph`** (C implementation in `lib/flashback/flat_flashback.c`, variant string `flattened_flashback`): FlashBack's bilateral scan with run compression removed. It walks inward from both ends like FlashBack but takes exactly one residue from each end per step, so `CASSAYFF` becomes `@ -> CF_1 -> AF_2 -> SY_3 -> SA_4 -> $`. A token is `{front}{back}_{step}`; an odd-length sequence leaves one residue unpaired in the middle, written `{residue}$_{step}`. It exists to isolate one variable at a time: against `FlashBackGraph` it holds the bilateral scan constant and removes only run compression, and against `NaiveGraph` it holds "no compression" constant and adds only the bilateral scan. Same API surface as `NaiveGraph`, including `pseq`, `pseq_analysis`, exact `path_count`, and the `pgen*` compatibility aliases.
- `flat_decompose(sequence)` module-level function.
- New `LZGVariant` value `LZG_VARIANT_FLAT_FB = 4`.
- `naive_decompose(sequence)` module-level function, the counterpart of `flashback_decompose`.
- `NaiveGraph.pseq_analysis()`, returning the same `FlashBackPseqAnalysis` that `FlashBackGraph.pseq_analysis()` does. The Mellin transform and everything derived from it are forward dynamic programs over edges, so they are generic over any sentinel-bounded DAG.
- New `LZGVariant` value `LZG_VARIANT_NAIVE_POS = 3`. Old readers reject `.lzg` files carrying it with `LZG_ERR_INVALID_VARIANT`.

### Naming

- On `NaiveGraph` the probability methods are spelled **`pseq`**, not `pgen`: `pseq`, `pseq_moments`, `pseq_diagnostics`, `pseq_dynamic_range`, `pseq_dynamic_range_detail`. The quantity is the probability of *observing* a sequence under a model fitted to observed repertoires, which is not the probability of the VDJ machinery *generating* it — the `Pgen` that OLGA and the LZ graphs report. The `pgen*` spellings remain as aliases to the same functions so code that accepts either graph class (the shared spectrum and scoring pipelines call `.pgen()` and `.pgen_moments()` on whichever graph they are handed) keeps working. The C symbols keep the library-wide `lzg_*_pgen` spelling; only the Python API renames. `FlashBackGraph` is unchanged.

### Fixed

- Set operations (`union`, `intersection`, `difference`, `weighted_merge`) and `feature_aligned` re-derived node labels by appending `node_pos` whenever the variant was not `LZG_VARIANT_NAIVE`, which corrupted labels for any variant whose label already carries its own identity (`A_2` became `A_2_4294967295`). The check is now on the actual invariant, `node_pos == UINT32_MAX`.
- `FlashBackPseqAnalysis` computed each node's character contribution as `len(label.rsplit("_", 1)[0])`, counting sentinel characters as residues. Every reconstructed sequence length was therefore too long for any encoding whose labels carry a sentinel: one too long for a bare `"$"` sink, and one too long again for a `"{residue}$_{k}"` middle token. The rule is now "count the characters of the token base that are not `@` or `$`", which is correct for every variant at once. FlashBack graphs are unaffected, verified two ways: no node label on the foundation graph is a bare sentinel, and its `length_spectra.csv` and `length_summary.csv` regenerate byte-identically after the change.

## [3.2.0] - 2026

### Added

- **Sampling-free p-sequence analytics for `FlashBackGraph`** via the new `pseq_analysis()` method, returning a `FlashBackPseqAnalysis`. The exact core is the Mellin/power-sum transform `M(q) = sum_s P(s)^q`, computed as a forward dynamic program over graph edges with no Monte Carlo:
  - `mellin(q)` / `log_mellin(q)` / `derivatives(q, order)`: exact transform and its derivatives.
  - `moments()` / `cumulants()`: exact surprisal moments and cumulants.
  - `length_profile()` / `length_derivatives(q, order)`: exact mass and surprisal moments stratified by reconstructed sequence length.
  - `exact_atoms()`: exhaustive enumeration of probability atoms for small supports.
  - `histogram()`: deterministic surprisal-grid reconstruction for large supports, reporting its grid spacing and a rounding-error bound, with no Monte Carlo variance.
  - `saddlepoint()`: smooth Lugannani-Rice PDF/CDF inversion built from exact transform quantities.
  - `position(sequence)`: locates one sequence in both the generated and counting spectra.
  - `expected_richness(n)` / `expected_frequency_spectrum(n, max_count)`: finite-depth occupancy prediction.
- New public types `FlashBackPseqAnalysis`, `PseqAtoms`, `PseqHistogram`, and `PseqSaddlepoint`.
- **Input layer** (`LZGraphs._io`): content-based format detection for FASTA, FASTQ, AIRR TSV/CSV, plain, and `sequence<TAB>count`, under transparent gzip, bzip2 and xz. Formats are detected by content, not by file extension, so a misnamed file still works. zstd is recognised and reports an install hint.
- `read_sequences` returns a `RecordStats` under a new `stats` key, counting total, kept, malformed and non-productive records.
- AIRR `productive` filtering, with `keep_nonproductive=True` to retain them.
- **Terminal layer** (`LZGraphs._term`): a zero-dependency renderer with a live panelled display on a TTY and a scrolling, greppable `key=value` log in CI and pipes. New global flags `--ui {auto,rich,plain,quiet}` and `--no-color`. `NO_COLOR`, `TERM=dumb` and `CI` are honoured.
- `lzg build` now reports dropped-record counts and warns when the input alphabet does not match the chosen engine.

### Changed

- `FlashBackGraph.path_count` is now exact in arbitrary precision and returns a Python `int` instead of a `float`. It is computed natively by the new `lzg_flashback_path_count_exact()` C entry point, which carries the count in base-2^32 limbs over a topological DAG dynamic program. The previous double accumulator saturated at 2^53 and overflowed to infinity past ~1.8e308; on a 71k-node, 11.7M-edge foundation graph the true count is 36 digits, of which a double preserved only 16. Note that `hill_number(0)`, `power_sum(0)`, and the `uniformity` field of `diversity_profile()` still use the double-precision path, so they agree with `path_count` only to double precision on large graphs.
- `FlashBackGraph.path_count` now raises `RuntimeError` if the graph has no valid topological order.
- `FlashBackGraph.pgen_distribution()` is documented as the legacy Gaussian-mixture approximation. Its per-component fit still uses sampled walks; `pseq_analysis()` is the sampling-free replacement.
- **`--expect-format` now accepts `fasta` and `fastq`**, which it previously rejected outright, and is an assertion in every path rather than a coercion in some.
- Input that cannot be interpreted now fails loudly rather than being ingested: duplicate column names, an empty or binary file, and a declared format that disagrees with the content.
- `from_file` no longer streams in constant memory for compressed or non-plain input; it buffers, which is the cost of routing those formats through the correct reader.
- stdout carries data only. All presentation goes to stderr, so pipes and redirects are unaffected by rendering.

### Removed

- `LZGraphs._io` is now a package; the former single-module implementation and its 197 unreachable lines are gone.

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
