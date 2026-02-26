# Changelog

All notable changes to LZGraphs are documented here.

This project follows [Semantic Versioning](https://semver.org/).

---

## [2.2.0] - 2026

### Added
- **Bayesian posterior personalization** via `get_posterior(sequences, abundances=None, kappa=1.0)`:
    - Personalizes a population-level LZGraph to an individual repertoire using Dirichlet-Multinomial conjugacy
    - Updates edge weights, initial state probabilities, and stop probabilities
    - `kappa` parameter controls prior strength (higher = trust population more, lower = trust individual more)
    - Novel edges/nodes from the individual are incorporated automatically
    - The returned posterior is a full LZGraph — supports `simulate()`, `walk_probability()`, all metrics, etc.

### Changed
- **`pandas` is no longer a required dependency**. All graph constructors accept plain `list[str]` as the primary input. `pd.DataFrame` and `pd.Series` inputs are still supported via duck-typing for backward compatibility, but pandas is not installed automatically.
- `compare_repertoires()` now returns a plain `dict` instead of `pd.Series`
- Internal attributes (`initial_state_counts`, `terminal_state_counts`, `marginal_v_genes`, `marginal_j_genes`) are now plain `dict` instead of `pd.Series`

### Refactored (internal — no public API change)
- Extracted 5 mixins from `lz_graph_base.py`: `GraphTopologyMixin`, `LZPgenDistributionMixin`, `WalkAnalysisMixin`, `BayesianPosteriorMixin`, `SerializationMixin` — base class reduced from ~2,500 to ~960 lines
- Replaced `verbose_driver` magic-number logging with descriptive `_log_step(message, verbose)` method

---

## [2.1.2] - 2026

### Changed
- **Naming convention cleanup** across the entire public API for consistency:
    - Functions: `LZCentrality` -> `lz_centrality`, `K_Diversity` -> `k_diversity`, `K100_Diversity` -> `k100_diversity`, `K500_Diversity` -> `k500_diversity`, `K1000_Diversity` -> `k1000_diversity`, `K5000_Diversity` -> `k5000_diversity`, `adaptive_K_Diversity` -> `adaptive_k_diversity`
    - Attributes: `.genetic` -> `.has_gene_data`, `.subpattern_individual_probability` -> `.node_probability`, `.terminal_states` -> `.terminal_state_counts`, `.initial_states` -> `.initial_state_counts`, `.initial_states_probability` -> `.initial_state_probabilities`, `.length_distribution` -> `.length_counts`, `.length_distribution_proba` -> `.length_probabilities`, `.per_node_observed_frequency` -> `.node_outgoing_counts`, `.marginal_vgenes` -> `.marginal_v_genes`, `.marginal_jgenes` -> `.marginal_j_genes`, `.n_subpatterns` -> `.num_subpatterns`, `.n_transitions` -> `.num_transitions`, `.initial_state_threshold` -> `.min_initial_state_count`
    - Methods: `clean_node()` -> `extract_subpattern()`, `cac_random_gene_walk()` -> `vj_combination_random_walk()`
    - Visualization: `draw_graph` -> `plot_graph`, `sequence_genomic_edges_variability_plot` -> `plot_gene_edge_variability`, `sequence_genomic_node_variability_plot` -> `plot_gene_node_variability`, `sequence_possible_paths_plot` -> `plot_possible_paths`, `ancestors_descendants_curves_plot` -> `plot_ancestor_descendant_curves`
    - Parameters: `Vgene=` -> `v_gene=`, `Jgene=` -> `j_gene=`
    - Internal dict keys: `'wsif/sep'` -> `'stop_probability'`

### Added
- **Sequence abundance weighting** documentation: all three graph types (AAPLZGraph, NDPLZGraph, NaiveLZGraph) support weighting sequences by clonotype abundance during graph construction
- New tutorial section on abundance weighting in the Graph Construction guide

---

## [2.1.0] - 2026

### Added
- **Information-theoretic metrics** for advanced repertoire characterization:
    - `transition_predictability` — measures transition determinism, stable across sample sizes (~0.60 for AAPLZGraph)
    - `graph_compression_ratio` — quantifies path sharing efficiency (edge reuse)
    - `repertoire_compressibility_index` — compression-framed alias for transition predictability
    - `path_entropy_rate` — average bits per subpattern step via Monte Carlo
    - `transition_kl_divergence` — transition-level KL divergence between two graphs
    - `transition_jsd` — symmetric, bounded transition-level Jensen-Shannon divergence
    - `transition_mutual_information_profile` — position-specific MI along the CDR3 sequence
- `compare_repertoires` now includes `transition_jsd`, `transition_predictability_1`, and `transition_predictability_2`
- `LZPgenDistribution` — analytical Pgen distribution as a Gaussian mixture (no sampling needed)
- `compare_lzpgen_distributions` — compare two empirical log-probability distributions
- `simulate(n, seed, return_walks)` method for fast batch sequence generation with pre-computed walk cache
- New example notebooks: **Information-Theoretic Analysis**, **LZPgen Example**, **Advanced Features**

---

## [2.0.0] - 2026

### Changed
- **Breaking**: All internal modules renamed to snake_case (graphs/, metrics/, utilities/, mixins/, etc.)
- Complete `EdgeData` refactor — raw counts as source of truth
- `graph_union` rewritten to merge via `EdgeData.merge()` + `recalculate()`
- Walk probability model consolidated into LZGraphBase
- Laplace smoothing via `smoothing_alpha` parameter

### Added
- `remove_sequence()` method on LZGraphBase
- `recalculate()` method to recompute all derived state from raw counts
- `to_networkx()` for external tool compatibility
- `walk_log_probability` on all graph types
- Professional documentation with MkDocs Material theme
- Comprehensive tutorials, how-to guides, and API reference

---

## [1.1.1] - 2024

### Fixed
- Fixed `gene_variation()` to correctly identify V/J genes (gene names like "TRBV30-1*01")
- Fixed visualization functions to use correct graph attribute access
- Added `clean_node()` method to `NaiveLZGraph` for consistency

### Added
- `use_log` parameter to `walk_probability()` for all graph types
- `save()` and `load()` methods to `NaiveLZGraph`
- Enhanced Jupyter notebook examples

---

## [1.1.0] - 2024

### Added
- Custom exceptions module with comprehensive error hierarchy
- Automated CI/CD pipeline with semantic-release
- Type hints throughout the codebase
- `py.typed` marker for type checker support

### Changed
- Updated minimum Python version to 3.9
- Improved docstrings for all public APIs
- Enhanced test coverage

---

## [1.0.0] - 2024

### Added
- Initial stable release
- `AAPLZGraph` for amino acid sequences
- `NDPLZGraph` for nucleotide sequences
- `NaiveLZGraph` for non-positional analysis
- K-diversity metrics (K100, K500, K1000, K5000)
- Entropy and perplexity functions
- Jensen-Shannon divergence
- Visualization functions
- LZBOW vectorizer
- NodeEdgeSaturationProbe

---

## Version History

For the complete version history, see the [GitHub Releases](https://github.com/MuteJester/LZGraphs/releases) page.

## Migration Guides

### Upgrading to 2.2.0

- **pandas removal**: If your code passes `pd.DataFrame` or `pd.Series` to graph constructors, it still works — no changes needed. But you can now pass plain lists instead and drop the pandas dependency from your own project.
- **`compare_repertoires` return type**: Previously returned `pd.Series`, now returns `dict`. Replace `result['metric']` with `result['metric']` (same syntax) or `result.metric` → `result['metric']`.
- **New method**: `get_posterior()` is available on all graph types. See the [Posterior Personalization guide](../how-to/posterior-personalization.md).

### Upgrading to 2.1.2

v2.1.2 renames many public API names to follow consistent snake_case conventions. **This is a clean break — old names are removed, not deprecated.** Key changes: `LZCentrality` -> `lz_centrality`, `K1000_Diversity` -> `k1000_diversity`, `clean_node()` -> `extract_subpattern()`, `.initial_states` -> `.initial_state_counts`, `.terminal_states` -> `.terminal_state_counts`, `.marginal_vgenes` -> `.marginal_v_genes`, visualization functions renamed (e.g., `draw_graph` -> `plot_graph`). Pickle/JSON files saved with old names are handled automatically via migration logic. See the full rename table in the v2.1.2 changelog entry above.

### Upgrading to 2.1.x

No breaking changes from 2.0. New information-theoretic metrics are additive.

### Upgrading from 1.x to 2.0

- All internal module paths changed to snake_case (e.g., `LZGraphs.graphs.amino_acid_positional`)
- Edge data now uses `EdgeData` objects: access via `graph[a][b]['data'].weight`
- Public class/function names unchanged — imports like `from LZGraphs import AAPLZGraph` still work

### Upgrading from Pre-1.0

If upgrading from early development versions:

1. Update import statements to use new module structure
2. Use `AAPLZGraph` instead of deprecated class names
3. Update column names to `cdr3_amino_acid` / `cdr3_rearrangement`
