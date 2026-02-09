# Changelog

All notable changes to LZGraphs are documented here.

This project follows [Semantic Versioning](https://semver.org/).

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
- New example notebook: **Information-Theoretic Analysis** with full walkthrough and visualizations

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
