# Changelog

All notable changes to LZGraphs are documented here.

This project follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Professional documentation with MkDocs Material theme
- Comprehensive tutorials and how-to guides
- API reference documentation
- FAQ and troubleshooting guide

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

### Upgrading to 1.1.x

No breaking changes. New features are additive.

### Upgrading from Pre-1.0

If upgrading from early development versions:

1. Update import statements to use new module structure
2. Use `AAPLZGraph` instead of deprecated class names
3. Update column names to `cdr3_amino_acid` / `cdr3_rearrangement`
