# Changelog

All notable changes to LZGraphs will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom exceptions module with comprehensive exception hierarchy for better error handling
- Information-theoretic metrics module (`LZGraphs.Metrics.entropy`)
  - `node_entropy()` - Shannon entropy of node probability distribution
  - `edge_entropy()` - Shannon entropy of edge transition probabilities
  - `graph_entropy()` - Combined graph entropy measure
  - `normalized_graph_entropy()` - Size-normalized entropy for comparison
  - `sequence_perplexity()` - Model perplexity for individual sequences
  - `repertoire_perplexity()` - Aggregate perplexity across repertoires
  - `jensen_shannon_divergence()` - Symmetric divergence between graphs
  - `cross_entropy()` - Cross-entropy between model and test distributions
  - `kl_divergence()` - Kullback-Leibler divergence
  - `mutual_information_genes()` - MI between paths and V/J genes
- Pre-commit configuration for code quality
- Modern pyproject.toml configuration with PEP 621 metadata
- Support for Python 3.10, 3.11, and 3.12
- `py.typed` marker for type checking support
- Comprehensive test suite (189 tests)

### Changed
- Migrated project metadata from setup.py to pyproject.toml
- Updated tox.ini for modern Python versions (3.8-3.12)
- Improved error messages with custom exception classes
- Replaced print statements with proper logging

### Removed
- Support for Python 3.6 and 3.7 (end of life)

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

[Unreleased]: https://github.com/MuteJester/LZGraphs/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/MuteJester/LZGraphs/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/MuteJester/LZGraphs/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MuteJester/LZGraphs/releases/tag/v1.0.0
