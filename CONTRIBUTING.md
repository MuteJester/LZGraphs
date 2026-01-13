# Contributing to LZGraphs

Thank you for your interest in contributing to LZGraphs! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone regardless of experience level.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/LZGraphs.git
   cd LZGraphs
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/MuteJester/LZGraphs.git
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the package in development mode with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Running Tests

Run the full test suite:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=src/LZGraphs --cov-report=term-missing
```

Run a specific test file:
```bash
pytest tests/test_aap_lzgraph.py -v
```

Run tests matching a pattern:
```bash
pytest tests/ -k "test_walk" -v
```

## Code Style

We use the following tools to maintain code quality:

- **Black** - Code formatting (line length: 100)
- **isort** - Import sorting
- **Ruff** - Linting

### Formatting Code

Format all code:
```bash
black src/LZGraphs tests
isort src/LZGraphs tests
```

Check formatting without changes:
```bash
black --check src/LZGraphs tests
isort --check-only src/LZGraphs tests
```

### Linting

Run the linter:
```bash
ruff check src/LZGraphs tests
```

Fix auto-fixable issues:
```bash
ruff check --fix src/LZGraphs tests
```

### Pre-commit Hooks

Pre-commit hooks run automatically on each commit. To run manually:
```bash
pre-commit run --all-files
```

## Submitting Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-metric` - New features
- `fix/walk-probability-error` - Bug fixes
- `docs/update-api-reference` - Documentation
- `refactor/optimize-graph-creation` - Refactoring

### Commit Messages (Conventional Commits)

We use [Conventional Commits](https://www.conventionalcommits.org/) for automatic versioning and changelog generation. Your commit messages **directly control version bumps**:

**Format:**
```
<type>(<optional scope>): <description>

[optional body]

[optional footer]
```

**Types and their effects:**

| Type | Version Bump | Description |
|------|--------------|-------------|
| `feat` | **Minor** (1.x.0) | New feature for users |
| `fix` | **Patch** (1.1.x) | Bug fix for users |
| `perf` | **Patch** (1.1.x) | Performance improvement |
| `docs` | No bump | Documentation only |
| `style` | No bump | Formatting, missing semicolons, etc. |
| `refactor` | No bump | Code change that neither fixes a bug nor adds a feature |
| `test` | No bump | Adding or correcting tests |
| `chore` | No bump | Maintenance tasks |
| `ci` | No bump | CI/CD changes |
| `build` | No bump | Build system changes |

**Breaking Changes** (Major version bump):
Add `BREAKING CHANGE:` in the footer or `!` after the type:
```
feat!: remove deprecated walk_probability parameter

BREAKING CHANGE: The `legacy_mode` parameter has been removed.
```

**Examples:**
```bash
# Patch release (1.1.1 -> 1.1.2)
git commit -m "fix: correct edge weight calculation for sparse graphs"

# Minor release (1.1.2 -> 1.2.0)
git commit -m "feat: add jensen-shannon divergence metric"

# Major release (1.2.0 -> 2.0.0)
git commit -m "feat!: redesign graph serialization API"

# No release (documentation)
git commit -m "docs: update API reference for random walks"

# With scope
git commit -m "fix(metrics): handle empty repertoire in k1000"

# With issue reference
git commit -m "fix: resolve memory leak in large graphs

Closes #42"
```

**Important:** Commits that don't follow this format won't trigger releases!

### Pull Request Process

1. Create a new branch from `master`:
   ```bash
   git checkout master
   git pull upstream master
   git checkout -b feature/your-feature
   ```

2. Make your changes and commit them

3. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```

4. Open a Pull Request on GitHub

5. Ensure all checks pass (tests, linting)

6. Request a review from maintainers

### Pull Request Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code is formatted (`black`, `isort`)
- [ ] Linting passes (`ruff check`)
- [ ] New features have tests
- [ ] Documentation is updated if needed
- [ ] Commit messages follow [Conventional Commits](#commit-messages-conventional-commits) format

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code example
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**:
   - Python version
   - LZGraphs version
   - Operating system
   - Relevant package versions

Example:
```markdown
## Description
`walk_probability` raises KeyError for valid sequences

## Steps to Reproduce
```python
from LZGraphs import AAPLZGraph
import pandas as pd

data = pd.DataFrame({'cdr3_amino_acid': ['CASSLGQAYEQYF']})
graph = AAPLZGraph(data)
prob = graph.walk_probability('CASSLGQAYEQYF')  # Raises KeyError
```

## Expected Behavior
Should return the probability of the sequence

## Actual Behavior
Raises `KeyError: 'node_name'`

## Environment
- Python 3.10.4
- LZGraphs 1.1.1
- Ubuntu 22.04
```

### Feature Requests

For feature requests, please describe:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing!
