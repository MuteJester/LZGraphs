# Installation

This guide covers different ways to install LZGraphs and verify your installation.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **C Compiler**: Required only if installing from source or if a pre-built wheel is not available for your platform (GCC, Clang, or MSVC).

## Install from PyPI

The recommended way to install LZGraphs is via pip:

```bash
pip install LZGraphs
```

LZGraphs provides pre-built binary wheels for most common platforms (Linux x86_64/aarch64, macOS arm64, Windows AMD64).

## Install from Source

If you need to install from source (e.g., for a specific architecture or development):

```bash
git clone https://github.com/MuteJester/LZGraphs.git
cd LZGraphs
pip install .
```

*Note: This requires a C compiler and Python development headers.*

### Development Installation

If you want to contribute or run tests:

```bash
pip install -e ".[dev]"
```

This includes additional tools for testing and code quality.

### Visualization Support

To use plotting features in tutorials and examples, install with the `viz` extra:

```bash
pip install "LZGraphs[viz]"
```

## Dependencies

LZGraphs automatically installs these core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.20 | Numerical operations |

Optional extras:

- `[viz]`: `matplotlib`, `seaborn` (for plotting)
- `[dev]`: `pytest`, `ruff`, etc. (for development)

## Verify Installation

After installation, verify everything works:

```python
import LZGraphs
print(f"LZGraphs version: {LZGraphs.__version__}")
```

You should see the version number printed without errors.

### Quick Test

Run a minimal example to ensure all components work:

```python
from LZGraphs import LZGraph

# Create minimal test data
sequences = ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF']

# Build a graph
graph = LZGraph(sequences, variant='aap')
print(f"Graph has {graph.n_nodes} nodes and {graph.n_edges} edges")
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

1. Ensure you're using the correct Python environment
2. Try reinstalling: `pip install --force-reinstall LZGraphs`

### Version Conflicts

If you have dependency conflicts:

```bash
pip install LZGraphs --upgrade
```

Or create a fresh virtual environment:

```bash
python -m venv lzgraphs_env
source lzgraphs_env/bin/activate  # On Windows: lzgraphs_env\Scripts\activate
pip install LZGraphs
```

### Still Having Issues?

- Check the [FAQ](../resources/faq.md) for common problems
- [Open an issue](https://github.com/MuteJester/LZGraphs/issues) on GitHub

## Next Steps

With LZGraphs installed, proceed to the [Core Ideas](core-ideas.md) page, then the [LZGraph Quickstart](quickstart-lzgraph.md) to build your first graph.
