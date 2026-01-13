# Installation

This guide covers different ways to install LZGraphs and verify your installation.

## Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows, macOS, or Linux
- **Dependencies**: Automatically installed (see below)

## Install from PyPI

The recommended way to install LZGraphs is via pip:

```bash
pip install LZGraphs
```

This will install LZGraphs along with all required dependencies.

## Install from Source

For the latest development version:

```bash
git clone https://github.com/MuteJester/LZGraphs.git
cd LZGraphs
pip install -e .
```

### Development Installation

If you want to contribute or run tests:

```bash
pip install -e ".[dev]"
```

This includes additional tools for testing and code quality.

## Dependencies

LZGraphs automatically installs these dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| `networkx` | ≥3.0 | Graph data structures |
| `numpy` | ≥1.24 | Numerical operations |
| `pandas` | ≥1.5 | Data manipulation |
| `matplotlib` | ≥3.7 | Visualization |
| `seaborn` | ≥0.12 | Statistical plots |
| `scipy` | ≥1.10 | Scientific computing |
| `tqdm` | ≥4.65 | Progress bars |

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
import pandas as pd
from LZGraphs import AAPLZGraph

# Create minimal test data
data = pd.DataFrame({
    'cdr3_amino_acid': ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF']
})

# Build a graph
graph = AAPLZGraph(data, verbose=False)
print(f"Graph has {graph.graph.number_of_nodes()} nodes")
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

With LZGraphs installed, proceed to the [Quick Start](quickstart.md) guide to build your first graph.
