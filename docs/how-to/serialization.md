---
tags:
  - IO
---

# Save and Load Graphs

Learn how to persist LZGraphs to disk using the high-performance binary format.

## Quick Reference

```python
from LZGraphs import LZGraph

# Save
graph.save("my_repertoire.lzg")

# Load
loaded = LZGraph.load("my_repertoire.lzg")
```

## Saving Graphs

LZGraphs uses a custom binary format (`.lzg`) that is optimized for speed and space. It is much faster and more compact than Python's `pickle`.

### Basic Save

```python
from LZGraphs import LZGraph

# Build a graph
sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", "CASSLEPQTFTDTFFF"]
graph = LZGraph(sequences, variant='aap')

# Save to disk
graph.save("my_graph.lzg")
```

### Save with Custom Path

```python
from pathlib import Path

# Save to specific directory
output_dir = Path("models/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

graph.save(output_dir / "repertoire_2026.lzg")
```

## Loading Graphs

The `LZGraph.load()` method automatically detects the graph variant (AAP, NDP, or Naive) from the file metadata.

### Basic Load

```python
from LZGraphs import LZGraph

# Load a saved graph
graph = LZGraph.load("my_graph.lzg")

# Verify it works
print(f"Variant: {graph.variant}")
print(f"Nodes:   {graph.n_nodes}")
print(f"Edges:   {graph.n_edges}")
```

## Use Cases

### Avoid Recomputation

Building large graphs from millions of sequences can take time. Saving the resulting graph allows you to skip the construction step in future sessions.

```python
from pathlib import Path
from LZGraphs import LZGraph

cache_file = Path("cached_graph.lzg")

if cache_file.exists():
    # Load from cache (very fast)
    graph = LZGraph.load(cache_file)
    print("Loaded from cache")
else:
    # Build and cache
    sequences = load_millions_of_sequences()
    graph = LZGraph(sequences, variant='aap')
    graph.save(cache_file)
    print("Built and cached")
```

### Analysis Pipeline

```python
# Step 1: Build and save
for sample in samples:
    graph = LZGraph(sample['sequences'], variant='aap')
    graph.save(f"graphs/{sample['id']}.lzg")

# Step 2: Analyze (fast, can be rerun)
for lzg_file in Path("graphs").glob("*.lzg"):
    graph = LZGraph.load(lzg_file)
    # Run diversity, richness, etc.
```

## The .lzg Format

The `.lzg` format is a specialized binary format designed for LZGraphs:

- **High performance**: Loads and saves at native speed via the C core.
- **Cross-variant**: A single `LZGraph.load()` handles any variant.
- **Robust**: Includes metadata and checksums to prevent corruption.

!!! info "Not a Pickle"
    Unlike previous versions, LZGraphs 3.0+ does NOT use `pickle`. The `.lzg` format is more stable across Python versions and much more efficient for large graphs.

## Best Practices

### 1. Use the .lzg Extension
While not strictly required, using `.lzg` helps identify LZGraph files.

### 2. Version Your Graphs
If you are running long-term experiments, include the version or date in the filename.

```python
import LZGraphs
from datetime import datetime

filename = f"graph_v{LZGraphs.__version__}_{datetime.now():%Y%m%d}.lzg"
graph.save(filename)
```

## Troubleshooting

### "File not found" Error
Ensure the path is correct and accessible.

```python
from pathlib import Path
filepath = Path("my_graph.lzg")
if not filepath.exists():
    print(f"File does not exist: {filepath.absolute()}")
```

### "Corrupt or unsupported LZG file"
This error occurs if the file is not a valid LZGraph binary or was created with an incompatible version of the library.

```python
from LZGraphs import LZGraph, CorruptFileError

try:
    graph = LZGraph.load("possibly_corrupt.lzg")
except CorruptFileError:
    print("The file is invalid or from an old version.")
```

## Next Steps

- [Sequence Generation](sequence-generation.md) - Generate sequences from saved graphs
- [Compare Repertoires](repertoire-comparison.md) - Compare multiple saved graphs
- [Distribution Analytics](distribution-analytics.md) - Analyze the probability model of a loaded graph
