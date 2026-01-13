# Save and Load Graphs

Learn how to persist LZGraphs to disk and reload them later.

## Quick Reference

```python
# Save
graph.save("my_graph.pkl")

# Load
loaded = AAPLZGraph.load("my_graph.pkl")
```

## Saving Graphs

### Basic Save

```python
from LZGraphs import AAPLZGraph
import pandas as pd

# Build a graph
data = pd.read_csv("repertoire.csv")
graph = AAPLZGraph(data, verbose=True)

# Save to disk
graph.save("my_graph.pkl")
```

### Save with Custom Path

```python
from pathlib import Path

# Save to specific directory
output_dir = Path("models/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

graph.save(output_dir / "repertoire_2024.pkl")
```

## Loading Graphs

### Basic Load

```python
from LZGraphs import AAPLZGraph

# Load a saved graph
graph = AAPLZGraph.load("my_graph.pkl")

# Verify it works
print(f"Nodes: {graph.graph.number_of_nodes()}")
print(f"Edges: {graph.graph.number_of_edges()}")
```

### Loading Different Graph Types

```python
from LZGraphs import AAPLZGraph, NDPLZGraph, NaiveLZGraph

# Load the correct type
aap_graph = AAPLZGraph.load("aap_graph.pkl")
ndp_graph = NDPLZGraph.load("ndp_graph.pkl")
naive_graph = NaiveLZGraph.load("naive_graph.pkl")
```

!!! warning "Type Matching"
    You must use the same class to load that was used to save. Loading an `AAPLZGraph` file with `NDPLZGraph.load()` will raise an error.

## Use Cases

### Avoid Recomputation

```python
from pathlib import Path

cache_file = Path("cached_graph.pkl")

if cache_file.exists():
    # Load from cache
    graph = AAPLZGraph.load(cache_file)
    print("Loaded from cache")
else:
    # Build and cache
    data = pd.read_csv("large_repertoire.csv")
    graph = AAPLZGraph(data, verbose=True)
    graph.save(cache_file)
    print("Built and cached")
```

### Batch Processing

```python
from pathlib import Path
import pandas as pd
from LZGraphs import AAPLZGraph

# Process multiple repertoires
input_dir = Path("data/repertoires")
output_dir = Path("data/graphs")
output_dir.mkdir(exist_ok=True)

for csv_file in input_dir.glob("*.csv"):
    graph_file = output_dir / f"{csv_file.stem}.pkl"

    if not graph_file.exists():
        data = pd.read_csv(csv_file)
        graph = AAPLZGraph(data, verbose=False)
        graph.save(graph_file)
        print(f"Saved: {graph_file.name}")
```

### Analysis Pipeline

```python
# Step 1: Build and save (expensive)
# script: build_graphs.py
for repertoire_file in repertoire_files:
    data = pd.read_csv(repertoire_file)
    graph = AAPLZGraph(data)
    graph.save(f"graphs/{repertoire_file.stem}.pkl")

# Step 2: Analyze (fast, can be rerun)
# script: analyze_graphs.py
for graph_file in Path("graphs").glob("*.pkl"):
    graph = AAPLZGraph.load(graph_file)
    # Run analysis...
```

## File Format

LZGraphs uses Python's `pickle` format:

- **Binary format**: Compact and fast
- **Python-specific**: Not portable to other languages
- **Version-sensitive**: Best used with same Python/LZGraphs version

### Checking File Contents

```python
import pickle

with open("my_graph.pkl", "rb") as f:
    obj = pickle.load(f)

print(f"Type: {type(obj)}")
print(f"Has graph: {hasattr(obj, 'graph')}")
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
graph.save("tcr_repertoire_patient001_2024-01.pkl")

# Avoid
graph.save("g1.pkl")
```

### 2. Version Your Graphs

```python
from datetime import datetime
import LZGraphs

# Include version info
filename = f"graph_v{LZGraphs.__version__}_{datetime.now():%Y%m%d}.pkl"
graph.save(filename)
```

### 3. Store Metadata Separately

```python
import json

# Save graph
graph.save("my_graph.pkl")

# Save metadata
metadata = {
    "source_file": "repertoire.csv",
    "n_sequences": len(data),
    "created": str(datetime.now()),
    "lzgraphs_version": LZGraphs.__version__
}

with open("my_graph_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

## Troubleshooting

### "File not found" Error

```python
from pathlib import Path

filepath = Path("my_graph.pkl")
if not filepath.exists():
    print(f"File does not exist: {filepath.absolute()}")
```

### "Invalid pickle" Error

The file may be corrupted or from an incompatible version:

```python
try:
    graph = AAPLZGraph.load("my_graph.pkl")
except Exception as e:
    print(f"Error loading graph: {e}")
    # Rebuild from source data
    data = pd.read_csv("original_data.csv")
    graph = AAPLZGraph(data)
```

### Type Mismatch Error

```python
from LZGraphs import AAPLZGraph, NDPLZGraph

try:
    # Wrong type
    graph = NDPLZGraph.load("aap_graph.pkl")
except TypeError as e:
    print(f"Type mismatch: {e}")
    # Use correct type
    graph = AAPLZGraph.load("aap_graph.pkl")
```

## Next Steps

- [Sequence Generation](sequence-generation.md) - Generate sequences from saved graphs
- [Compare Repertoires](repertoire-comparison.md) - Compare multiple saved graphs
