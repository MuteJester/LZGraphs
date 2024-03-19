# Learning Guide for LZGraphs

Welcome to the learning guide for LZGraphs, where we're dedicated to helping you get up and running with this powerful library. Whether you're new to T-cell receptor beta chain (TCRB) repertoire analysis or looking to deepen your understanding, this guide is designed to provide a clear, straightforward path to mastering LZGraphs.

## Getting Started

Diving into a new library can be daunting, but we're here to ensure a smooth learning curve. Start with these simple steps:

1. **Installation**: Make sure LZGraphs is installed in your environment:
    ```bash
    pip install LZGraphs
    ```
2. **First Program**: Let's write your first piece of code to create an LZGraph from a dataset. This immediate hands-on approach will help cement your understanding:
    ```python
    from src.LZGraphs import AAPLZGraph
    import pandas as pd

    # Load your dataset
    data = pd.read_csv("path/to/your/dataset.csv")

    # Initialize and create your LZGraph
    lzgraph = AAPLZGraph(data, verbose=True)

    print("LZGraph created successfully!")
    ```

## Building Confidence Through Code

To build confidence, it's crucial to engage with the library directly. Here's a simple yet impactful exercise:

- **Task**: Use the LZGraph you've created to list the first 10 nodes and edges.
    ```python
    # Display the first 10 nodes
    print("First 10 Nodes:", list(lzgraph.nodes)[:10])

    # Display the first 10 edges
    print("First 10 Edges:", list(lzgraph.edges)[:10])
    ```

## Achieve Immediate Success

We believe in the power of quick wins. Try calculating the LZCentrality for a sequence in your dataset:
```python
from src.LZGraphs import LZCentrality

sequence_of_interest = "your_sequence_here"
lz_centrality_score = LZCentrality(lzgraph, sequence_of_interest)

print(f"LZCentrality for {sequence_of_interest}: {lz_centrality_score}")
```


