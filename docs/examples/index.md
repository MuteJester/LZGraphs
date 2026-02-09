# Examples Gallery

Interactive Jupyter notebooks demonstrating LZGraphs in action.

## Available Notebooks

<div class="example-grid" markdown>

<div class="example-card" markdown>
![AAPLZGraph Example](../images/ad_curve_example.png)

### AAPLZGraph Example

**Complete amino acid graph tutorial**

Build graphs, calculate probabilities, generate sequences, and visualize results.

[:material-notebook: View Notebook](https://github.com/MuteJester/LZGraphs/blob/master/Examples/AAPLZGraph%20Example.ipynb){ .md-button }
</div>

<div class="example-card" markdown>
![NDPLZGraph Example](../images/sequence_path_number_example.png)

### NDPLZGraph Example

**Nucleotide sequence analysis**

Work with nucleotide CDR3 sequences using double positional encoding.

[:material-notebook: View Notebook](https://github.com/MuteJester/LZGraphs/blob/master/Examples/NDPLZGraph%20Example.ipynb){ .md-button }
</div>

<div class="example-card" markdown>
![Metrics Example](../images/number_of_vj_at_nodes_example.png)

### Metrics Example

**Diversity and entropy analysis**

Calculate K1000, LZCentrality, entropy metrics, and compare repertoires.

[:material-notebook: View Notebook](https://github.com/MuteJester/LZGraphs/blob/master/Examples/Metrics%20Example.ipynb){ .md-button }
</div>

<div class="example-card" markdown>
![NaiveLZGraph Example](../images/number_of_vj_at_edges_example.png)

### NaiveLZGraph Example

**Feature extraction for ML**

Use NaiveLZGraph for consistent feature vectors and cross-repertoire analysis.

[:material-notebook: View Notebook](https://github.com/MuteJester/LZGraphs/blob/master/Examples/NaiveLZGraph%20Example.ipynb){ .md-button }
</div>

<div class="example-card" markdown>

### Information-Theoretic Analysis

**Advanced repertoire characterization**

Transition predictability, compression ratio, path entropy rate, transition JSD, mutual information profiles, and repertoire fingerprinting.

[:material-notebook: View Notebook](https://github.com/MuteJester/LZGraphs/blob/master/Examples/Information-Theoretic%20Analysis.ipynb){ .md-button }
</div>

</div>

## Running Notebooks Locally

### 1. Clone the Repository

```bash
git clone https://github.com/MuteJester/LZGraphs.git
cd LZGraphs
```

### 2. Install Dependencies

```bash
pip install -e .
pip install jupyter
```

### 3. Launch Jupyter

```bash
cd Examples
jupyter notebook
```

## Sample Data

The examples use these datasets included in the repository:

| File | Description | Sequences |
|------|-------------|-----------|
| `ExampleData1.csv` | TCR repertoire sample 1 | ~20,000 |
| `ExampleData2.csv` | TCR repertoire sample 2 | ~15,000 |
| `ExampleData3.csv` | TCR repertoire sample 3 | ~10,000 |

### Data Format

```python
import pandas as pd

data = pd.read_csv("Examples/ExampleData1.csv")
print(data.columns.tolist())
# ['cdr3_amino_acid', 'cdr3_rearrangement', 'V', 'J']

print(data.head())
```

## Quick Examples

### Build Your First Graph

```python
import pandas as pd
from LZGraphs import AAPLZGraph

# Load example data
data = pd.read_csv("Examples/ExampleData1.csv")

# Build graph
graph = AAPLZGraph(data, verbose=True)

# Check stats
print(f"Nodes: {graph.graph.number_of_nodes()}")
print(f"Edges: {graph.graph.number_of_edges()}")
```

### Calculate Diversity

```python
from LZGraphs import K1000_Diversity, AAPLZGraph

sequences = data['cdr3_amino_acid'].tolist()
k1000 = K1000_Diversity(sequences, AAPLZGraph.encode_sequence, draws=30)
print(f"K1000: {k1000:.1f}")
```

### Generate Sequences

```python
# Generate with gene constraints
walk, v_gene, j_gene = graph.genomic_random_walk()
sequence = ''.join([AAPLZGraph.clean_node(n) for n in walk])
print(f"Generated: {sequence}")
print(f"V: {v_gene}, J: {j_gene}")
```

## What's Covered

| Notebook | Topics |
|----------|--------|
| AAPLZGraph | Graph construction, probabilities, random walks, visualization |
| NDPLZGraph | Nucleotide encoding, double positions, gene analysis |
| Metrics | K-diversity, entropy, perplexity, JS divergence |
| NaiveLZGraph | Fixed dictionaries, eigenvector centrality, ML features |
| Information-Theoretic Analysis | Transition predictability, compression ratio, path entropy, TMIP, transition JSD |

## Next Steps

- [Getting Started](../getting-started/index.md) - Installation and basics
- [Tutorials](../tutorials/index.md) - Step-by-step guides
- [API Reference](../api/index.md) - Complete documentation
