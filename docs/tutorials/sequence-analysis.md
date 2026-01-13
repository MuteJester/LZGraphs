# Sequence Analysis

This tutorial covers analyzing sequences using LZGraphs, including probability calculation, encoding, and sequence generation.

## Prerequisites

Build a graph first:

```python
import pandas as pd
from LZGraphs import AAPLZGraph

data = pd.read_csv("Examples/ExampleData1.csv")
graph = AAPLZGraph(data, verbose=True)
```

---

## Sequence Encoding

Before analyzing a sequence, you must encode it into the graph's format.

### Encoding with AAPLZGraph

```python
sequence = "CASRGERGDNEQFF"

# Encode the sequence
encoded = AAPLZGraph.encode_sequence(sequence)
print(encoded)
```

**Output:**
```python
['C_1', 'A_2', 'S_3', 'R_4', 'G_5', 'E_6', 'RG_8', 'D_9', 'N_10', 'EQ_12', 'F_13', 'F_14']
```

Each node has the format `<subpattern>_<position>`.

### Decoding Back to Sequence

```python
# Clean each node to get the original subpattern
clean_nodes = [AAPLZGraph.clean_node(node) for node in encoded]
print(clean_nodes)

# Reconstruct the sequence
reconstructed = ''.join(clean_nodes)
print(f"Original:      {sequence}")
print(f"Reconstructed: {reconstructed}")
```

---

## Calculating Sequence Probability (Pgen)

The generation probability quantifies how likely a sequence is given the repertoire:

### Basic Probability

```python
sequence = "CASRGERGDNEQFF"
encoded = AAPLZGraph.encode_sequence(sequence)

pgen = graph.walk_probability(encoded)
print(f"{sequence}: P(gen) = {pgen:.2e}")
```

**Output:**
```
CASRGERGDNEQFF: P(gen) = 6.69e-13
```

### Log Probability

For very small probabilities, use log-space to avoid numerical underflow:

```python
log_pgen = graph.walk_probability(encoded, use_log=True)
print(f"log P(gen) = {log_pgen:.2f}")
```

!!! tip "When to use log probability"
    Use `use_log=True` when:

    - Comparing many sequences
    - Working with very rare sequences
    - Performing numerical operations on probabilities

### Handling Unknown Sequences

If a sequence contains patterns not in the graph:

```python
unknown_seq = "CASSXYZABC"  # XYZ unlikely in real repertoire
encoded = AAPLZGraph.encode_sequence(unknown_seq)

pgen = graph.walk_probability(encoded, verbose=True)
print(f"P(gen) = {pgen}")  # Returns 0 if path doesn't exist
```

---

## Generating New Sequences

Generate sequences that follow the statistical patterns of your repertoire.

### Unsupervised Random Walk

Generate without gene constraints:

```python
walk = graph.random_walk()
sequence = ''.join([AAPLZGraph.clean_node(node) for node in walk])
print(f"Generated: {sequence}")
```

### Gene-Constrained Generation

Generate sequences consistent with specific V/J genes:

```python
# Generate with random V/J selection
walk, v_gene, j_gene = graph.genomic_random_walk()

sequence = ''.join([AAPLZGraph.clean_node(node) for node in walk])
print(f"Sequence: {sequence}")
print(f"V gene: {v_gene}")
print(f"J gene: {j_gene}")
```

**Output:**
```
Sequence: CSATGGTGGELFF
V gene: TRBV29-1*01
J gene: TRBJ2-5*01
```

### Generating Multiple Sequences

```python
generated = []
for _ in range(100):
    walk, v, j = graph.genomic_random_walk()
    seq = ''.join([AAPLZGraph.clean_node(n) for n in walk])
    generated.append({'sequence': seq, 'v_gene': v, 'j_gene': j})

df = pd.DataFrame(generated)
print(df.head())
```

---

## Exploring Graph Structure

### Node Properties

```python
# Check if a node exists
node = "C_1"
exists = graph.graph.has_node(node)
print(f"Node {node} exists: {exists}")

# Get node successors (outgoing edges)
successors = list(graph.graph.successors("C_1"))
print(f"C_1 can transition to: {successors[:5]}...")
```

### Edge Properties

```python
# Check edge weight
if graph.graph.has_edge("C_1", "A_2"):
    weight = graph.graph["C_1"]["A_2"]["weight"]
    print(f"Edge C_1 -> A_2 weight: {weight:.4f}")
```

### Subpattern Probabilities

```python
# Get probability of a subpattern at a position
proba_df = graph.subpattern_individual_probability
print(proba_df.head())
```

---

## Working with Terminal States

Terminal states are the final subpatterns of sequences:

```python
# All terminal states with counts
print(graph.terminal_states.head(10))

# Check if a node is terminal
node = "F_15"
is_terminal = node in graph.terminal_states.index
print(f"{node} is terminal: {is_terminal}")
```

### Terminal State Map

Get terminal states reachable from any position:

```python
# Get terminal states for position 12
terminals_at_12 = graph.terminal_states_map.get(12, {})
print(f"Terminal states at position 12: {list(terminals_at_12.keys())[:5]}")
```

---

## Practical Example: Sequence Comparison

Compare sequences by their generation probability:

```python
sequences = [
    "CASSLEPSGGTDTQYF",  # Likely common
    "CASSLGQGSTEAFF",    # Also common
    "CASSXYZABCDEFGH",   # Likely rare/impossible
]

results = []
for seq in sequences:
    try:
        encoded = AAPLZGraph.encode_sequence(seq)
        pgen = graph.walk_probability(encoded, use_log=True)
        results.append({'sequence': seq, 'log_pgen': pgen})
    except Exception as e:
        results.append({'sequence': seq, 'log_pgen': float('-inf')})

df = pd.DataFrame(results)
df = df.sort_values('log_pgen', ascending=False)
print(df)
```

---

## Next Steps

- [Diversity Metrics Tutorial](diversity-metrics.md) - Measure repertoire diversity
- [Concepts: Probability Model](../concepts/probability-model.md) - Understand how Pgen works
- [How-To: Generate Sequences](../how-to/sequence-generation.md) - Advanced generation techniques
