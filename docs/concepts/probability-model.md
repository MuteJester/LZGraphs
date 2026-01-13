# Probability Model

LZGraphs calculates sequence generation probability (Pgen) using edge-weighted random walks. This page explains the mathematical foundation.

## The Core Idea

A sequence's probability is the product of:

1. **Initial probability**: Likelihood of starting with the first pattern
2. **Transition probabilities**: Likelihood of each pattern-to-pattern transition

\[
P(\text{sequence}) = P(\text{start}) \times \prod_{i=1}^{n-1} P(\text{node}_{i+1} | \text{node}_i)
\]

## How It Works

### Step 1: Encode the Sequence

```python
from LZGraphs import AAPLZGraph

sequence = "CASSLE"
encoded = AAPLZGraph.encode_sequence(sequence)
print(encoded)
# ['C_1', 'A_2', 'S_3', 'SL_5', 'E_6']
```

### Step 2: Look Up Initial Probability

The probability of starting with `C_1`:

```python
# Initial state probability
p_start = graph.subpattern_individual_probability.loc['C_1', 'proba']
print(f"P(start with C_1) = {p_start:.4f}")
```

### Step 3: Multiply Edge Weights

For each transition, get the edge weight:

```python
# Transition probabilities (edge weights)
p_C1_A2 = graph.graph['C_1']['A_2']['weight']
p_A2_S3 = graph.graph['A_2']['S_3']['weight']
p_S3_SL5 = graph.graph['S_3']['SL_5']['weight']
p_SL5_E6 = graph.graph['SL_5']['E_6']['weight']

# Total probability
pgen = p_start * p_C1_A2 * p_A2_S3 * p_S3_SL5 * p_SL5_E6
print(f"P(CASSLE) = {pgen:.2e}")
```

### Using walk_probability

This is exactly what `walk_probability` computes:

```python
pgen = graph.walk_probability(encoded)
print(f"P(CASSLE) = {pgen:.2e}")
```

## Edge Weight Normalization

Edge weights are normalized transition probabilities:

\[
w(A \to B) = \frac{\text{count}(A \to B)}{\sum_{X} \text{count}(A \to X)}
\]

This ensures outgoing edges from any node sum to 1:

```python
# Check normalization
node = 'C_1'
successors = list(graph.graph.successors(node))
total_weight = sum(graph.graph[node][s]['weight'] for s in successors)
print(f"Sum of weights from {node}: {total_weight:.4f}")  # Should be ~1.0
```

## Log Probability

For numerical stability with very small probabilities, use log-space:

```python
# Direct probability (may underflow for long sequences)
pgen = graph.walk_probability(encoded)
print(f"P = {pgen}")  # Might be 0.0 due to underflow

# Log probability (numerically stable)
log_pgen = graph.walk_probability(encoded, use_log=True)
print(f"log P = {log_pgen:.2f}")
```

### Mathematical Relationship

\[
\log P = \log P(\text{start}) + \sum_{i=1}^{n-1} \log P(\text{edge}_i)
\]

### When to Use Log Probability

- Comparing many sequences
- Working with long sequences
- Performing arithmetic on probabilities
- Avoiding numerical underflow (P < 10^{-300})

## Gene-Constrained Probability

When V/J genes are annotated, edges also carry gene information:

```python
# Edge has gene distribution
edge_data = graph.graph['C_1']['A_2']
print(edge_data.keys())
# dict_keys(['weight', 'gene_weight', 'V', 'J'])
```

### Gene-Weighted Walks

The `genomic_random_walk` uses gene weights to constrain generation:

```python
# Generate sequence consistent with gene usage
walk, v_gene, j_gene = graph.genomic_random_walk()
```

## Probability of Zero

A sequence has probability 0 if:

1. **Missing node**: A pattern was never observed
2. **Missing edge**: A transition was never observed

```python
# Check why probability is zero
sequence = "CASSXYZABC"  # Contains rare pattern XYZ
encoded = AAPLZGraph.encode_sequence(sequence)

for i, node in enumerate(encoded):
    if not graph.graph.has_node(node):
        print(f"Missing node: {node}")

for i in range(len(encoded) - 1):
    if not graph.graph.has_edge(encoded[i], encoded[i+1]):
        print(f"Missing edge: {encoded[i]} -> {encoded[i+1]}")
```

## Probability Interpretation

### Absolute Probability

The raw Pgen value indicates how likely this exact sequence is:

- **High Pgen** (e.g., 10^-8): Common sequence patterns
- **Low Pgen** (e.g., 10^-20): Rare sequence patterns
- **Zero Pgen**: Contains unobserved patterns

### Relative Probability

More useful for comparing sequences:

```python
sequences = ['CASSLEPSGGTDTQYF', 'CASSLGQGSTEAFF', 'CASSXYZRARESEQ']

probs = []
for seq in sequences:
    try:
        enc = AAPLZGraph.encode_sequence(seq)
        log_p = graph.walk_probability(enc, use_log=True)
        probs.append((seq, log_p))
    except:
        probs.append((seq, float('-inf')))

# Sort by probability
probs.sort(key=lambda x: x[1], reverse=True)
for seq, log_p in probs:
    print(f"{seq}: log P = {log_p:.2f}")
```

## Connection to Other Metrics

### Perplexity

Perplexity is derived from probability:

\[
\text{Perplexity} = P^{-1/n}
\]

Where n is the sequence length. Lower perplexity means the sequence "fits" the model better.

### LZCentrality

LZCentrality combines probability with graph structure:

```python
from LZGraphs import LZCentrality

centrality = LZCentrality(graph, sequence)
```

### Entropy

Graph entropy relates to the distribution of probabilities across all possible paths.

## Practical Example

Compare sequences by their generation probability:

```python
import pandas as pd
from LZGraphs import AAPLZGraph

# Build graph
data = pd.read_csv("repertoire.csv")
graph = AAPLZGraph(data, verbose=False)

# Analyze sequences
test_sequences = [
    "CASSLEPSGGTDTQYF",
    "CASSLGQGSTEAFF",
    "CASSELPSGGTDTQYF",  # Slight variant
]

results = []
for seq in test_sequences:
    enc = AAPLZGraph.encode_sequence(seq)
    log_pgen = graph.walk_probability(enc, use_log=True)
    results.append({
        'sequence': seq,
        'length': len(seq),
        'log_pgen': log_pgen,
        'pgen': 10**log_pgen if log_pgen > -300 else 0
    })

df = pd.DataFrame(results)
print(df.sort_values('log_pgen', ascending=False))
```

## Next Steps

- [LZ76 Algorithm](lz76-algorithm.md) - How sequences become walks
- [Graph Types](graph-types.md) - Different encoding schemes
- [Tutorials: Sequence Analysis](../tutorials/sequence-analysis.md) - Apply to real data
