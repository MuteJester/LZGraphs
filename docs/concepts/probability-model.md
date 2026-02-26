# Probability Model

LZGraphs calculates sequence generation probability (Pgen) using edge-weighted random walks. This page explains the mathematical foundation.

## The Core Idea

A sequence's probability is the product of:

1. **Initial probability**: Likelihood of starting with the first pattern
2. **Transition probabilities**: Likelihood of each pattern-to-pattern transition
3. **Stop probability**: Likelihood of terminating at the last node

\[
P(\text{sequence}) = P(\text{start}) \times \prod_{i=1}^{n-1} P(\text{node}_{i+1} | \text{node}_i) \times P(\text{stop} | \text{node}_n)
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
p_start = graph.initial_state_probabilities['C_1']
print(f"P(start with C_1) = {p_start:.4f}")
```

### Step 3: Multiply Edge Weights

For each transition, get the edge weight from the `EdgeData` object:

```python
# Transition probabilities (edge weights)
p_C1_A2 = graph.graph['C_1']['A_2']['data'].weight
p_A2_S3 = graph.graph['A_2']['S_3']['data'].weight
p_S3_SL5 = graph.graph['S_3']['SL_5']['data'].weight
p_SL5_E6 = graph.graph['SL_5']['E_6']['data'].weight

# Stop probability at the last node
p_stop = graph._stop_probability_cache.get('E_6', 0)

# Total probability
pgen = p_start * p_C1_A2 * p_A2_S3 * p_S3_SL5 * p_SL5_E6 * p_stop
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
total_weight = sum(graph.graph[node][s]['data'].weight for s in successors)
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
\log P = \log P(\text{start}) + \sum_{i=1}^{n-1} \log P(\text{edge}_i) + \log P(\text{stop} | \text{node}_n)
\]

### When to Use Log Probability

- Comparing many sequences
- Working with long sequences
- Performing arithmetic on probabilities
- Avoiding numerical underflow (P < 10^{-300})

## Gene-Constrained Probability

When V/J genes are annotated, edges carry gene information via the `EdgeData` object:

```python
# Access edge data
edge = graph.graph['C_1']['A_2']['data']
print(f"Weight: {edge.weight}")
print(f"V genes: {edge.v_genes}")   # dict of {gene_name: count}
print(f"J genes: {edge.j_genes}")   # dict of {gene_name: count}
```

### Gene-Weighted Walks

The `genomic_random_walk` uses gene weights to constrain generation:

```python
# Generate sequence consistent with gene usage
walk, v_gene, j_gene = graph.genomic_random_walk()
```

## Bayesian Posterior Graphs

LZGraphs supports **Bayesian posterior personalization** — updating a population-level graph with an individual's observed sequences using Dirichlet-Multinomial conjugacy.

### The Core Idea

A graph built from a large population captures general patterns of repertoire structure. But a specific individual's repertoire may differ. `get_posterior()` creates a new graph that blends the population-level "prior" with the individual's observed data:

```python
# Build a population-level prior graph
prior = AAPLZGraph(population_sequences, verbose=False)

# Personalize to an individual
posterior = prior.get_posterior(individual_sequences, kappa=1.0)

# The posterior is a full graph — use it like any other
log_p = posterior.walk_probability("CASSLEPSGGTDTQYF", use_log=True)
simulated = posterior.simulate(1000, seed=42)
```

### Mathematical Framework

For each node \(a\) with outgoing edges to successors \(b_1, b_2, \ldots, b_K\):

**Prior** (from population graph):

\[
\text{Dir}(\kappa \cdot \pi_1^f, \; \kappa \cdot \pi_2^f, \; \ldots, \; \kappa \cdot \pi_K^f)
\]

where \(\pi_i^f = P_{\text{prior}}(b_i \mid a)\) and \(\kappa\) is the concentration parameter.

**Observed counts** from the individual: \(c_1, c_2, \ldots, c_K\)

**Posterior** (Dirichlet update):

\[
P_{\text{post}}(b_i \mid a) = \frac{\kappa \cdot \pi_i^f + c_i}{\kappa + n_a}
\]

where \(n_a = \sum_i c_i\) is the total number of outgoing traversals observed from node \(a\).

The same conjugate update applies to **initial state probabilities** and **stop probabilities**.

### The Kappa Parameter

\(\kappa\) controls how much the posterior trusts the prior versus the individual's data:

| \(\kappa\) | Behavior | Use Case |
|------------|----------|----------|
| \(\to 0\) | Posterior ≈ MLE from individual data alone | Maximize fit to individual |
| 1.0 (default) | Equal weight per count | Balanced personalization |
| 100 | Prior dominates until ~100 individual counts accumulate | Trust population, minor adaptation |
| \(\to \infty\) | Posterior ≈ prior (unchanged) | Use population model as-is |

### With Abundance Weighting

When sequences carry abundance counts (e.g., clonotype expansion), pass them to weight each observation:

```python
posterior = prior.get_posterior(
    sequences,
    abundances=[150, 42, 7, ...],
    kappa=100.0
)
```

Highly expanded clones contribute proportionally more to the posterior update.

### What Gets Updated

The posterior graph updates three components:

1. **Edge weights**: Transition probabilities blended between prior and individual
2. **Initial state probabilities**: Which starting patterns are likely
3. **Stop probabilities**: Where sequences tend to terminate

Novel edges and nodes observed in the individual (but absent in the prior) are incorporated automatically with no prior penalty.

!!! tip "Practical guide"
    See [How-To: Personalize Graphs](../how-to/posterior-personalization.md) for step-by-step workflows including kappa sensitivity analysis and prior-vs-posterior comparison.

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
        log_p = graph.walk_probability(seq, use_log=True)
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

### lz_centrality

lz_centrality combines probability with graph structure:

```python
from LZGraphs import lz_centrality

centrality = lz_centrality(graph, sequence)
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
    log_pgen = graph.walk_probability(seq, use_log=True)
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
