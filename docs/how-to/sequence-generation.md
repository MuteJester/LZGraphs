# Generate Sequences

Learn how to generate new sequences that follow your repertoire's statistical patterns.

## Quick Reference

```python
# Basic generation
walk = graph.random_walk()

# With gene constraints
walk, v_gene, j_gene = graph.genomic_random_walk()

# Convert to sequence
sequence = ''.join([AAPLZGraph.clean_node(n) for n in walk])
```

## Basic Sequence Generation

### Random Walk

Generate a sequence following edge probabilities:

```python
from LZGraphs import AAPLZGraph
import pandas as pd

# Build graph
data = pd.read_csv("repertoire.csv")
graph = AAPLZGraph(data, verbose=False)

# Generate a random walk
walk = graph.random_walk()
print(f"Walk: {walk}")

# Convert to sequence
sequence = ''.join([AAPLZGraph.clean_node(node) for node in walk])
print(f"Sequence: {sequence}")
```

### Generate Multiple Sequences

```python
generated = []
for _ in range(100):
    walk = graph.random_walk()
    seq = ''.join([AAPLZGraph.clean_node(n) for n in walk])
    generated.append(seq)

print(f"Generated {len(generated)} sequences")
print(f"Example: {generated[0]}")
```

## Gene-Constrained Generation

### Using genomic_random_walk

Generate sequences consistent with V/J gene usage:

```python
# Generate with gene constraints
walk, v_gene, j_gene = graph.genomic_random_walk()

sequence = ''.join([AAPLZGraph.clean_node(n) for n in walk])
print(f"Sequence: {sequence}")
print(f"V gene: {v_gene}")
print(f"J gene: {j_gene}")
```

!!! info "Requirements"
    `genomic_random_walk()` requires that the graph was built with V and J gene columns.

### Generate with Specific Gene Frequencies

The generated sequences follow the marginal gene distribution:

```python
from collections import Counter

# Generate many sequences
results = []
for _ in range(1000):
    walk, v, j = graph.genomic_random_walk()
    results.append({'v_gene': v, 'j_gene': j})

# Check gene distribution
df = pd.DataFrame(results)
print("V gene distribution:")
print(df['v_gene'].value_counts(normalize=True).head())

# Compare to original
print("\nOriginal V gene distribution:")
print(graph.marginal_vgenes.head())
```

## Advanced Generation

### Generate Specific Lengths

Filter generated sequences by length:

```python
def generate_with_length(graph, target_length, max_attempts=1000):
    for _ in range(max_attempts):
        walk = graph.random_walk()
        seq = ''.join([AAPLZGraph.clean_node(n) for n in walk])
        if len(seq) == target_length:
            return seq, walk
    return None, None

# Generate 15-mer
seq, walk = generate_with_length(graph, 15)
if seq:
    print(f"Generated 15-mer: {seq}")
```

### Generate from Specific Start

Start from a specific initial state:

```python
# Check available initial states
print("Initial states:")
print(graph.initial_states)

# Note: random_walk starts from initial states by default
# The initial state is chosen based on observed frequencies
```

### Batch Generation with Statistics

```python
import pandas as pd
from tqdm import tqdm

def generate_repertoire(graph, n_sequences, use_genes=True):
    """Generate a synthetic repertoire."""
    results = []

    for _ in tqdm(range(n_sequences), desc="Generating"):
        if use_genes:
            walk, v, j = graph.genomic_random_walk()
        else:
            walk = graph.random_walk()
            v, j = None, None

        seq = ''.join([AAPLZGraph.clean_node(n) for n in walk])
        results.append({
            'sequence': seq,
            'length': len(seq),
            'v_gene': v,
            'j_gene': j
        })

    return pd.DataFrame(results)

# Generate 1000 sequences
synthetic = generate_repertoire(graph, 1000)
print(synthetic.describe())
```

## Evaluating Generated Sequences

### Check Probability

```python
# Generate a sequence
walk, v, j = graph.genomic_random_walk()
sequence = ''.join([AAPLZGraph.clean_node(n) for n in walk])

# Calculate its probability
encoded = AAPLZGraph.encode_sequence(sequence)
pgen = graph.walk_probability(encoded, use_log=True)

print(f"Generated: {sequence}")
print(f"log P(gen): {pgen:.2f}")
```

### Compare to Original Distribution

```python
# Original sequence lengths
original_lengths = pd.Series(graph.lengths)

# Generated sequence lengths
synthetic = generate_repertoire(graph, 1000)
generated_lengths = synthetic['length'].value_counts()

# Plot comparison
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
original_lengths.plot(kind='bar', alpha=0.5, label='Original', ax=ax)
generated_lengths.plot(kind='bar', alpha=0.5, label='Generated', ax=ax)
ax.legend()
ax.set_xlabel('Sequence Length')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('length_comparison.png')
```

## Use Cases

### Data Augmentation

```python
# Augment a small repertoire
small_data = pd.read_csv("small_repertoire.csv")  # 100 sequences
graph = AAPLZGraph(small_data)

# Generate more sequences
augmented = generate_repertoire(graph, 1000)

# Combine
combined = pd.concat([
    small_data[['cdr3_amino_acid', 'V', 'J']].rename(
        columns={'cdr3_amino_acid': 'sequence'}
    ),
    augmented[['sequence', 'v_gene', 'j_gene']].rename(
        columns={'v_gene': 'V', 'j_gene': 'J'}
    )
])
```

### Null Model Generation

```python
def generate_null_repertoire(graph, n_sequences):
    """Generate sequences for statistical testing."""
    return generate_repertoire(graph, n_sequences)

# Generate null distribution
null_seqs = generate_null_repertoire(graph, 10000)

# Test a specific sequence against null
test_seq = "CASSLEPSGGTDTQYF"
test_pgen = graph.walk_probability(
    AAPLZGraph.encode_sequence(test_seq),
    use_log=True
)

# Calculate p-value
null_pgens = []
for seq in null_seqs['sequence']:
    try:
        pgen = graph.walk_probability(
            AAPLZGraph.encode_sequence(seq),
            use_log=True
        )
        null_pgens.append(pgen)
    except:
        pass

p_value = sum(1 for p in null_pgens if p <= test_pgen) / len(null_pgens)
print(f"P-value: {p_value:.4f}")
```

## Troubleshooting

### "No gene data" Error

```python
try:
    walk, v, j = graph.genomic_random_walk()
except Exception as e:
    print("Gene data not available. Using random_walk instead.")
    walk = graph.random_walk()
```

### Empty Walks

```python
walk = graph.random_walk()
if not walk:
    print("Empty walk generated - check graph structure")
else:
    print(f"Walk length: {len(walk)}")
```

## Next Steps

- [Compare Repertoires](repertoire-comparison.md) - Compare generated vs. original
- [Tutorials: Sequence Analysis](../tutorials/sequence-analysis.md) - More analysis techniques
