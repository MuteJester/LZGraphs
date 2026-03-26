# Examples

Practical examples you can copy-paste and run. Each example is self-contained and uses the sample datasets included in the repository.

---

## Jupyter Notebooks

Full interactive notebooks are available on GitHub:

<div class="grid" markdown>

<div class="card" markdown>
### 1. [Getting Started](https://github.com/MuteJester/LZGraphs/blob/master/examples/01_Getting_Started.ipynb)
**Beginner** — Build a graph, score sequences, simulate new ones.
</div>

<div class="card" markdown>
### 2. [Analytics and Diversity](https://github.com/MuteJester/LZGraphs/blob/master/examples/02_Analytics_and_Diversity.ipynb)
**Intermediate** — Hill numbers, richness predictions, occupancy models.
</div>

<div class="card" markdown>
### 3. [Advanced Usage](https://github.com/MuteJester/LZGraphs/blob/master/examples/03_Advanced_Usage.ipynb)
**Advanced** — Posterior personalization, feature alignment, PGEN distributions.
</div>

</div>

To run them locally:

```bash
git clone https://github.com/MuteJester/LZGraphs.git
cd LZGraphs
pip install . && pip install jupyter
cd examples && jupyter notebook
```

---

## Example 1: Basic repertoire analysis

A complete workflow from loading data to measuring diversity.

```python
import csv
from LZGraphs import LZGraph

# Load sequences from a CSV file
seqs, v_genes, j_genes = [], [], []
with open("examples/ExampleData3.csv") as f:
    for row in csv.DictReader(f):
        seqs.append(row['cdr3_amino_acid'])
        v_genes.append(row.get('v_call', ''))
        j_genes.append(row.get('j_call', ''))

# Build the graph
graph = LZGraph(seqs, variant='aap')
print(graph)
# LZGraph(variant='aap', nodes=1721, edges=9644)

# Basic statistics
print(f"Sequences:  {graph.n_sequences}")
print(f"Lengths:    {sorted(graph.length_distribution.keys())}")
print(f"Density:    {graph.density:.4f}")

# Score a known sequence
log_p = graph.lzpgen("CASSLEPSGGTDTQYF")
print(f"\nlog P(CASSLEPSGGTDTQYF) = {log_p:.2f}")

# Diversity
print(f"\nEffective diversity D(1): {graph.effective_diversity():,.0f}")
print(f"Inverse Simpson D(2):     {graph.hill_number(2):,.0f}")

# Predicted richness at different depths
for depth in [1_000, 10_000, 100_000]:
    r = graph.predicted_richness(depth)
    print(f"At depth {depth:>7,d}: {r:>8,.0f} unique sequences")
```

---

## Example 2: Comparing two repertoires

Build graphs from two samples and measure their divergence.

```python
import csv
from LZGraphs import LZGraph, jensen_shannon_divergence

# Split data into two "repertoires"
with open("examples/ExampleData3.csv") as f:
    all_seqs = [row['cdr3_amino_acid'] for row in csv.DictReader(f)]

seqs_a = all_seqs[:2500]
seqs_b = all_seqs[2500:]

graph_a = LZGraph(seqs_a, variant='aap')
graph_b = LZGraph(seqs_b, variant='aap')

# Jensen-Shannon Divergence
jsd = jensen_shannon_divergence(graph_a, graph_b)
print(f"JSD: {jsd:.4f}")

# Compare diversity profiles
for alpha in [0, 1, 2]:
    da = graph_a.hill_number(alpha)
    db = graph_b.hill_number(alpha)
    print(f"D({alpha}):  A = {da:,.0f}   B = {db:,.0f}")

# Cross-score: how probable are A's sequences under B's model?
import numpy as np
cross_scores = graph_b.lzpgen(seqs_a[:100])
print(f"\nMean log P(A seqs | B model): {np.mean(cross_scores):.2f}")
```

---

## Example 3: Sequence generation and evaluation

Generate synthetic sequences and verify they match the repertoire's patterns.

!!! note "Setup"
    Examples 3-6 assume `seqs` and `graph` from Example 1 above.

```python
from collections import Counter
import numpy as np

# Generate 10,000 sequences
result = graph.simulate(10_000, seed=42)

# Length distribution of generated sequences
gen_lengths = Counter(len(s) for s in result.sequences)
print("Generated length distribution:")
for length in sorted(gen_lengths):
    bar = "#" * (gen_lengths[length] // 100)
    print(f"  {length:2d}: {gen_lengths[length]:5d} {bar}")

# Novelty: how many are not in the training data?
train_set = set(seqs)
novel = [s for s in set(result.sequences) if s not in train_set]
print(f"\nUnique generated: {len(set(result.sequences))}")
print(f"Novel (not in training): {len(novel)}")
print(f"Novelty rate: {len(novel) / len(set(result.sequences)):.1%}")

# Log-probability statistics
print(f"\nSimulated log P:  mean={result.log_probs.mean():.2f}, "
      f"std={result.log_probs.std():.2f}")
```

---

## Example 4: Graph algebra

Combine and decompose repertoires using set operations.

```python
from LZGraphs import LZGraph

# Split into two "conditions" for demonstration
healthy_seqs = seqs[:2500]
disease_seqs = seqs[2500:]

healthy = LZGraph(healthy_seqs, variant='aap')
disease = LZGraph(disease_seqs, variant='aap')

# Union: combined repertoire
combined = healthy | disease
print(f"Healthy: {healthy.n_edges} edges")
print(f"Disease: {disease.n_edges} edges")
print(f"Combined: {combined.n_edges} edges")

# Intersection: shared structure
shared = healthy & disease
print(f"Shared: {shared.n_edges} edges")

# Difference: what's unique to disease
disease_only = disease - healthy
print(f"Disease-specific: {disease_only.n_edges} edges")

# Score a candidate against the disease-specific model
candidate = "CASSLGQAYEQYF"
print(f"\nlog P(candidate | disease-specific): {disease_only.lzpgen(candidate):.2f}")
print(f"log P(candidate | healthy):          {healthy.lzpgen(candidate):.2f}")
```

---

## Example 5: ML feature extraction

Build a classification pipeline using graph-derived features.

```python
import numpy as np
from LZGraphs import LZGraph

# Build a reference graph from all data
ref = LZGraph(seqs, variant='aap')
print(f"Feature dimension: {ref.n_nodes}")

# Split into "samples" of 500 seqs each for demonstration
samples = [seqs[i:i+500] for i in range(0, 5000, 500)]
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # first 5 "healthy", last 5 "disease"

# Extract features for each sample
def get_features(sample_seqs):
    g = LZGraph(sample_seqs, variant='aap')
    aligned = ref.feature_aligned(g)     # reference-aligned (n_nodes dim)
    stats = g.feature_stats()             # 15-dim summary
    mass = g.feature_mass_profile()       # 31-dim position profile
    return np.concatenate([aligned, stats, mass])

X = np.array([get_features(s) for s in samples])
y = np.array(labels)

print(f"Feature matrix: {X.shape}")

# Train a classifier (requires scikit-learn)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy: {scores.mean():.1%} +/- {scores.std():.1%}")
```

---

## Example 6: Bayesian personalization

Adapt a population model to an individual patient.

```python
from LZGraphs import LZGraph

# Population-level graph (using first 4000 sequences as "population")
population = LZGraph(seqs[:4000], variant='aap')

# One patient's sequences (last 1000 as "individual")
patient_seqs = seqs[4000:]

# Create personalized model with different prior strengths
for kappa in [1.0, 10.0, 100.0]:
    personal = population.posterior(patient_seqs, kappa=kappa)
    d1 = personal.effective_diversity()
    print(f"kappa={kappa:5.1f}  D(1)={d1:,.0f}  "
          f"nodes={personal.n_nodes}  edges={personal.n_edges}")
```

Small $\kappa$ → the patient's data dominates (personalized). Large $\kappa$ → the population prior dominates (conservative).

---

## Sample Data

The repository includes these datasets for testing and examples:

| File | Sequences | Content |
|------|:---------:|---------|
| `examples/ExampleData3.csv` | 5,000 | Amino acid CDR3 sequences with V/J gene annotations |

---

## See Also

- [Getting Started](../getting-started/quickstart.md) — installation and first graph
- [Tutorials](../tutorials/index.md) — step-by-step learning paths
- [API Reference](../api/index.md) — complete method documentation
