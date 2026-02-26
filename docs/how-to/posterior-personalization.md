# Personalize Graphs with Bayesian Posteriors

Learn how to adapt a population-level LZGraph to an individual's repertoire using `get_posterior()`.

## Quick Reference

```python
posterior = population_graph.get_posterior(individual_sequences, kappa=1.0)
```

## When to Use

**Bayesian posterior personalization** is useful when you have:

1. A **foundation graph** built from a large population or reference dataset
2. A **smaller individual repertoire** that you want to analyze in the context of the foundation

The posterior graph blends the population's structural knowledge with the individual's observed transitions, controlled by the `kappa` parameter.

**Typical use cases:**

- Personalizing a healthy-population model to a specific patient
- Comparing how different individuals diverge from a shared baseline
- Regularizing small repertoire samples with population-level priors
- Building patient-specific generative models for downstream simulation

## Basic Usage

### Build a Prior Graph

Start with a large population-level graph:

```python
from LZGraphs import AAPLZGraph

# Build from a large population dataset
population_sequences = [...]  # thousands of sequences
prior = AAPLZGraph(population_sequences, verbose=True)
```

### Create a Posterior

Personalize the prior with an individual's observed sequences:

```python
individual_sequences = [...]  # this patient's repertoire

# kappa=1.0: balanced prior/data influence
posterior = prior.get_posterior(individual_sequences, kappa=1.0)
```

The returned `posterior` is a full `AAPLZGraph` — it supports every method the prior does.

## With Abundance Weighting

If your individual data includes clonotype counts, pass them as abundances. Expanded clones will have proportionally more influence on the posterior:

```python
sequences  = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]
abundances = [150, 42, ...]

posterior = prior.get_posterior(sequences, abundances=abundances, kappa=100.0)
```

!!! tip "Auto-detection from DataFrames"
    If you pass a DataFrame, `get_posterior` automatically detects columns named `abundance`, `templates`, or `count` for abundance weighting.

## Understanding Kappa

The `kappa` parameter controls how strongly the prior influences the posterior:

| Kappa | Effect | When to use |
|-------|--------|-------------|
| 0.01 | Posterior is essentially the individual's MLE | Full trust in individual data |
| 1.0 | Prior and data have equal influence per count | Default, balanced |
| 10 | Prior needs ~10 individual counts to shift noticeably | Moderate regularization |
| 100 | Prior dominates; individual adds subtle adjustments | Strong regularization |
| 10,000+ | Posterior is nearly identical to the prior | Minimal personalization |

### Exploring Kappa Sensitivity

Test how sensitive your conclusions are to the choice of kappa:

```python
from LZGraphs import compare_repertoires

kappas = [0.1, 1.0, 10.0, 100.0, 1000.0]

for k in kappas:
    post = prior.get_posterior(individual_sequences, kappa=k)

    # Compare posterior to prior
    metrics = compare_repertoires(prior, post)
    print(f"kappa={k:>7.1f}  JSD={metrics['js_divergence']:.4f}  "
          f"transition_JSD={metrics['transition_jsd']:.4f}")
```

As kappa increases, the posterior converges to the prior (JSD → 0).

## Using the Posterior

The posterior graph supports all standard LZGraph operations:

### Probability Queries

```python
# How likely is this sequence under the personalized model?
log_p = posterior.walk_probability("CASSLEPSGGTDTQYF", use_log=True)
print(f"log P(gen) = {log_p:.2f}")
```

### Sequence Simulation

```python
# Generate sequences from the personalized model
simulated = posterior.simulate(10000, seed=42)
print(f"Generated {len(simulated)} sequences")
```

### Metrics

```python
from LZGraphs import graph_entropy, transition_predictability

h = graph_entropy(posterior)
tp = transition_predictability(posterior)
print(f"Entropy: {h:.2f}, Predictability: {tp:.3f}")
```

### Analytical Distributions

```python
# Pgen distribution of the personalized model
dist = posterior.lzpgen_analytical_distribution()
print(f"Mean log-Pgen: {dist.mean():.2f}")
```

## Comparing Prior vs Posterior

Use `compare_repertoires` to quantify how the individual's data changed the model:

```python
from LZGraphs import compare_repertoires

metrics = compare_repertoires(prior, posterior)

print(f"JSD (node-level):       {metrics['js_divergence']:.4f}")
print(f"JSD (transition-level): {metrics['transition_jsd']:.4f}")
print(f"Shared nodes:           {metrics['shared_nodes']}")
print(f"Jaccard (nodes):        {metrics['jaccard_nodes']:.3f}")
```

## Real-World Workflow

A complete workflow for patient-level analysis:

```python
from LZGraphs import AAPLZGraph, compare_repertoires

# 1. Build foundation from healthy population
healthy_sequences = [...]  # large healthy cohort
foundation = AAPLZGraph(healthy_sequences, verbose=True)

# 2. Save for reuse
foundation.save("foundation.json")

# 3. Personalize per patient
patients = {
    "patient_A": (seqs_a, abundances_a),
    "patient_B": (seqs_b, abundances_b),
    "patient_C": (seqs_c, abundances_c),
}

results = {}
for name, (seqs, abd) in patients.items():
    post = foundation.get_posterior(seqs, abundances=abd, kappa=100.0)
    metrics = compare_repertoires(foundation, post)
    results[name] = {
        "posterior": post,
        "jsd": metrics["js_divergence"],
        "transition_jsd": metrics["transition_jsd"],
    }
    print(f"{name}: JSD={metrics['js_divergence']:.4f}")

# 4. Compare patients to each other
for (n1, r1), (n2, r2) in combinations(results.items(), 2):
    m = compare_repertoires(r1["posterior"], r2["posterior"])
    print(f"{n1} vs {n2}: JSD={m['js_divergence']:.4f}")
```

## What Gets Updated

The posterior updates three probability components:

| Component | Prior source | Individual source | Update rule |
|-----------|-------------|-------------------|-------------|
| **Edge weights** | `P_prior(b\|a)` | Edge traversal counts | Dirichlet-Multinomial |
| **Initial states** | Prior initial probs | Starting node counts | Dirichlet-Multinomial |
| **Stop probabilities** | `P(stop\|node)` | Terminal vs. continuation counts | Beta-Binomial (2-category Dirichlet) |

Novel edges and nodes from the individual (not in the prior) are added with no prior penalty — kappa does not suppress genuinely new structure.

## Next Steps

- [Concepts: Probability Model](../concepts/probability-model.md#bayesian-posterior-graphs) — Mathematical details
- [Compare Repertoires](repertoire-comparison.md) — Full repertoire comparison toolkit
- [API: AAPLZGraph](../api/aaplzgraph.md#get_posterior) — Method reference
