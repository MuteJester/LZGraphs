---
tags:
  - Construction
  - Genes
---

# Personalize Graphs with Bayesian Posteriors

Learn how to adapt a population-level LZGraph to an individual's repertoire using Bayesian posteriors.

## Quick Reference

```python
from LZGraphs import LZGraph

# Personalize a prior graph with new data
posterior = population_graph.posterior(individual_sequences, kappa=1.0)
```

## When to Use

**Bayesian posterior personalization** is useful when you have:

1. A **prior graph** built from a large population or reference dataset.
2. A **smaller individual repertoire** that you want to analyze in the context of that prior.

The posterior graph blends the population's structural knowledge with the individual's observed transitions, controlled by the `kappa` parameter.

**Typical use cases:**

- Regularizing small repertoire samples with population-level structural knowledge.
- Comparing how different individuals diverge from a shared healthy baseline.
- Building patient-specific generative models for downstream simulation.

## Basic Usage

### Build a Prior Graph

Start with a large population-level graph (the "prior"):

```python
from LZGraphs import LZGraph

# Build from a large population dataset
prior = LZGraph(population_sequences, variant='aap')
```

### Create a Posterior

Personalize the prior with an individual's observed sequences:

```python
individual_sequences = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF", ...]

# kappa=1.0: balanced prior/data influence
posterior = prior.posterior(individual_sequences, kappa=1.0)
```

The returned `posterior` is a full `LZGraph` — it supports every method the prior does.

## With Abundance Weighting

If your individual data includes clonotype counts, pass them as abundances. Expanded clones will have proportionally more influence on the posterior update:

```python
sequences  = ["CASSLEPSGGTDTQYF", "CASSDTSGGTDTQYF"]
abundances = [150, 42]

posterior = prior.posterior(sequences, abundances=abundances, kappa=10.0)
```

## Understanding Kappa

The `kappa` parameter controls the "strength" of the prior. It represents the number of virtual observations the prior contributes to each node.

| Kappa | Effect | When to use |
|-------|--------|-------------|
| 0.1 | Posterior is essentially the individual's data | Full trust in individual sample |
| 1.0 | Prior and data have equal influence per count | Default, balanced |
| 10 | Prior dominates until ~10 counts accumulate | Moderate regularization |
| 100+ | Prior dominates; individual adds minor adjustments | Strong regularization |

### Exploring Kappa Sensitivity

You can test how sensitive your personalized model is to the choice of kappa by measuring the divergence from the prior:

```python
from LZGraphs import jensen_shannon_divergence

kappas = [0.1, 1.0, 10.0, 100.0]

for k in kappas:
    post = prior.posterior(individual_sequences, kappa=k)
    jsd = jensen_shannon_divergence(prior, post)
    print(f"kappa={k:>7.1f}  JSD from prior: {jsd:.4f}")
```

## Using the Posterior

Once created, the posterior is used exactly like any other `LZGraph`.

### Probability and Simulation

```python
# How likely is this sequence under the personalized model?
log_p = posterior.lzpgen("CASSLEPSGGTDTQYF")

# Generate sequences from the personalized model
simulated = posterior.simulate(1000)
```

### Comparing Individuals

By personalizing the SAME prior for different patients, you can compare the patients in a shared structural context:

```python
# Personalize for Patient A and Patient B
post_a = prior.posterior(seqs_a, kappa=1.0)
post_b = prior.posterior(seqs_b, kappa=1.0)

# Compare the personalized models
dist = jensen_shannon_divergence(post_a, post_b)
print(f"Patient A vs B Divergence: {dist:.4f}")
```

## What Gets Updated

The posterior updates three probability components using Dirichlet-Multinomial conjugacy:

1. **Edge weights**: Transition probabilities are blended between the prior and the new data.
2. **Initial states**: The likelihood of starting with specific patterns is updated.
3. **Stop probabilities**: Where sequences tend to terminate is adjusted based on observed sequence lengths.

!!! note "Novel Structure"
    Novel edges and nodes found in the individual's data (but absent in the prior) are added to the posterior automatically. `kappa` only regularizes transitions that exist in the prior.

## Next Steps

- [Concepts: Probability Model](../concepts/probability-model.md#bayesian-posterior-updates) — Mathematical details
- [How-To: Compare Repertoires](repertoire-comparison.md) — Compare multiple personalized graphs
- [API Reference](../api/index.md) — Detailed method documentation
