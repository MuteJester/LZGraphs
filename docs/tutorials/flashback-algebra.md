---
tags:
  - FlashBackGraph
---

# FlashBack: Personalization & Algebra

**Applies to: FlashBackGraph**

## What you'll do

Adapt a population model to one individual, remove a donor's contribution from
a graph, and combine or contrast cohorts with graph algebra. All of these
return a new `FlashBackGraph` you can then score, simulate, or measure.

## The biological question

Repertoire questions are often relative. How does this patient differ from a
healthy population baseline? What does a cohort look like with one donor left
out (for leave-one-out validation)? What structure do two cohorts share, and
what is unique to each? These map directly onto Bayesian updating and set
operations on graphs.

## Personalize a population model

`posterior` treats the current graph as a Dirichlet prior and updates it toward
an individual's sequences. `kappa` sets how much the prior is trusted: `kappa=0`
is pure individual, large `kappa` stays close to the population.

```python
from LZGraphs import FlashBackGraph

cohort = FlashBackGraph(['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF', 'CASSLAPGATNEKLFF'])
patient = ['CASSLEPSGGTDTQYF', 'CASSQETQYF']

personal = cohort.posterior(patient, kappa=1.0)   # a new, personalized graph
```

## Leave a donor out

`without` removes the contribution of given sequences, rebuilding the affected
edge weights. This gives you a leave-donor-out graph in seconds, without
rebuilding from the full source data.

```python
reduced = cohort.without(['CASSDTSGGTDTQYF'])
```

## Combine and contrast cohorts

Graph algebra operates on edge counts and returns a new graph:

```python
other = FlashBackGraph(['CASSQETQYF', 'CASSLAPGATNEKLFF'])

combined = cohort.union(other)         # sum edge counts (pooled cohort)
shared   = cohort.intersection(other)  # structure common to both
only_a   = cohort.difference(other)    # structure unique to `cohort`
```

## Interpreting the result

- A `posterior` graph scores the individual's own sequences higher than the
  bare population prior would, while still borrowing strength from the
  population for sequences the individual did not show.
- `intersection` retaining many nodes indicates two cohorts share a lot of
  structure; a small intersection indicates divergent repertoires.
- Every result is a full `FlashBackGraph`, so you can immediately call
  `effective_diversity()`, `pgen()`, or `scale_score()` on it.

## Next steps

- [FlashBack: Exact Diversity](flashback-diversity.md)
- [FlashBack: Anomaly Detection](flashback-anomaly.md)
- [FlashBackGraph API](../api/flashback-graph.md)
