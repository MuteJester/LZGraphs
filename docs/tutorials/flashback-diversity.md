---
tags:
  - FlashBackGraph
---

# FlashBack: Exact Diversity

**Applies to: FlashBackGraph**

## What you'll do

Build a FlashBackGraph and read its diversity off directly: effective
diversity, Hill numbers across a range of orders, and the full diversity
profile. You will also see why "exact" is more than a technicality here.

## The biological question

How diverse is this repertoire? A diverse repertoire spreads its probability
mass across many distinct sequences; a focused or clonally expanded repertoire
concentrates mass on a few. Diversity is usually summarised with Hill numbers
`D(q)`: `D(0)` counts the effective richness, `D(1)` is the exponential of
Shannon entropy, and `D(2)` is the inverse Simpson index that weights common
clonotypes more heavily.

## Build and measure

```python
from LZGraphs import FlashBackGraph

sequences = ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF',
             'CASSLAPGATNEKLFF', 'CASSQETQYF']
fb = FlashBackGraph(sequences)

fb.effective_diversity()      # exp(Shannon entropy) = Hill D(1)
fb.hill_number(2)             # inverse Simpson, Hill D(2)
fb.hill_numbers([0, 1, 2])    # several orders at once -> numpy array
```

For a full curve to plot a diversity profile:

```python
curve = fb.hill_curve()       # dict with 'orders' and 'values'
curve['orders']               # e.g. [0, 0.25, 0.5, ..., 10]
curve['values']               # Hill number at each order
```

And the underlying breakdown:

```python
profile = fb.diversity_profile()
profile['effective_diversity']
profile['entropy_nats']
profile['entropy_bits']
profile['uniformity']         # how evenly mass is spread (0 to 1)
```

## Why "exact" matters

For the `LZGraph` family, some diversity quantities are estimated, often by
simulating sequences and aggregating. That introduces sampling variance: run it
again with a different seed and the number moves a little.

`FlashBackGraph` is strictly Markovian, so the entire sequence distribution
factorises over the graph's edges. That lets it sum over all paths exactly by
forward dynamic programming. The consequences for you:

- **Reproducible**: the same graph always returns the same diversity. No seed,
  no run-to-run noise.
- **No truncation bias**: nothing is dropped because it was not sampled, so
  rare-but-present structure is counted.
- **Fast**: one pass over the graph, not thousands of simulations.

This is exactly what you want when diversity is the headline number in a figure
or a statistical test, where sampling noise would otherwise be a confound.

## Interpreting the result

- `D(0) >= D(1) >= D(2)` always holds; the gap between them tells you how uneven
  the repertoire is. Equal values mean a flat, even distribution.
- A low `uniformity` with a high `D(0)` means many sequences exist but a few
  dominate the mass.

## Next steps

- [FlashBack: Anomaly Detection](flashback-anomaly.md)
- [FlashBack Exact Model](../concepts/flashback-model.md) for the math behind
  the exact computation
- [FlashBackGraph API](../api/flashback-graph.md)
