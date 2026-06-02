---
tags:
  - FlashBackGraph
---

# FlashBack Exact Model

This page explains why `FlashBackGraph` can compute diversity, entropy, Hill
numbers, and generation probability **exactly**, and how that differs from the
estimated quantities in the [`LZGraph`](graph-types.md) family. No heavy math is
required; the key idea is structural.

## A Markovian graph over FlashBack tokens

The FlashBack decomposition recursively peels matching runs of characters from
both ends of a sentinel-wrapped sequence, producing a sequence of tokens. Build
a graph whose nodes are those tokens and whose edges are the transitions
between consecutive tokens, with edge weights estimated from your data.

The important property is that this graph is **strictly Markovian**: the
probability of the next token depends only on the current token, not on the
whole history. That makes the probability of a full sequence a simple product
of edge probabilities along its path.

## Why that makes analytics exact

When a distribution factorises over the edges of a directed acyclic graph, you
can compute sums over *all* sequences the graph can produce without ever
enumerating them, by a single sweep in topological order. This is **forward
dynamic programming**: each node accumulates a quantity from its predecessors,
and one pass yields the total.

Because of this, the following are computed exactly, with no sampling:

- **Generation probability** (`pgen`) of any sequence: a product along its path.
- **Path count**: how many distinct sequences the graph can produce.
- **Shannon entropy and effective diversity**: an expectation over the full
  distribution, summed by the same forward pass.
- **Hill numbers** `D(q)` and the diversity profile.

## Exact versus estimated

The `LZGraph` family uses a coarsened LZ76 dictionary that is not strictly
Markovian in the same way, so some quantities (notably constrained simulation
and certain diversity measures) are obtained by sampling sequences and
aggregating. Sampling introduces variance: a different random seed gives a
slightly different answer, and rare structure can be missed.

`FlashBackGraph` trades the `LZGraph` family's gene awareness and broad feature
set for this exactness. Use it when the number itself must be reproducible and
unbiased, for example as a reported statistic or a test input. Use `LZGraph`
when you need V/J gene modeling, ML features, or occupancy and sharing
predictions.

## The anomaly score (SCALE)

The same exact `pgen` underlies [SCALE](../tutorials/flashback-anomaly.md), the
self-calibrated anomaly score. Raw `-log pgen` flags anomalies but grows with
sequence length; SCALE calibrates against the graph's own simulated output
(per-length median and IQR of `-log pgen`) so the score is length-invariant and
comparable across sequences.

## See also

- [FlashBack: Exact Diversity](../tutorials/flashback-diversity.md)
- [FlashBack: Anomaly Detection](../tutorials/flashback-anomaly.md)
- [FlashBackGraph API](../api/flashback-graph.md)
- [Graph Variants](graph-types.md) for the two-family overview
