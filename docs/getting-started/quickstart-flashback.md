---
tags:
  - FlashBackGraph
---

# FlashBackGraph Quickstart

**Applies to: FlashBackGraph**

In five minutes you will build a FlashBackGraph from a list of CDR3 sequences,
read an exact diversity number off it, and score a sequence for how surprising
it is. FlashBackGraph is the family to use when you want exact, sampling-free
analytics or anomaly detection. If you also need V/J gene modeling or ML
features, see the [LZGraph Quickstart](quickstart-lzgraph.md) instead.

## Step 1: Build a graph

```python
from LZGraphs import FlashBackGraph

sequences = ['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF', 'CASSLAPGATNEKLFF']
fb = FlashBackGraph(sequences)
```

You can also build from a file with constant memory:

```python
fb = FlashBackGraph.from_file('repertoire.txt')  # one sequence per line, or sequence<TAB>abundance
```

## Step 2: Measure diversity, exactly

Because the graph is strictly Markovian, these numbers are computed exactly by
forward dynamic programming. There is no sampling and no variance to worry
about: the same graph always gives the same answer.

```python
fb.effective_diversity()   # exp(Shannon entropy), the same as Hill D(1)
fb.hill_number(2)          # inverse Simpson diversity
fb.hill_numbers([0, 1, 2]) # several orders at once (numpy array)
```

Biologically, a higher effective diversity means the repertoire spreads its
probability mass over more distinct sequences; a low number means a few
clonotypes dominate.

## Step 3: Score a sequence's generation probability

```python
log_p = fb.pgen('CASSLEPSGGTDTQYF')      # exact log probability under the model
```

## Step 4: Detect anomalies with SCALE

SCALE answers "how surprising is this sequence given the repertoire?" You
calibrate once against the graph, then score (higher = more anomalous).

```python
cal = fb.calibrate_scale(seed=0)              # self-calibrate (simulate from the graph)
fb.scale_score('CASSLEPSGGTDTQYF', cal)        # typical -> near 0
fb.scale_score('KKKKWWWWPPPP', cal)            # anomalous -> large positive
```

See the [anomaly-detection tutorial](../tutorials/flashback-anomaly.md) for
choosing a flag threshold.

## Step 5: Simulate new sequences

```python
sim = fb.simulate(1000, seed=42)
list(sim)[:5]         # the generated sequences
```

## Complete example

```python
from LZGraphs import FlashBackGraph

fb = FlashBackGraph(['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF', 'CASSLAPGATNEKLFF'])

print(f"Effective diversity: {fb.effective_diversity():.2f}")
print(f"Inverse Simpson D(2): {fb.hill_number(2):.2f}")
print(f"log pgen: {fb.pgen('CASSLEPSGGTDTQYF'):.3f}")
cal = fb.calibrate_scale(seed=0)
print(f"SCALE: {fb.scale_score('CASSLEPSGGTDTQYF', cal):.3f}")
```

## What's next?

- [FlashBack: Exact Diversity](../tutorials/flashback-diversity.md)
- [FlashBack: Anomaly Detection](../tutorials/flashback-anomaly.md)
- [FlashBack: Personalization & Algebra](../tutorials/flashback-algebra.md)
- [FlashBackGraph API reference](../api/flashback-graph.md)
