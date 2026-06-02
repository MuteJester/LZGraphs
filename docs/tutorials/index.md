# Learn by Family

In-depth, step-by-step tutorials. They are organized by graph family, so pick
the track for the graph you chose in
[Which Graph Should I Use?](../getting-started/choose-your-graph.md). The two
tracks are independent; you can follow either or both.

---

## LZGraph track

The general-purpose family: gene-aware, scalable, ML-ready.

<div class="grid" markdown>

<div class="card" markdown>
### 1. [Graph Construction](graph-construction.md)
**Beginner · 15 min**

Build AAP, NDP, and Naive graph variants from your data, with gene annotations
and abundance weighting.
</div>

<div class="card" markdown>
### 2. [Sequence Analysis](sequence-analysis.md)
**Beginner · 20 min**

Score sequences with `pgen`, explore graph structure, and simulate new
sequences.
</div>

<div class="card" markdown>
### 3. [Diversity Metrics](diversity-metrics.md)
**Intermediate · 15 min**

Measure repertoire complexity with k-diversity, Hill numbers, and occupancy
models.
</div>

</div>

---

## FlashBackGraph track

The Markovian family: exact, sampling-free analytics and anomaly scoring.

<div class="grid" markdown>

<div class="card" markdown>
### 1. [Exact Diversity](flashback-diversity.md)
Compute Hill numbers and effective diversity exactly via forward dynamic
programming, and understand why "exact" matters.
</div>

<div class="card" markdown>
### 2. [Anomaly Detection](flashback-anomaly.md)
Score sequences for surprise with SCALE, the self-calibrated anomaly score,
and interpret the result.
</div>

<div class="card" markdown>
### 3. [Personalization & Algebra](flashback-algebra.md)
Bayesian posterior updates, leave-donor-out construction, and graph algebra
(union, intersection, difference).
</div>

</div>

---

## Prerequisites

- [Installed LZGraphs](../getting-started/installation.md)
- Basic Python knowledge
- Sample data, or use the [example datasets](../examples/index.md)

## Sample data

The LZGraph-track tutorials use example data included with LZGraphs:

```python
import csv

with open("examples/data/ExampleData1.csv") as f:
    sequences = [row['cdr3_rearrangement'] for row in csv.DictReader(f)]
```

## Next steps

- [Concepts](../concepts/index.md) for deeper understanding
- [How-To Guides](../how-to/index.md) for specific tasks
- [Examples](../examples/index.md) for complete notebooks
