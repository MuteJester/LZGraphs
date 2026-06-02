---
tags:
  - FlashBackGraph
---

# FlashBack: Anomaly Detection with SCALE

**Applies to: FlashBackGraph**

## What you'll do

Flag error or noise sequences in a repertoire with **SCALE**, the
self-calibrated FlashBack anomaly score: build a calibration on the graph
once, then score any sequence so that higher means more anomalous.

## The biological question

Given a model of a repertoire, is a new sequence typical of it or unusual? An
unusual sequence might be a sequencing or processing error, a contaminant from
another individual or condition, or a genuinely rare clonotype worth a closer
look.

A natural starting point is the generation probability: anomalous sequences
have low `pgen`, so high `-log pgen` looks like a flag. The catch is that
`-log pgen` grows with sequence length, so a raw cutoff flags long sequences
regardless of how typical they are. SCALE fixes this by calibrating against the
graph's own output.

## SCALE in two steps

**Step 1, calibrate (once).** `calibrate_scale` simulates sequences from the
graph and records, per length, the median and IQR of `-log pgen`. This is the
reusable calibration cache.

```python
from LZGraphs import FlashBackGraph

reference = FlashBackGraph(['CASSLEPSGGTDTQYF', 'CASSDTSGGTDTQYF',
                            'CASSLAPGATNEKLFF', 'CASSQETQYF'])

calibration = reference.calibrate_scale(n_sim=50_000, seed=0)
calibration.save('scale_calibration.json')   # reuse later without re-simulating
```

**Step 2, score.** `scale_score` returns the length-calibrated score
`(-log pgen(s) - median[len]) / IQR[len]`. Higher is more anomalous.

```python
reference.scale_score('CASSLEPSGGTDTQYF', calibration)   # a sequence the model knows
reference.scale_score('KKKKWWWWPPPP', calibration)        # a sequence it does not
```

The first is near zero (typical); the second is large and positive (anomalous).
Score many at once by passing a list:

```python
seqs = ['CASSLEPSGGTDTQYF', 'CASSQETQYF', 'KKKKWWWWPPPP']
scores = reference.scale_score(seqs, calibration)   # numpy array
order = scores.argsort()[::-1]                      # most anomalous first
```

## Choosing a flag threshold

Because SCALE is length-invariant, a single threshold works across lengths. A
common choice is the high percentile of the scores of sequences you consider
clean (for example, the foundation's own simulated output), which fixes the
false-positive rate:

```python
import numpy as np

clean = reference.scale_score(list(reference.simulate(20_000, seed=1)), calibration)
threshold = np.percentile(clean, 98)   # ~2% false-positive rate on clean
flagged = scores >= threshold
```

## Interpreting the result

- A score near 0 means the sequence is as surprising as a typical sequence of
  its length under the model.
- Large positive scores are the anomalies: errors, contaminants, or rare
  sequences. Rank by score and inspect the top.
- The score is a robust z-score (median/IQR), so it is not thrown off by the
  heavy tail of `-log pgen`.

## Reusing a calibration

Calibration simulates from the graph, so do it once and reuse it:

```python
from LZGraphs import ScaleCalibration
calibration = ScaleCalibration.load('scale_calibration.json')
```

## Next steps

- [FlashBack: Exact Diversity](flashback-diversity.md)
- [FlashBack: Personalization & Algebra](flashback-algebra.md)
- [FlashBackGraph API](../api/flashback-graph.md)
