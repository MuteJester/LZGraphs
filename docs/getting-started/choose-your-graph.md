---
tags:
  - LZGraph
  - FlashBackGraph
---

# Which Graph Should I Use?

LZGraphs has two graph families. Pick by the question you are asking, not by the
algorithm underneath.

## Quick answer

| If you want to... | Use |
|-------------------|-----|
| Exact diversity, Hill numbers, or entropy with no sampling noise | **FlashBackGraph** |
| Self-calibrated per-sequence anomaly scores (SCALE) | **FlashBackGraph** |
| Exact generation probability and the most or least probable sequences | **FlashBackGraph** |
| V/J gene-aware modeling | **LZGraph** |
| ML feature vectors (reference-aligned, mass profiles, stats) | **LZGraph** |
| Occupancy and sharing predictions (richness at depth, sample overlap) | **LZGraph** |
| Very large repertoires with constant-memory streaming construction | **LZGraph** |

## The difference in one paragraph

**LZGraph** uses a coarsened LZ76 dictionary. It is the general-purpose tool:
it carries V/J gene statistics, scales to large repertoires, produces ML
features, and answers occupancy and sharing questions. Diversity and generation
are estimated efficiently, with simulation where a closed form is intractable.

**FlashBackGraph** builds a strictly Markovian graph, so diversity, entropy,
Hill numbers, and generation probability are computed exactly by forward
dynamic programming (no sampling), and it provides the fast FlashBack Anomaly
Score for surprise detection. It does not model V/J genes.

## Still unsure?

Start with **LZGraph** (`variant='aap'`). It covers the widest range of
repertoire analysis tasks. Move to FlashBackGraph when you specifically need
exact, sampling-free analytics or anomaly scoring.

## Next steps

- [LZGraph Quickstart](quickstart-lzgraph.md)
- [FlashBackGraph Quickstart](quickstart-flashback.md)
- Conceptual deep dive: [Graph Variants](../concepts/graph-types.md)
