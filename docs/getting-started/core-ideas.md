---
tags:
  - LZGraph
  - FlashBackGraph
---

# Core Ideas

Getting started with LZGraphs takes only a few ideas. This page covers them
before you build your first graph; no graph theory background needed.

## Your repertoire as a graph

A TCR or BCR repertoire is, at heart, a list of CDR3 sequences. LZGraphs turns
that list into a directed graph: sequences that share motifs share paths, and
where they diverge the graph branches. The shape of that graph, how it branches
and how often each path is taken, is what lets you ask questions about
diversity, generation probability, and similarity, without alignment or a
germline reference.

You give LZGraphs a plain `list[str]` of sequences. Optionally you can also
provide `abundances` (clonotype counts) so the graph reflects the expanded
state of the repertoire rather than treating every unique sequence equally.

## Nodes and edges

- A **node** is a recurring subpattern of sequence (and, in some graph types,
  its position). Shared subpatterns collapse into a single node.
- An **edge** is an observed transition from one subpattern to the next. Edge
  weights are transition probabilities estimated from your data.

You can see the subpatterns a sequence decomposes into:

```python
from LZGraphs import lz76_decompose

print(lz76_decompose("CASSLEPSGGTDTQYF"))
# ['C', 'A', 'S', 'SL', 'E', 'P', 'SG', 'G', 'T', 'D', 'TQ', 'Y', 'F']
```

Each token becomes a node; consecutive tokens become edges. That is the whole
idea. For the algorithm behind it, see
[LZ76 Algorithm](../concepts/lz76-algorithm.md).

## Two graph families

LZGraphs gives you two ways to build this graph:

- **LZGraph** uses a coarsened LZ76 dictionary. It is the general-purpose
  choice: V/J gene aware, scales to large repertoires, and supports ML features
  and occupancy / sharing predictions. It has three variants (`aap`, `ndp`,
  `naive`) for amino acid, nucleotide, and position-free analysis.
- **FlashBackGraph** builds a strictly Markovian graph from the FlashBack
  decomposition, so diversity, entropy, Hill numbers, and generation
  probability are computed exactly (no sampling), and it provides a fast
  per-sequence anomaly score.

Not sure which to use? See [Which Graph Should I Use?](choose-your-graph.md).

## Next steps

- [Which Graph Should I Use?](choose-your-graph.md)
- [LZGraph Quickstart](quickstart-lzgraph.md)
- [FlashBackGraph Quickstart](quickstart-flashback.md)
- [LZ76 Algorithm](../concepts/lz76-algorithm.md) for the encoding itself
