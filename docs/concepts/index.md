# Concepts

Understanding the theory behind LZGraphs will help you use it more effectively and interpret results correctly.

## Core Concepts

<div class="grid" markdown>

<div class="card" markdown>
### [LZ76 Algorithm](lz76-algorithm.md)
How Lempel-Ziv compression creates sequence encodings
</div>

<div class="card" markdown>
### [Graph Types](graph-types.md)
Comparison of AAPLZGraph, NDPLZGraph, and NaiveLZGraph
</div>

<div class="card" markdown>
### [Probability Model](probability-model.md)
How LZGraphs calculates sequence generation probabilities
</div>

</div>

## The Big Picture

LZGraphs represents a TCR repertoire as a directed graph where:

1. **Sequences become walks** - Each CDR3 sequence is a path through the graph
2. **Patterns become nodes** - Subpatterns from LZ76 decomposition are nodes
3. **Transitions become edges** - Observed pattern transitions are edges
4. **Frequencies become weights** - How often transitions occur determines edge weights

This representation enables:

- **Efficient probability calculation** - O(n) instead of O(n²)
- **Pattern discovery** - Find common motifs and rare variations
- **Sequence generation** - Sample new sequences with realistic statistics
- **Diversity quantification** - Measure complexity through graph topology

## Why Graphs?

Traditional approaches to repertoire analysis face challenges:

| Challenge | Traditional Approach | LZGraphs Approach |
|-----------|---------------------|-------------------|
| Comparing sequences | Pairwise alignment (O(n²)) | Walk probability (O(n)) |
| Finding patterns | K-mer counting | Graph structure |
| Generating sequences | Statistical models | Random walks |
| Cross-repertoire comparison | Sequence overlap | Graph divergence |

## Key Insights

### 1. Position Matters

The positional encoding in AAPLZGraph and NDPLZGraph captures that:

- The same amino acid at position 3 vs position 10 has different meaning
- CDR3 structure follows positional constraints
- V/J gene contributions vary by position

### 2. Context Matters

The graph captures:

- Which patterns can follow which
- Gene-specific transition preferences
- Repertoire-specific motifs

### 3. Frequency Matters

Edge weights encode:

- Common vs rare transitions
- Probability of sequence generation
- Deviation from expected patterns

## Next Steps

Dive deeper into specific concepts:

- [LZ76 Algorithm](lz76-algorithm.md) - Understand the encoding
- [Graph Types](graph-types.md) - Choose the right representation
- [Probability Model](probability-model.md) - Calculate sequence likelihood
