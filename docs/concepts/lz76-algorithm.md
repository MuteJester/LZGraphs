# LZ76 Algorithm

The Lempel-Ziv 1976 (LZ76) algorithm is the foundation of LZGraphs' sequence encoding. Understanding how it works helps you interpret graph structure and encoding results.

## What is LZ76?

LZ76 is a lossless compression algorithm that decomposes a sequence into a series of unique subpatterns. Each new pattern extends a previously seen pattern by one character.

## How It Works

### Step-by-Step Decomposition

Let's decompose the sequence `CASSLE`:

```
Input: C A S S L E
       │ │ │ │ │ │
Step 1: C          → Dictionary: {C}
Step 2:   A        → Dictionary: {C, A}
Step 3:     S      → Dictionary: {C, A, S}
Step 4:       S L  → "SL" (S + new char L) → Dictionary: {C, A, S, SL}
Step 5:           E → Dictionary: {C, A, S, SL, E}

Result: [C, A, S, SL, E]
```

### Key Properties

1. **Unique patterns**: Each pattern in the decomposition is unique
2. **Incremental building**: New patterns extend previous ones
3. **Lossless**: Original sequence can be reconstructed
4. **Deterministic**: Same input always produces same output

## Python Implementation

LZGraphs uses this algorithm internally:

```python
from LZGraphs.utilities import lempel_ziv_decomposition

# Basic decomposition
sequence = "CASSLEPSGGTDTQYF"
patterns = lempel_ziv_decomposition(sequence)
print(patterns)
# ['C', 'A', 'S', 'SL', 'E', 'P', 'SG', 'G', 'T', 'D', 'TQ', 'Y', 'F']
```

### Verify Reconstruction

```python
# Patterns concatenate to original
reconstructed = ''.join(patterns)
print(f"Original:      {sequence}")
print(f"Reconstructed: {reconstructed}")
print(f"Match: {sequence == reconstructed}")
```

## Why LZ76 for Sequences?

### 1. Captures Structure

Unlike k-mer approaches, LZ76 adapts to the sequence:

```python
# Repetitive sequence
rep_seq = "AAAAAA"
print(lempel_ziv_decomposition(rep_seq))
# ['A', 'AA', 'AAA']  - Captures repetition

# Diverse sequence
div_seq = "ABCDEF"
print(lempel_ziv_decomposition(div_seq))
# ['A', 'B', 'C', 'D', 'E', 'F']  - All unique
```

### 2. Variable-Length Patterns

LZ76 naturally finds patterns of different lengths:

```python
sequence = "CASSLEPSGGTDTQYF"
patterns = lempel_ziv_decomposition(sequence)

# Pattern lengths vary based on repetition
lengths = [len(p) for p in patterns]
print(f"Patterns: {patterns}")
print(f"Lengths:  {lengths}")
```

### 3. Compression Reflects Complexity

```python
# More repetitive = fewer patterns
simple = "AAAAAAAAAAAA"
complex = "ABCDEFGHIJKL"

simple_patterns = lempel_ziv_decomposition(simple)
complex_patterns = lempel_ziv_decomposition(complex)

print(f"Simple ({len(simple)} chars):  {len(simple_patterns)} patterns")
print(f"Complex ({len(complex)} chars): {len(complex_patterns)} patterns")
```

## Positional Encoding

LZGraphs extends LZ76 with position information:

### AAPLZGraph Encoding

```python
from LZGraphs import AAPLZGraph

sequence = "CASSLE"
encoded = AAPLZGraph.encode_sequence(sequence)
print(encoded)
# ['C_1', 'A_2', 'S_3', 'SL_5', 'E_6']
```

The `_N` suffix indicates the **ending position** of each pattern:

| Pattern | Start | End | Encoded |
|---------|-------|-----|---------|
| C | 1 | 1 | C_1 |
| A | 2 | 2 | A_2 |
| S | 3 | 3 | S_3 |
| SL | 4 | 5 | SL_5 |
| E | 6 | 6 | E_6 |

### NDPLZGraph Encoding

NDPLZGraph uses reading frame and position encoding:

```python
from LZGraphs import NDPLZGraph

sequence = "TGTGCC"
encoded = NDPLZGraph.encode_sequence(sequence)
print(encoded)
# ['T0_1', 'G1_2', 'TG2_4', 'C1_5', 'C2_6']
```

Each node has the format `{subpattern}{reading_frame}_{position}`, where the reading frame
(0, 1, or 2) indicates the codon position.

## Why Position Matters

In CDR3 sequences, position carries biological meaning:

- **Early positions**: Often V-gene derived
- **Middle positions**: Junction diversity
- **Late positions**: Often J-gene derived

Positional encoding allows the graph to distinguish:

```python
# Same pattern, different positions
seq1 = "CASSG"   # G at position 5
seq2 = "GCASS"   # G at position 1

enc1 = AAPLZGraph.encode_sequence(seq1)
enc2 = AAPLZGraph.encode_sequence(seq2)

print(f"seq1: {enc1}")  # [..., 'G_5']
print(f"seq2: {enc2}")  # ['G_1', ...]
```

## Mathematical Properties

### Compression Ratio

The number of LZ76 patterns relative to sequence length indicates complexity:

```python
def complexity_ratio(sequence):
    patterns = lempel_ziv_decomposition(sequence)
    return len(patterns) / len(sequence)

# Compare sequences
sequences = [
    "AAAAAAAAAA",      # Very simple
    "CASSLEPSGGTDTQYF", # Typical CDR3
    "ABCDEFGHIJ"        # Maximum complexity
]

for seq in sequences:
    ratio = complexity_ratio(seq)
    print(f"{seq[:15]:15s} ratio: {ratio:.2f}")
```

### Pattern Growth

For a sequence of length n:

- **Maximum patterns**: n (every character is new)
- **Minimum patterns**: O(log n) (highly repetitive)
- **Typical CDR3**: Between these extremes

## Connection to Diversity

LZ76 complexity connects to repertoire diversity:

- **K-diversity**: Counts unique patterns across samples
- **Graph nodes**: Each pattern becomes a potential node
- **Graph edges**: Pattern transitions become edges

This creates a natural bridge from sequences to graph metrics.

## Next Steps

- [Graph Types](graph-types.md) - How encoding affects graph structure
- [Probability Model](probability-model.md) - Using patterns for probability
- [Tutorials: Diversity Metrics](../tutorials/diversity-metrics.md) - Apply to real data
