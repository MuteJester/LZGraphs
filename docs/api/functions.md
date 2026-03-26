# Module Functions

Standalone functions that operate on graphs or sequences. Import them from the top-level package:

```python
from LZGraphs import (
    jensen_shannon_divergence,
    k_diversity,
    saturation_curve,
    lz76_decompose,
    set_log_level,
    set_log_callback,
)
```

---

## Repertoire Comparison

### jensen_shannon_divergence

```python
jensen_shannon_divergence(graph_a, graph_b)
```

Compute the Jensen-Shannon Divergence between two LZGraphs. JSD is a symmetric, bounded measure of distributional divergence over the shared subpattern space.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph_a` | `LZGraph` | First graph |
| `graph_b` | `LZGraph` | Second graph (must be the same variant) |

**Returns:** `float` — JSD value in $[0, 1]$. 0 = identical distributions, 1 = maximally different.

**Example:**

```python
from LZGraphs import LZGraph, jensen_shannon_divergence

g1 = LZGraph(seqs_healthy, variant='aap')
g2 = LZGraph(seqs_disease, variant='aap')

jsd = jensen_shannon_divergence(g1, g2)
print(f"JSD = {jsd:.4f}")
# 0.00-0.05: nearly identical
# 0.05-0.15: very similar
# 0.15-0.30: moderately different
# 0.30+: substantially different
```

---

## Diversity Analysis

### k_diversity

```python
k_diversity(sequences, k, *, variant='aap', draws=100, seed=None)
```

Subsample-based diversity metric. Repeatedly draws $k$ sequences, builds an LZGraph, counts unique subpattern nodes, and reports statistics across `draws` resampling rounds.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `list[str]` | — | Input CDR3 sequences |
| `k` | `int` | — | Subsample size |
| `variant` | `str` | `'aap'` | Graph encoding variant |
| `draws` | `int` | `100` | Number of resampling rounds |
| `seed` | `int` or `None` | `None` | RNG seed for reproducibility (-1 or None = random) |

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `'mean'` | `float` | Mean number of unique nodes across draws |
| `'std'` | `float` | Standard deviation |
| `'ci_low'` | `float` | Lower bound of 95% confidence interval |
| `'ci_high'` | `float` | Upper bound of 95% confidence interval |

**Example:**

```python
from LZGraphs import k_diversity

result = k_diversity(sequences, k=1000, variant='aap', draws=100)
print(f"K(1000) = {result['mean']:.1f} +/- {result['std']:.1f}")
print(f"95% CI:  [{result['ci_low']:.1f}, {result['ci_high']:.1f}]")
```

!!! tip "Choosing k"
    Pick $k$ well below your smallest repertoire so all samples can be compared on the same footing. Common choices: 500, 1000, 5000.

---

### saturation_curve

```python
saturation_curve(sequences, *, variant='aap', log_every=100)
```

Compute the node/edge saturation curve: add sequences one at a time and record how the graph grows.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequences` | `list[str]` | — | Input sequences (order matters) |
| `variant` | `str` | `'aap'` | Graph encoding variant |
| `log_every` | `int` | `100` | Record a data point every N sequences |

**Returns:** `list[dict]`, where each dict has:

| Key | Type | Description |
|-----|------|-------------|
| `'n_sequences'` | `int` | Number of sequences added so far |
| `'n_nodes'` | `int` | Total nodes in the graph at this point |
| `'n_edges'` | `int` | Total edges in the graph at this point |

**Example:**

```python
from LZGraphs import saturation_curve

curve = saturation_curve(sequences, variant='aap', log_every=500)
for point in curve[:5]:
    print(f"After {point['n_sequences']:5d} seqs: "
          f"{point['n_nodes']:5d} nodes, {point['n_edges']:5d} edges")
```

---

## LZ76 Utilities

### lz76_decompose

```python
lz76_decompose(sequence)
```

Decompose a string into its LZ76 subpatterns. This is the raw decomposition without positional encoding — the same algorithm used internally by all graph variants.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequence` | `str` | Input string (amino acid, nucleotide, or any characters) |

**Returns:** `list[str]` — the LZ76 subpatterns.

**Example:**

```python
from LZGraphs import lz76_decompose

tokens = lz76_decompose("CASSLEPSGGTDTQYF")
print(tokens)
# ['C', 'A', 'S', 'SL', 'E', 'P', 'SG', 'G', 'T', 'D', 'TQ', 'Y', 'F']

# The number of tokens measures the sequence's LZ76 complexity
print(f"Complexity: {len(tokens)} tokens for {len('CASSLEPSGGTDTQYF')} chars")
```

!!! info "How LZ76 works"
    At each step, the algorithm finds the shortest substring that hasn't appeared before. `C` is new, `A` is new, `S` is new. Then `S` is already known, so we extend to `SL` (new). This greedy process continues until the string is consumed. See [LZ76 Algorithm](../concepts/lz76-algorithm.md) for a detailed explanation.

---

## Configuration

### set_log_level

```python
set_log_level(level)
```

Enable or disable logging from the C backend. Messages are written to stderr.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `str` | One of: `'none'`, `'error'`, `'warn'`, `'info'`, `'debug'`, `'trace'` |

**Example:**

```python
import LZGraphs

LZGraphs.set_log_level('info')     # See build progress and timing
graph = LZGraph(sequences, variant='aap')

LZGraphs.set_log_level('none')     # Silence all output (default)
```

### set_log_callback

```python
set_log_callback(callback, level='info')
```

Route log messages to a custom Python callback instead of stderr. Useful for integrating with Python's `logging` module.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `callback` | `callable` or `None` | Function `(level: int, message: str) -> None`. Pass `None` to disable. |
| `level` | `str` | Maximum level to emit (default: `'info'`) |

Level values passed to the callback: 1=error, 2=warn, 3=info, 4=debug, 5=trace.

**Example:**

```python
import logging
import LZGraphs

logger = logging.getLogger('lzgraphs')
LEVEL_MAP = {1: logging.ERROR, 2: logging.WARNING, 3: logging.INFO,
             4: logging.DEBUG, 5: logging.DEBUG}

LZGraphs.set_log_callback(
    lambda lvl, msg: logger.log(LEVEL_MAP.get(lvl, logging.DEBUG), msg),
    level='info',
)
```

---

## See Also

- [LZGraph class](lzgraph.md) — the main class with all instance methods
- [CLI Reference](cli.md) — command-line equivalents
- [Diversity Metrics tutorial](../tutorials/diversity-metrics.md) — using k_diversity and JSD in practice
