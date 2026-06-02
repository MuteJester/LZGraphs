---
tags:
  - FlashBackGraph
description: Complete API reference for the FlashBackGrammar class, a weighted probabilistic context-free grammar over FlashBack decomposition trees with exact analytics.
search:
  boost: 2
---

# FlashBackGrammar

A weighted probabilistic context-free grammar (PCFG) over FlashBack decomposition **trees**. Where [`FlashBackGraph`](flashback-graph.md) models a Markov chain on linearised tokens, `FlashBackGrammar` models the decomposition tree directly, with non-terminals indexed by the boundary characters of the current middle. Rules are shared across recursion depths that share a boundary pair, giving cross-depth motif reuse while still guaranteeing every generated tree is a canonical FlashBack decomposition.

All analytics (path-count series, entropy, Hill numbers, dynamic range) are computed exactly via small linear systems over the non-terminal space.

## Constructor

```python
FlashBackGrammar(
    sequences,
    *,
    abundances=None,
    abundance_mode='linear',
    smoothing=0.0,
    backoff='none'
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sequences` | `list[str]` | Training CDR3 strings |
| `abundances` | `list[int]` | Per-sequence observed counts (default: all 1) |
| `abundance_mode` | `str` | How abundances become rule-count contributions: `'none'`, `'linear'` (default), or `'log'` |
| `smoothing` | `float` | Laplace alpha over observed rules at each non-terminal (default: 0.0) |
| `backoff` | `str` | Unseen-rule handling for `pgen`: `'none'` (strict MLE) or `'gt'` (Good-Turing). Analytics always use pure MLE; backoff only affects `pgen`. |

### from_file (classmethod)

```python
FlashBackGrammar.from_file(path, ...)
```
Build directly from a plain-text file (one sequence per line, or `sequence<TAB>abundance`).

## Core Methods

### pgen

```python
pgen(sequence, *, log=True)
```
Generation probability of one or more sequences under the grammar. Honours the `backoff` mode chosen at construction.

- **Parameters**: `sequence` (str or list[str]), `log` (keyword-only bool)
- **Returns**: `float` (single) or `np.ndarray` (list)

### pgen_mle

```python
pgen_mle(sequence, *, log=True)
```
Strict maximum-likelihood probability, ignoring any backoff. Useful for comparison against `pgen`.

### simulate

```python
simulate(n, *, seed=None)
```
Generate `n` sequences by sampling decomposition trees from the grammar.

- **Returns**: [SimulationResult](simulation-result.md)

### top_k_sequences

```python
top_k_sequences(k=100, *, most_probable=True)
```
Highest- (or lowest-) probability sequences under the grammar.

- **Returns**: [SimulationResult](simulation-result.md)

### posterior

```python
posterior(sequences, *, abundances=None, kappa=1.0)
```
Bayesian posterior grammar using this grammar as a prior. Returns a new `FlashBackGrammar`.

### without

```python
without(sequences, *, abundances=None)
```
Return a new grammar with the contribution of `sequences` removed.

## Exact Analytics

All values are computed exactly via linear systems over the non-terminal space.

| Method | Description |
|--------|-------------|
| `path_count(max_length)` | Number of distinct trees up to `max_length`. |
| `path_count_series(max_length)` | Path counts per length as an `np.ndarray`. |
| `length_pmf(max_length)` | Probability mass over sequence lengths. |
| `diversity_profile()` | Full Shannon diversity breakdown. |
| `effective_diversity()` | Exact `exp(H)`, equivalent to Hill `D(1)`. |
| `entropy()` | Exact Shannon entropy of the grammar. |
| `hill_number(alpha)` | Exact Hill number `D(alpha)`. |
| `hill_numbers(orders)` | Exact Hill numbers for multiple orders. |
| `hill_curve(orders=None)` | Dict of `orders` and `values` for plotting. |
| `power_sum(alpha)` | Exact power sum `M(alpha)`. |
| `pgen_dynamic_range(max_length=30)` | Dynamic range of probabilities in orders of magnitude. |
| `pgen_dynamic_range_detail(max_length=30)` | Full dynamic-range breakdown. |

## Grammar Inspection

| Member | Description |
|--------|-------------|
| `nonterminals` | List of `(a, z, is_leaf)` non-terminal descriptors. |
| `rules_at(a, z)` | Rules at the non-terminal indexed by boundary characters `a`, `z`. |
| `rules_at_start()` | Rules at the start non-terminal. |
| `top_rules(k=20, by='weight')` | The k highest-weight (or highest-count) rules. |
| `sentinel_rule_weights` | Weights of the sentinel rules. |

## IO

### save

```python
save(path)
```
Save to `.lzg` binary format.

### load (classmethod)

```python
FlashBackGrammar.load(path)
```
Load from a `.lzg` binary file.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `variant` | `str` | Always `'flashback_grammar'` |
| `n_nonterminals` | `int` | Number of non-terminals |
| `n_rules` | `int` | Total number of rules |
| `n_internal_rules` | `int` | Internal (branching) rules |
| `n_leaf_rules` | `int` | Leaf (terminal) rules |
| `alphabet` | `list[str]` | Characters seen in training |
| `spectral_radius` | `float` | Spectral radius of the rule system (consistency indicator) |
| `is_consistent` | `bool` | Whether the grammar defines a proper distribution |
| `smoothing` | `float` | Laplace alpha used |
| `backoff_mode` | `str` | `'none'` or `'gt'` |
| `abundance_mode` | `str` | `'none'`, `'linear'`, or `'log'` |
| `max_length` | `int` | Longest training sequence length |
| `length_counts` | `dict` | `{length: count}` mapping |
| `n_sequences` | `int` | Number of training sequences (abundance-weighted) |

## Decomposition utilities

The module-level functions `flashback_decompose` and `flashback_reverse` convert between a sequence and its FlashBack token decomposition. See [Module Functions](functions.md).
