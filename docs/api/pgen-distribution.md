# PgenDistribution

Analytical Gaussian mixture model of a repertoire's generation probability (PGEN) distribution. Obtained via [`LZGraph.pgen_distribution()`](lzgraph.md).

The distribution models $\log P_{\text{gen}}$ across all producible sequences as a mixture of Gaussians, fitted from length-stratified forward propagation through the graph. This allows you to evaluate the PDF, CDF, and draw samples without simulation.

## Quick Example

```python
dist = graph.pgen_distribution()

# Summary
print(f"Components: {dist.n_components}")
print(f"Mean log-Pgen: {dist.mean:.2f}")
print(f"Std log-Pgen:  {dist.std:.2f}")

# Evaluate the density at a specific point
print(f"PDF at mean: {dist.pdf(dist.mean):.6f}")

# What fraction of sequences have log-Pgen < -20?
print(f"CDF at -20: {dist.cdf(-20.0):.4f}")

# Draw random samples
samples = dist.sample(10000, seed=42)
print(f"Sample mean: {samples.mean():.2f}")
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_components` | `int` | Number of Gaussian components in the mixture |
| `weights` | `np.ndarray` | Mixture weights (sum to 1.0), one per component |
| `means` | `np.ndarray` | Mean log-Pgen for each component |
| `stds` | `np.ndarray` | Standard deviation for each component |
| `mean` | `float` | Overall mean log-Pgen (weighted average of component means) |
| `std` | `float` | Overall standard deviation |

!!! info "What are the components?"
    Each Gaussian component corresponds to a **sequence length stratum**. Sequences of length 10 have a different Pgen distribution than sequences of length 15, because they take different paths through the graph. The mixture model captures this length-dependent structure.

## Methods

### `pdf(x)`

Evaluate the probability density function at one or more points.

```python
import numpy as np

dist = graph.pgen_distribution()

# Single point
density = dist.pdf(-15.0)

# Array of points
x = np.linspace(-30, -5, 500)
densities = dist.pdf(x)  # returns np.ndarray of same shape
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `float` or `np.ndarray` | Log-Pgen value(s) to evaluate |

**Returns:** `float` or `np.ndarray` — density value(s)

### `cdf(x)`

Evaluate the cumulative distribution function — the probability that a random sequence has log-Pgen $\leq x$.

```python
# What fraction of sequences have log-Pgen below -20?
fraction_below = dist.cdf(-20.0)
print(f"{fraction_below:.1%} of sequences have log-Pgen < -20")
```

**Parameters:** Same as `pdf`.

**Returns:** `float` or `np.ndarray` — cumulative probability value(s) in $[0, 1]$.

### `sample(n, seed=None)`

Draw random log-Pgen values from the mixture distribution.

```python
samples = dist.sample(10000, seed=42)
print(f"Mean:   {samples.mean():.2f}")
print(f"Median: {np.median(samples):.2f}")
print(f"Std:    {samples.std():.2f}")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `n` | `int` | Number of samples to draw |
| `seed` | `int` or `None` | RNG seed for reproducibility |

**Returns:** `np.ndarray` of shape `(n,)` — sampled log-Pgen values.

## Plotting the Distribution

```python
import numpy as np
import matplotlib.pyplot as plt

dist = graph.pgen_distribution()

x = np.linspace(dist.mean - 4*dist.std, dist.mean + 4*dist.std, 500)
pdf = dist.pdf(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, pdf, linewidth=2, label='Analytical PDF')
ax.axvline(dist.mean, color='red', linestyle='--', alpha=0.7, label=f'Mean = {dist.mean:.1f}')
ax.fill_between(x, pdf, alpha=0.15)
ax.set_xlabel('log P(gen)')
ax.set_ylabel('Density')
ax.set_title('PGEN Distribution')
ax.legend()
plt.tight_layout()
plt.show()
```

## Comparing Distributions

You can compare the PGEN distributions of two repertoires by overlaying their PDFs:

```python
dist_a = graph_a.pgen_distribution()
dist_b = graph_b.pgen_distribution()

x = np.linspace(-35, -5, 500)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, dist_a.pdf(x), label=f'Repertoire A (mean={dist_a.mean:.1f})')
ax.plot(x, dist_b.pdf(x), label=f'Repertoire B (mean={dist_b.mean:.1f})')
ax.set_xlabel('log P(gen)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.show()
```

A rightward shift means the repertoire tends to produce higher-probability (more "expected") sequences. A wider distribution means more spread between common and rare sequences.

## See Also

- [`LZGraph.pgen_moments()`](lzgraph.md#pgen_moments) — quick mean/variance without the full mixture
- [`LZGraph.pgen_dynamic_range()`](lzgraph.md) — how many orders of magnitude the distribution spans
- [Distribution Analytics (concepts)](../concepts/distribution-analytics.md) — mathematical foundations
- [Diversity Metrics tutorial](../tutorials/diversity-metrics.md) — how the distribution relates to diversity
