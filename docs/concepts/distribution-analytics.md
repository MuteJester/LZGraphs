# Distribution Analytics

LZGraphs can answer fundamental questions about the generative distribution encoded in a graph: How many sequences can it produce? Is the probability distribution well-formed? What is its dynamic range? This page explains the mathematical and computational foundations of these analytics.

## The Generative Distribution

An LZGraph defines a probability distribution over sequences. Each sequence corresponds to a walk from the root node (`@`) to a terminal node (containing `$`).

The methods described below characterize this distribution using a combination of **exact forward propagation** and **high-performance Monte Carlo estimation**.

## Path Count (Richness)

**Question**: How many unique sequences can the graph produce?

This is the combinatorial size of the sequence space encoded by the graph. Because immune repertoires are extremely diverse, this number is often astronomically large (e.g., \(10^{80}\) or more).

**Algorithm**: LZGraphs uses a frequentist estimator (Chao1) based on a large-scale simulation to provide a lower bound on the total number of reachable sequences.

```python
print(f"Combinatorial space: {graph.path_count:.2e}")
```

## PGEN Moments & Diagnostics

**Question**: What are the average properties of the distribution?

LZGraphs uses a high-performance **forward propagation** engine to compute the exact moments of the log-PGEN distribution directly from the graph topology.

**Algorithm**: Moments are propagated through the DAG in topological order. This provides the exact mean, variance, skewness, and kurtosis of the generative model.

```python
moments = graph.pgen_moments()
print(f"Mean log-Pgen: {moments['mean']:.2f}")
print(f"Std Dev log-Pgen: {moments['std']:.2f}")
```

**Diagnostics**: The `pgen_diagnostics()` method verifies that the transition probabilities are correctly normalized and that the total probability mass sums to ~1.0.

## Diversity Metrics (Hill Numbers)

**Question**: How concentrated is the probability mass?

Hill numbers (\(D^\alpha\)) provide a unified profile of repertoire diversity. Higher orders of \(\alpha\) emphasize common sequences, while lower orders emphasize rare ones.

| Order | Name | Interpretation |
|:-----:|------|----------------|
| 0 | Richness | Total number of unique sequences |
| 1 | Shannon | Effective diversity (\(e^H\)) |
| 2 | Simpson | Collision diversity (\(1/\sum p^2\)) |

**Algorithm**: LZGraphs uses exact-probability Monte Carlo estimators. Because every simulated sequence carries its **exact** generation probability (not just its frequency in the sample), these estimators are far more accurate than traditional frequency-based methods, especially for rare sequences.

```python
# Compute Hill numbers for orders 0, 1, and 2
hills = graph.hill_numbers([0, 1, 2])
```

## Occupancy Predictions

**Question**: What will my repertoire look like at a different sequencing depth?

One of the most powerful features of LZGraphs is its ability to predict **richness** and **overlap** at arbitrary depths using the Poisson occupancy model.

### Predicted Richness

The expected number of distinct sequences \(F(d)\) at depth \(d\) is calculated using a sophisticated **splitting + Taylor + Wynn** algorithm:

1. **Split**: High-probability sequences are discovered via simulation and their contribution is calculated exactly.
2. **Taylor**: The contribution of the "long tail" of rare sequences is calculated using a Taylor series expansion of the occupancy function.
3. **Wynn**: The Wynn epsilon algorithm is applied to accelerate the convergence of the series, ensuring machine-precision results even at massive depths.

```python
# How many unique sequences would I see at 1 million reads?
richness = graph.predicted_richness(1_000_000)
```

### Predicted Overlap

The expected overlap between two samples of depths \(d_i\) and \(d_j\) is computed using the identity:
\[
G(d_i, d_j) = F(d_i) + F(d_j) - F(d_i + d_j)
\]

This allows for rapid estimation of sample similarity without needing to simulate the sampling process.

## Analytical PGEN Distribution

LZGraphs can construct an analytical Gaussian mixture model that approximates the full log-PGEN distribution of the repertoire.

```python
dist = graph.pgen_distribution()
# Use the distribution to compute PDF, CDF, or sample values
pdf_value = dist.pdf(-25.0)
```

This distribution is used internally for sharing spectrum predictions across large cohorts.

## Summary of Analytics

| Method | Returns | Type |
|--------|---------|------|
| `path_count` | Float | Combinatorial: total paths |
| `pgen_moments()` | Dict | Exact: Mean, Std, Skewness |
| `effective_diversity()` | Float | Estimated: \(e^H\) |
| `hill_number(alpha)` | Float | Estimated: Diversity at order \(\alpha\) |
| `predicted_richness(d)` | Float | Analytical: Expected richness at depth \(d\) |
| `pgen_distribution()` | Object | Analytical: Gaussian mixture of PGEN |

## Next Steps

- [How-To: Distribution Analytics](../how-to/distribution-analytics.md) — Practical code examples
- [Probability Model](probability-model.md) — How individual probabilities are computed
- [API Reference](../api/index.md) — Detailed method documentation
