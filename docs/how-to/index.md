# How-To Guides

Task-oriented guides for specific LZGraphs operations. Each guide focuses on a single task and gives you the recipe to accomplish it.

## Available Guides

<div class="grid" markdown>

<div class="card" markdown>
### [Prepare Your Data](data-preparation.md)
Load sequences from CSV/TSV, clean input, and handle AIRR-format files
</div>

<div class="card" markdown>
### [Save & Load Graphs](serialization.md)
Persist graphs to disk in the fast `.lzg` binary format
</div>

<div class="card" markdown>
### [Generate Sequences](sequence-generation.md)
Create new sequences with gene constraints, filtering, and reproducibility
</div>

<div class="card" markdown>
### [Compare Repertoires](repertoire-comparison.md)
Measure similarity with JSD, diversity profiles, and cross-scoring
</div>

<div class="card" markdown>
### [Personalize Graphs](posterior-personalization.md)
Adapt a population graph to an individual using Bayesian posteriors
</div>

<div class="card" markdown>
### [Distribution Analytics](distribution-analytics.md)
Validate distributions, measure diversity, and predict occupancy
</div>

<div class="card" markdown>
### [Graph Algebra](graph-algebra.md)
Union, intersection, difference — combine and decompose repertoires
</div>

<div class="card" markdown>
### [Feature Extraction for ML](feature-extraction.md)
Extract fixed-size feature vectors for classifiers and pipelines
</div>

</div>

## Quick Reference

| Task | Guide | Key Functions |
|------|-------|---------------|
| Load from CSV/TSV | [Data Preparation](data-preparation.md) | `csv.DictReader` + `LZGraph()` |
| Save a graph | [Serialization](serialization.md) | `graph.save()` |
| Load a graph | [Serialization](serialization.md) | `LZGraph.load()` |
| Generate sequences | [Sequence Generation](sequence-generation.md) | `graph.simulate()` |
| Compare repertoires | [Comparison](repertoire-comparison.md) | `jensen_shannon_divergence()` |
| Personalize a graph | [Posterior](posterior-personalization.md) | `graph.posterior()` |
| Measure diversity | [Distribution Analytics](distribution-analytics.md) | `graph.hill_number()`, `graph.predicted_richness()` |
| Combine repertoires | [Graph Algebra](graph-algebra.md) | `graph \| other`, `graph & other`, `graph - other` |
| ML features | [Feature Extraction](feature-extraction.md) | `ref.feature_aligned(query)`, `graph.feature_stats()` |

## Need More?

- **Step-by-step learning?** See [Tutorials](../tutorials/index.md)
- **Understanding the theory?** See [Concepts](../concepts/index.md)
- **API details?** See [API Reference](../api/index.md)
