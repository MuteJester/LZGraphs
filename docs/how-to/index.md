# How-To Guides

Task-oriented guides for specific LZGraphs operations.

## Available Guides

<div class="grid" markdown>

<div class="card" markdown>
### [Save & Load Graphs](serialization.md)
Persist graphs to disk and reload them later
</div>

<div class="card" markdown>
### [Generate Sequences](sequence-generation.md)
Create new sequences following repertoire statistics
</div>

<div class="card" markdown>
### [Compare Repertoires](repertoire-comparison.md)
Measure similarity between different repertoires
</div>

<div class="card" markdown>
### [Personalize Graphs](posterior-personalization.md)
Adapt a population graph to an individual using Bayesian posteriors
</div>

</div>

## Quick Reference

| Task | Guide | Key Functions |
|------|-------|---------------|
| Save a graph | [Serialization](serialization.md) | `graph.save()` |
| Load a graph | [Serialization](serialization.md) | `AAPLZGraph.load()` |
| Generate sequences | [Sequence Generation](sequence-generation.md) | `random_walk()`, `genomic_random_walk()` |
| Compare repertoires | [Comparison](repertoire-comparison.md) | `jensen_shannon_divergence()` |
| Personalize a graph | [Posterior](posterior-personalization.md) | `get_posterior()` |

## Need More?

- **Step-by-step learning?** → [Tutorials](../tutorials/index.md)
- **Understanding concepts?** → [Concepts](../concepts/index.md)
- **API details?** → [API Reference](../api/index.md)
