---
tags:
  - FlashBackGraph
  - Diversity
---

# FlashBack P-Sequence Analytics

`FlashBackGraph.pseq_analysis()` describes the complete probability spectrum
of the sequences a FlashBack graph can generate. It operates on root-to-sink
paths of the graph and does not estimate the spectrum from simulated walks.

## The Mellin transform

For generated sequences \(s\) with probabilities \(P(s)\), define

\[
M(q)=\sum_s P(s)^q.
\]

Important special cases are:

- \(M(0)\): the number of supported sequences;
- \(M(1)\): the total generated probability mass, normally one;
- \(M(2)\): the collision probability, whose reciprocal is Hill diversity
  of order two.

Use `mellin()` when only \(M(q)\) is needed and `derivatives()` when moments
of log-probability are also required:

```python
analysis = graph.pseq_analysis()

mass = analysis.mellin(1.0)
jet = analysis.derivatives(1.0, order=4)
```

The returned derivative jet is

\[
M^{(r)}(q)=\sum_s P(s)^q\left(\log P(s)\right)^r,
\qquad r=0,\ldots,k.
\]

Consequently, `moments()` and `cumulants()` describe surprisal
\(X=-\log P(s)\) under the generated distribution. Orders zero through eight
are supported. The dynamic program is exact and sampling-free; returned
arrays use floating-point arithmetic, so their numerical precision is about
16 significant decimal digits.

## Stable tilted moments

Unnormalized derivatives can overflow for strong probability tilts even when
the corresponding normalized distribution is well behaved. Define

\[
\pi_q(s)=\frac{P(s)^q}{M(q)}.
\]

`tilted_moments()` evaluates this distribution directly in the log domain:

```python
tilted = analysis.tilted_moments(q=-2.0, order=4)

log_mass = tilted["log_mass"]
mean_log_p = tilted["raw_log_probability_moments"][1]
variance_log_p = tilted["central_log_probability_moments"][2]
fourth_cumulant = tilted["log_probability_cumulants"][4]
```

The raw moments are \(E_{\pi_q}[(\log P)^r]\). The central moments are
\(E_{\pi_q}[(\log P-E_{\pi_q}[\log P])^r]\), and the cumulants are derivatives
of \(\log M(q)\) with respect to \(q\). Every merge in the graph dynamic
program uses log-sum-exp normalized weights. This keeps `log_mass` and the
normalized moments finite across tilts for which `M(q)` itself cannot be
represented as a float64 number.

## Tilted structural attribution

`attribution()` explains which nodes and transitions carry a tilted
p-sequence distribution:

```python
attribution = analysis.attribution(q=1.0)

node_usage = attribution.node_probability
edge_usage = attribution.edge_probability
important = attribution.top_edges(20, by="surprisal")
```

If a path is drawn from \(\pi_q\), `node_probability[u]` is the probability
that it visits node `u`, and `edge_probability[e]` is the probability that it
traverses CSR edge `e`. Thus `q=1` attributes ordinary generated probability
flow, whereas `q=0` gives every supported path equal weight and attributes
fractions of the generated repertoire. Positive `q>1` emphasizes common
sequences; negative `q` emphasizes rare sequences.

The computation is a log-domain forward-backward dynamic program. If
\(\alpha(u)\) is the log mass of tilted prefixes ending at `u`, \(\beta(u)\)
is the log mass of tilted completions beginning there, and `e=(u,v)`, then

\[
\Pr_q(u)=\exp(\alpha(u)+\beta(u)-\log M(q)),
\]

\[
\Pr_q(e)=\exp(\alpha(u)+q\log w_e+\beta(v)-\log M(q)).
\]

These marginals obey flow conservation: a non-terminal node's incoming flow,
node probability, and outgoing flow agree. The root probability and total sink
probability are one. This supplies useful numerical diagnostics as well as the
attribution itself.

Several derived quantities are available without another graph traversal:

- `edge_sensitivity[e]` is
  \(\partial\log M(q)/\partial\log w_e=q\Pr_q(e)\), treating edge weights as
  independent variables;
- `edge_surprisal_contribution[e]` is
  \(-\Pr_q(e)\log w_e\), and its sum is mean tilted path surprisal;
- `expected_path_edges` is the expected number of graph transitions, not the
  reconstructed amino-acid sequence length;
- `top_edges(k, by=...)` reports labels, weights, occupancy, sensitivity, and
  surprisal contribution for the strongest transitions.

The independent-edge qualification on sensitivity matters: jointly
renormalizing every outgoing transition is a different constrained
perturbation and has a different derivative. The returned arrays are
read-only float64 views of native storage; the dynamic program accumulates in
`long double`, and the views keep their storage alive even when retained on
their own.

## Saddlepoint PDF and CDF

For generated-sequence surprisal \(X=-\log P(s)\), the normalized
cumulant-generating function is

\[
K(t)=\log M(1-t)-\log M(1).
\]

At an interior surprisal value \(x\), the saddlepoint \(\hat t\) solves

\[
K'(\hat t)=x.
\]

The solver combines Newton steps with a maintained bracket, so a Newton step
can never leave the valid root interval. The PDF uses the standard
saddlepoint density and the CDF uses the Lugannani-Rice correction. These are
smooth analytical approximations built from exact transform derivatives; they
are not exact inversions of a discrete probability spectrum.

```python
saddlepoint = analysis.saddlepoint()
x = np.linspace(
    analysis.true_min_surprisal,
    analysis.true_max_surprisal,
    200,
)

# Fused native batch: roots are solved once for both outputs.
pdf, cdf = saddlepoint.pdf_cdf(x)
```

Prefer `pdf_cdf()` when both results are needed. `pdf()` and `cdf()` also
accept arrays, but calling them separately solves the batch twice. Very small
discrete supports automatically use the deterministic grid fallback because a
smooth saddlepoint density is not a faithful representation of a few atoms.

## Generated sequence length

Length means the literal number of amino-acid characters in the reconstructed
sequence. It is not the number of graph nodes or edges. Sentinel characters
`@` and `$`, token positions, and token metadata do not contribute.

```python
counts = graph.path_count_by_length()
d0_at_most_27 = sum(
    count for length, count in counts.items() if length <= 27
)
```

`path_count_by_length()` is the length-resolved version of richness. It counts
every generated root-to-sink path, including recombinations whose lengths did
not occur in the training repertoire.

For probability mass and log-probability moments by length, use:

```python
jets = analysis.length_derivatives(q=1.0, order=4)
profile = analysis.length_profile()
```

For each length \(L\), `length_derivatives()` returns

\[
M_L^{(r)}(q)=
\sum_{s:\,|s|=L}P(s)^q\left(\log P(s)\right)^r.
\]

Summing these jets over all generated lengths reproduces the global
`derivatives()` result. `length_profile()` converts the (q=1) jets into
conditional mass, mean surprisal, variance, skewness, and kurtosis.

## Exact atoms and deterministic grids

`exact_atoms()` explicitly enumerates every supported path and is appropriate
only when the support is small. For a large support, `histogram()` transports
the spectrum onto a deterministic surprisal grid:

```python
generated = analysis.histogram(
    bins=4096,
    measure="generated",
)

counting_at_length_15 = analysis.histogram(
    bins=4096,
    measure="counting",
    length=15,
)
```

The generated measure gives each sequence its probability \(P(s)\); without a
length restriction, its total mass is normally one. The counting measure gives
each supported sequence unit mass; without a restriction, its total is
\(M(0)\). Passing `length` restricts either raw measure to sequences with
exactly that reconstructed amino-acid length.

Each transition generally falls between two grid points, so its mass is split
linearly between the adjacent points. This conserves total mass and the first
surprisal moment. The reconstruction is deterministic and has no Monte Carlo
variance, but it is still a grid approximation: `grid_spacing` reports its
resolution and `max_rounding_error` bounds the worst accumulated placement
error along a path.

## Choosing an interface

| Question | Interface |
|---|---|
| How many sequences can be generated at each AA length? | `path_count_by_length()` |
| What are global probability or surprisal moments? | `derivatives()`, `moments()`, `cumulants()` |
| What happens under a strong probability tilt? | `log_mellin()`, `tilted_moments()` |
| Which nodes and edges carry a tilted distribution? | `attribution()` |
| What are those quantities at each AA length? | `length_derivatives()`, `length_profile()` |
| Can I enumerate every probability atom? | `exact_atoms()` for small supports |
| What does a large probability spectrum look like? | `histogram()` |
| What smooth PDF/CDF approximates the generated spectrum? | `saddlepoint().pdf_cdf()` |
| Where does one sequence lie in the spectrum? | `position()` |

Creating the analysis object reads the graph's existing topological structure;
it does not alter the graph. Reuse one analysis object when making several
queries:

```python
analysis = graph.pseq_analysis()
moments = analysis.moments()
by_length = analysis.length_profile()
histogram = analysis.histogram(4096)
```
