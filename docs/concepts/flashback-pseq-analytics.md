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

## Diversity after edge-threshold pruning

`diversity_under_edge_thresholds()` evaluates an entire edge-pruning curve in
one native call:

```python
thresholds = np.percentile(graph.adjacency_csr()["weights"], [0, 25, 50, 75])
curve = analysis.diversity_under_edge_thresholds(thresholds)

d0 = curve["D0"]
d1 = curve["D1"]
d2 = curve["D2"]
surviving_mass = curve["surviving_mass"]
edge_fraction = curve["edge_fraction"]
```

At threshold \(\tau\), an edge is retained strictly when \(w_e>\tau\).
Only paths that still reach a sink of the original graph count as generated
sequences. An internal node whose outgoing edges were removed is a dead end;
it does not become a new sequence endpoint.

Let \(S_\tau\) be the surviving sequences and
\(Z_\tau=\sum_{s\in S_\tau}P(s)\). The reported diversities describe the
conditional distribution \(P(s)/Z_\tau\):

\[
D_0=|S_\tau|,\qquad
D_1=\exp\left(-\sum_{s\in S_\tau}\frac{P(s)}{Z_\tau}
\log\frac{P(s)}{Z_\tau}\right),\qquad
D_2=\left(\sum_{s\in S_\tau}
\left(\frac{P(s)}{Z_\tau}\right)^2\right)^{-1}.
\]

The result also includes natural-log versions (`log_D0`, `log_D1`, and
`log_D2`) so very large richness values can be plotted without exponentiating.
If no complete path survives, ordinary diversities and mass are zero and the
log diversities are negative infinity. `kept_edges` and `edge_fraction` count
all CSR edges satisfying the cutoff, even if pruning makes an edge unreachable
from the root. Inputs may be unsorted or repeated; outputs preserve their
order.

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
marginals = analysis.length_marginals()
jets = analysis.length_derivatives(q=1.0, order=4)
profile = analysis.length_profile()
```

`length_marginals()` is the efficient interface when both supported richness
and generated probability mass are needed. It returns one entry per reachable
amino-acid length:

```python
{
    14: {"counting": 1.23e31, "generated": 0.18},
    # ...
}
```

The native dynamic program transports both measures in one topological DAG
walk. `counting` is the zeroth-order transform at (q=0), and `generated` is
the zeroth-order transform at (q=1). Counting uses float64, consistent with
`path_count_by_length()`; generated probability mass is accumulated in
extended precision before conversion to a Python float. Use
`length_derivatives()` when derivatives or another tilt are required.

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

# When both global measures are needed, construct them in one traversal.
spectra = analysis.histogram_pair(bins=4096)
counting = spectra["counting"]
generated = spectra["generated"]
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

`histogram_pair()` is the efficient global interface when both counting and
generated spectra are required. It computes the common grid bounds and walks
the DAG once, carrying both measures through the same linear transport. Its
results have the same `PseqHistogram` interpretation and rounding-error bound
as two independent `histogram()` calls. The transient states use float64 and
the sink reduction uses extended precision; this keeps memory proportional to
two active grid states while retaining a relative numerical agreement target
of `1e-12` with the independent extended-precision calculations. Use
`histogram()` for an exact-length restriction, which the fused global API does
not accept.

## Discovery and novelty curves

For independent draws from the generated sequence distribution, the expected
number of distinct sequences observed by depth \(n\) is

\[
R(n)=\sum_s\left[1-(1-P(s))^n\right].
\]

The probability that draw \(n\) is a sequence absent from the preceding
draws is

\[
U(n)=\sum_sP(s)(1-P(s))^{n-1}.
\]

Evaluate both curves together by passing every desired depth at once:

```python
draws = np.logspace(0, 20, 121)
curve = analysis.discovery_curve(draws, bins=16384)

richness = curve["expected_richness"]
novelty = curve["novelty_probability"]
```

The counting spectrum is constructed once for the whole batch. Calling
`expected_richness()` repeatedly would reconstruct that spectrum on every
call, so `discovery_curve()` is the appropriate interface for a sweep.
Non-integer depths are allowed for smooth analytical curves, while values must
be finite and at least one.

Small supports use explicitly enumerated exact atoms. Large supports use the
deterministic counting histogram, so `bins` controls the same grid
approximation and pathwise rounding-error bound described above. After the
spectrum exists, the native kernel uses `log1p` and `expm1`; this matters when
\(P(s)\) is far below machine epsilon, where directly subtracting
\(1-(1-P(s))^n\) would incorrectly erase the contribution.

Linear transport on a surprisal grid conserves counting mass and mean
surprisal, but exponentiating the grid locations can make the reconstructed
probabilities sum to slightly more or less than one. Before evaluating a
large-support discovery curve, the API therefore divides those probabilities
by their reconstructed mass. This guarantees the probability identities
`R(1) == U(1) == 1` and keeps novelty in `[0, 1]`. The result reports the raw
value as `spectrum_mass_before_normalization` and whether rescaling occurred as
`spectrum_normalized`, so grid drift remains visible rather than being hidden.

## Choosing an interface

| Question | Interface |
|---|---|
| How many sequences can be generated at each AA length? | `path_count_by_length()` |
| What are global probability or surprisal moments? | `derivatives()`, `moments()`, `cumulants()` |
| How many sequences and how much probability occur at each AA length? | `length_marginals()` |
| What happens under a strong probability tilt? | `log_mellin()`, `tilted_moments()` |
| Which nodes and edges carry a tilted distribution? | `attribution()` |
| How does diversity change as low-weight edges are pruned? | `diversity_under_edge_thresholds()` |
| What are those quantities at each AA length? | `length_derivatives()`, `length_profile()` |
| Can I enumerate every probability atom? | `exact_atoms()` for small supports |
| What does a large probability spectrum look like? | `histogram()`, or `histogram_pair()` for both global measures |
| What smooth PDF/CDF approximates the generated spectrum? | `saddlepoint().pdf_cdf()` |
| Where does one sequence lie in the spectrum? | `position()` |
| How quickly are sequences discovered, and how much novelty remains? | `discovery_curve()` |

Creating the analysis object reads the graph's existing topological structure;
it does not alter the graph. Reuse one analysis object when making several
queries:

```python
analysis = graph.pseq_analysis()
moments = analysis.moments()
by_length = analysis.length_profile()
histogram = analysis.histogram(4096)
both_histograms = analysis.histogram_pair(4096)
```
