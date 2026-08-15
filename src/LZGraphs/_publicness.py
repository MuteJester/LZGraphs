"""Analytical publicness: the Poisson-binomial repertoire-occupancy PMF.

Given a cohort of R repertoires whose sampling depths N_1..N_R (distinct
sequences contributed) are known, a sequence with model probability ``p`` is
present in repertoire ``r`` with probability

    pi_r = 1 - (1 - p)**N_r.

The depths differ, so the number of repertoires containing the sequence is
Poisson-binomial rather than binomial. Repertoires are collapsed into a few
dozen depth groups carrying multiplicities ``m_b``, and the probability
generating function of the occupancy count is

    G(z) = prod_b (1 - pi_b + pi_b * z)**m_b,

whose coefficients are recovered by evaluating G at the K-th roots of unity
and taking a **forward** DFT divided by K. The inverse transform returns the
same coefficients reflected about zero, which is a silent off-by-everything;
K must exceed R so that no mass wraps around.

Aggregating over a whole graph, the predicted number of sequences at each
publicness level is the sum over the p-sequence spectrum of (multiplicity at
that probability) times (occupancy PMF of that probability).

THE ROUNDOFF FLOOR, AND WHY THE TRUNCATION IS NOT OPTIONAL
----------------------------------------------------------
That inversion is exact in arithmetic and treacherous in floating point, in a
way specific to this application.

Every atom's PMF is scaled by its multiplicity in the counting spectrum. On a
foundation-scale graph those multiplicities reach 1.2e29 and sum to a D0 of
1.9e32. Atoms deep in the tail have ``p`` of order 1e-80, so their true PMF is
a delta at zero: those sequences are detected in no repertoire at all. But the
DFT returns that delta with relative roundoff of order 1e-16 smeared across
all 65,536 output bins, and 1e-16 times 1e32 is 1e16.

Clipping the PMF at zero -- the natural response to the tiny negatives that
appear -- keeps the positive half of that noise and rectifies it into a floor.
Summed over 15,574 atoms it produced roughly 2,000 spurious sequences in
*every* publicness bin, flattening the predicted tail instead of letting it
decay: 2,114 predicted in the top bin where 0 were observed, and where only
135 atoms, carrying 108 sequences between them, could physically reach at all.

The fix is that both moments of a Poisson-binomial are available in closed
form without touching the PMF,

    mu     = sum_b m_b * pi_b
    sigma2 = sum_b m_b * pi_b * (1 - pi_b),

so the retained support is truncated to ``[mu - T*sigma, mu + T*sigma]``,
widened by an absolute floor for the degenerate ``mu ~ 0`` atoms where sigma
is also ~ 0 but the first few integers still carry real mass. Everything
outside is zeroed *before* negatives are clipped, so the far-tail roundoff is
discarded rather than rectified. At ``T = 40`` the Chernoff bound on the
discarded mass is below 1e-300, so nothing real is lost.

Do not "simplify" the truncation away. The result looks entirely plausible and
is wrong by fifteen orders of magnitude in the tail. Each atom's retained mass
is checked against 1 so that a truncation mistake fails loudly rather than
quietly reshaping a tail.

WHAT THE TRUNCATION DOES NOT REMOVE
-----------------------------------
The absolute floor retains levels 0..``TAIL_FLOOR`` for every atom, so *inside*
that window a large multiplicity still rectifies inversion roundoff. The
truncation is a statement about the support, and it cannot help where the
transform never had the digits to begin with.

That case is handled separately rather than tolerated: atoms with mean
occupancy below ``SMALL_MU`` bypass the transform entirely and are inverted in
closed form. The residue they used to contribute was not a harmless floor. It
sat on odd occupancy levels only, so it surfaced as a comb of spurious
predictions across alternate levels, at multiplicities large enough to reach
hundreds of sequences per level, at occupancy a 1e-21 atom cannot reach even
in principle. Read as a ratio against observation it inverted the sign of the
comparison: cells the model should call impossible instead reported the model
over-predicting by up to 64x.

Above ``SMALL_MU`` the transform has the digits and the moment truncation
governs, so between them every atom is either represented exactly or bounded
by its own moments.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _clzgraph as _c

#: Half-width of the retained support, in standard deviations.
TAIL_SIGMA = 40.0

#: Absolute floor on the retained half-width, in occupancy levels. Covers the
#: degenerate ``mu ~ 0`` atoms, where sigma is ~ 0 but levels 0, 1, 2 ... still
#: hold real mass.
TAIL_FLOOR = 256.0

#: Mean occupancy below which the transform is bypassed for a closed form.
#:
#: The generating function of one atom is ``1 + mu*(z-1)``. Once ``mu`` falls
#: far below the double-precision epsilon of 1.0, that sum loses the whole
#: perturbation and the inverse transform returns quantisation residue rather
#: than a distribution. The residue is not even unbiased: rounding is an odd
#: nonlinearity applied to a cosine, so it lands on odd harmonics, and the
#: floor at odd occupancy levels runs about 1e12 times the floor at even ones.
#: Scaled by spectrum multiplicities of 1e17 and up, that becomes hundreds of
#: predicted sequences at occupancy levels a 1e-21 atom can never reach, in a
#: comb across alternate levels. The level-1 column degrades first and
#: separately: ``pmf[1]/mu`` falls to exactly 1/2 by ``mu ~ 3e-15``, because
#: the real part of the perturbation is annihilated while the imaginary part
#: survives intact.
#:
#: 1e-6 sits well above the onset of either failure, and both methods are
#: accurate there, so the crossover is continuous.
SMALL_MU = 1e-6

#: Occupancy levels retained by the closed form. ``mu**k / k!`` at ``SMALL_MU``
#: is below 1e-30 by k=5, so nothing above this carries representable mass.
_TINY_LEVELS = 5

#: A truncated PMF must still sum to one within this tolerance.
MASS_TOL = 1e-9

#: Doubles per PGF batch. One batch holds this many PMF entries and twice as
#: many generating-function entries, so roughly 48 MB at the default.
_CHUNK_DOUBLES = 1 << 21


def _next_power_of_two_above(n: int) -> int:
    k = 2
    while k <= n:
        k *= 2
    return k


def _depth_groups(
    depths: Any, depth_bins: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Collapse repertoire depths into (representative depth, multiplicity).

    The PGF product runs over these groups rather than over every repertoire.
    Grouping is *exact* whenever the cohort has at most ``depth_bins`` distinct
    depths; only beyond that are depths quantile-binned, with the geometric
    mean as the representative because detection probability is driven by
    log-depth over most of the range, so it distorts ``1 - (1-p)**N`` least.
    """
    d = np.asarray(depths, dtype=np.float64).reshape(-1)
    if d.size == 0:
        raise ValueError("depths must name at least one repertoire")
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError("every depth must be finite and non-negative")

    unique, counts = np.unique(d, return_counts=True)
    if unique.size <= depth_bins:
        return unique, counts.astype(np.float64), 1.0

    # Zero-depth repertoires contribute pi = 0 at every p; they are held out of
    # the quantile binning so that a geometric mean never sees a zero, and are
    # carried as their own group so the repertoire count still adds up.
    n_zero = int(np.count_nonzero(d <= 0))
    positive = np.sort(d[d > 0])
    edges = np.unique(
        np.quantile(positive, np.linspace(0.0, 1.0, depth_bins + 1))
    )
    index = np.clip(np.digitize(positive, edges[1:-1]), 0, edges.size - 2)

    representatives: list[float] = []
    multiplicities: list[float] = []
    spread = 1.0
    for b in range(edges.size - 1):
        group = positive[index == b]
        if group.size == 0:
            continue
        representatives.append(float(np.exp(np.mean(np.log(group)))))
        multiplicities.append(float(group.size))
        spread = max(spread, float(group[-1] / group[0]))
    if n_zero:
        representatives.append(0.0)
        multiplicities.append(float(n_zero))

    return (
        np.asarray(representatives, dtype=np.float64),
        np.asarray(multiplicities, dtype=np.float64),
        spread,
    )


class PublicnessModel:
    """Poisson-binomial occupancy model for one fixed cohort of depths.

    ``depths`` is the per-repertoire sampling depth, in distinct sequences
    contributed. The model is independent of any graph: it turns a sequence
    probability into the distribution of how many repertoires contain that
    sequence, and :meth:`expected_counts` aggregates that over a whole
    p-sequence spectrum.

    Read the module docstring before changing ``tail_sigma`` or
    ``tail_floor``. They are what keeps DFT roundoff, amplified by spectrum
    multiplicities of up to 1e29, out of the predicted tail.
    """

    def __init__(
        self,
        depths: Any,
        *,
        depth_bins: int = 64,
        k_fft: int | None = None,
        tail_sigma: float = TAIL_SIGMA,
        tail_floor: float = TAIL_FLOOR,
        mass_tol: float = MASS_TOL,
    ) -> None:
        if depth_bins < 1:
            raise ValueError("depth_bins must be positive")
        if tail_sigma < 0 or tail_floor < 0:
            raise ValueError("tail_sigma and tail_floor must be non-negative")
        if mass_tol <= 0:
            raise ValueError("mass_tol must be positive")

        self._depths, self._multiplicity, self._spread = _depth_groups(
            depths, depth_bins
        )
        self._n_repertoires = int(round(float(np.sum(self._multiplicity))))

        if k_fft is None:
            k_fft = _next_power_of_two_above(self._n_repertoires)
        if k_fft <= self._n_repertoires:
            raise ValueError(
                f"k_fft={k_fft} must exceed the {self._n_repertoires} "
                "repertoires, or occupancy mass wraps around the transform"
            )
        if k_fft % 2:
            raise ValueError("k_fft must be even")

        self._k_fft = int(k_fft)
        self.tail_sigma = float(tail_sigma)
        self.tail_floor = float(tail_floor)
        self.mass_tol = float(mass_tol)

    # ── Cohort description ────────────────────────────────────

    @property
    def n_repertoires(self) -> int:
        """Number of repertoires in the cohort."""
        return self._n_repertoires

    @property
    def n_depth_groups(self) -> int:
        """Number of factors in the generating-function product."""
        return int(self._depths.size)

    @property
    def depth_spread(self) -> float:
        """Widest within-group ratio of depths. ``1.0`` means exact grouping."""
        return self._spread

    @property
    def k_fft(self) -> int:
        """Number of roots of unity used to invert the generating function."""
        return self._k_fft

    def depth_groups(self) -> tuple[np.ndarray, np.ndarray]:
        """Representative depths and their multiplicities."""
        return self._depths.copy(), self._multiplicity.copy()

    # ── Per-sequence quantities ───────────────────────────────

    def moments(self, p: Any) -> tuple[Any, Any]:
        """Closed-form mean and variance of the occupancy count.

        Neither touches the PMF, which is exactly why they can be trusted to
        say which part of the inverted PMF is signal.
        """
        values = np.asarray(p, dtype=np.float64)
        flat = np.ascontiguousarray(values.reshape(-1))
        mean = np.empty(flat.size, dtype=np.float64)
        variance = np.empty(flat.size, dtype=np.float64)
        _c.publicness_moments(
            flat, self._depths, self._multiplicity, mean, variance
        )
        if np.ndim(p) == 0:
            return float(mean[0]), float(variance[0])
        return mean.reshape(values.shape), variance.reshape(values.shape)

    def pmf(self, p: float) -> np.ndarray:
        """Occupancy PMF of one sequence probability, over levels 0..R.

        Truncated, clipped and mass-checked exactly as the aggregate is, so
        this is the single-atom view of what :meth:`expected_counts` sums.
        """
        edges = np.arange(self._n_repertoires + 2, dtype=np.float64)
        counts, _, _, _ = self._aggregate(
            np.asarray([float(p)]), np.ones(1), edges
        )
        return counts

    def bin_mass(self, p: float, edges: Any) -> np.ndarray:
        """Probability the occupancy count falls in each half-open bin."""
        resolved = self._resolve_edges(edges)
        counts, _, _, _ = self._aggregate(
            np.asarray([float(p)]), np.ones(1), resolved
        )
        return counts

    # ── Aggregation over a spectrum ───────────────────────────

    def expected_counts(
        self,
        probabilities: Any,
        multiplicities: Any,
        *,
        levels: Any = None,
    ) -> dict[str, Any]:
        """Predicted sequences per publicness bin, over a whole spectrum.

        ``probabilities`` and ``multiplicities`` are a p-sequence spectrum:
        one probability atom and the number of distinct sequences carrying it.
        Atoms of zero multiplicity are dropped before any work is done, which
        matters because a deterministic surprisal grid is mostly empty.

        ``levels`` selects the publicness binning:

        * ``None`` (default) gives one bin per occupancy level, 0..R.
        * an ``int`` gives that many equal-width bins spanning 0..R+1, merged
          where rounding would produce empty ones.
        * a sequence gives explicit non-decreasing integer bin edges, read as
          half-open ranges ``[edges[i], edges[i+1])``.
        """
        p = np.ascontiguousarray(
            np.asarray(probabilities, dtype=np.float64).reshape(-1)
        )
        w = np.ascontiguousarray(
            np.asarray(multiplicities, dtype=np.float64).reshape(-1)
        )
        if p.size != w.size:
            raise ValueError(
                "probabilities and multiplicities must be the same length"
            )
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("every multiplicity must be finite and non-negative")

        keep = w > 0
        p, w = np.ascontiguousarray(p[keep]), np.ascontiguousarray(w[keep])
        if p.size == 0:
            raise ValueError("the spectrum carries no mass")

        edges = self._resolve_edges(levels)
        counts, mean, variance, retained = self._aggregate(p, w, edges)
        return {
            "edges": edges.astype(np.int64),
            "expected_counts": counts,
            "expected_total": float(np.sum(counts)),
            "atom_mean": mean,
            "atom_variance": variance,
            "min_retained_mass": float(np.min(retained)),
            "n_atoms": int(p.size),
            "n_repertoires": self._n_repertoires,
            "n_depth_groups": self.n_depth_groups,
            "depth_spread": self._spread,
            "k_fft": self._k_fft,
        }

    # ── Internals ─────────────────────────────────────────────

    def _resolve_edges(self, levels: Any) -> np.ndarray:
        if levels is None:
            return np.arange(self._n_repertoires + 2, dtype=np.float64)
        if np.ndim(levels) == 0:
            n_bins = int(levels)
            if n_bins < 1:
                raise ValueError("levels must be a positive bin count")
            grid = np.linspace(0.0, self._n_repertoires + 1.0, n_bins + 1)
            return np.unique(np.rint(grid))

        edges = np.asarray(levels, dtype=np.float64).reshape(-1)
        if edges.size < 2:
            raise ValueError("explicit bin edges need at least two entries")
        if np.any(edges != np.floor(edges)):
            raise ValueError("bin edges must be integers")
        if np.any(np.diff(edges) < 0):
            raise ValueError("bin edges must be non-decreasing")
        if edges[0] < 0 or edges[-1] > self._k_fft:
            raise ValueError(f"bin edges must lie within [0, {self._k_fft}]")
        return edges

    def _aggregate(
        self,
        p: np.ndarray,
        w: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        counts = np.zeros(edges.size - 1, dtype=np.float64)
        mean = np.empty(p.size, dtype=np.float64)
        variance = np.empty(p.size, dtype=np.float64)
        retained = np.empty(p.size, dtype=np.float64)
        _c.publicness_moments(p, self._depths, self._multiplicity,
                              mean, variance)

        # Atoms whose mean occupancy is far below machine precision cannot go
        # through the transform at all: their generating function is
        # 1 + mu(z-1) with mu ~ 1e-13, and adding that to 1.0 in double
        # precision destroys it. See SMALL_MU for what the transform returns
        # instead. They are inverted in closed form.
        tiny = mean < SMALL_MU
        big = np.flatnonzero(~tiny)

        step = max(1, _CHUNK_DOUBLES // self._k_fft)
        for start in range(0, big.size, step):
            idx = big[start:start + step]
            g = np.empty((idx.size, self._k_fft), dtype=np.complex128)
            _c.publicness_pgf(np.ascontiguousarray(p[idx]), self._depths,
                              self._multiplicity,
                              g.view(np.float64).reshape(-1))
            # Forward transform divided by K. np.fft.ifft would return the
            # same coefficients reflected about zero, which reads as a
            # plausible distribution and is wrong at every level but 0.
            pmf = np.ascontiguousarray(
                np.fft.fft(g, axis=-1).real / self._k_fft
            )
            chunk_retained = np.empty(idx.size, dtype=np.float64)
            _c.publicness_accumulate(
                pmf, np.ascontiguousarray(mean[idx]),
                np.ascontiguousarray(variance[idx]),
                np.ascontiguousarray(w[idx]),
                edges, counts, chunk_retained,
                self.tail_sigma, self.tail_floor, self.mass_tol,
            )
            retained[idx] = chunk_retained

        if tiny.any():
            self._accumulate_tiny(mean[tiny], w[tiny], edges, counts,
                                  retained, np.flatnonzero(tiny))
        return counts, mean, variance, retained

    def _accumulate_tiny(
        self,
        mu: np.ndarray,
        w: np.ndarray,
        edges: np.ndarray,
        counts: np.ndarray,
        retained: np.ndarray,
        where: np.ndarray,
    ) -> None:
        """Closed-form occupancy for atoms the transform cannot represent.

        Below ``SMALL_MU`` every per-repertoire detection probability is
        itself minute, so the Poisson-binomial count collapses to a Poisson
        with the same mean. The approximation error is bounded by Le Cam:
        total variation is at most ``sum_r pi_r**2 <= mu * max_r pi_r``, which
        at ``mu = 1e-6`` is below 1e-25 and shrinks quadratically from there.
        That is exact to double precision, and it is the regime where the
        transform returns nothing but rectified quantisation residue.

        Only the first few levels can carry mass: ``mu**k / k!`` at
        ``mu = SMALL_MU`` is already below 1e-30 by ``k = 5``, so levels above
        ``_TINY_LEVELS`` are exactly zero rather than nearly zero.
        """
        # Forward recurrence P(k) = P(k-1) * mu / k rather than
        # exp(-mu + k*log(mu) - log(k!)), which is undefined at mu = 0. A
        # probability-zero atom is a legitimate input and must give a delta at
        # occupancy zero, not a NaN.
        pmf = np.empty((mu.size, _TINY_LEVELS + 1), dtype=np.float64)
        pmf[:, 0] = np.exp(-mu)
        for k in range(1, _TINY_LEVELS + 1):
            pmf[:, k] = pmf[:, k - 1] * mu / k

        total = pmf.sum(axis=1)
        if not np.all(np.abs(total - 1.0) <= self.mass_tol):
            worst = int(np.argmax(np.abs(total - 1.0)))
            raise ValueError(
                f"closed-form occupancy for a mu={mu[worst]:.6g} atom retains "
                f"{total[worst]:.12f} of its mass over levels "
                f"0..{_TINY_LEVELS}; SMALL_MU and _TINY_LEVELS disagree"
            )
        retained[where] = total

        # Each level is a single integer, so it lands in exactly one bin.
        n_bins = edges.size - 1
        for k in range(_TINY_LEVELS + 1):
            b = int(np.searchsorted(edges, k, side="right")) - 1
            if 0 <= b < n_bins:
                counts[b] += float(np.dot(w, pmf[:, k]))

    def __repr__(self) -> str:
        return (
            f"PublicnessModel(n_repertoires={self._n_repertoires}, "
            f"n_depth_groups={self.n_depth_groups}, "
            f"depth_spread={self._spread:.3f}, k_fft={self._k_fft})"
        )
