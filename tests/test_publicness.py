"""Validation of the analytical publicness (repertoire-occupancy) model."""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from LZGraphs import FlashBackGraph, PublicnessModel
from LZGraphs import _clzgraph as _c


def reference_pmf(p, depths):
    """Brute-force Poisson-binomial by DP convolution, one repertoire at a time.

    O(R^2) and completely independent of the generating-function machinery.
    The detection probability goes through ``log1p``/``expm1`` deliberately:
    spelling it ``1 - (1 - p) ** N`` cancels catastrophically for small ``p``
    and makes the *reference* the inaccurate side of the comparison, by three
    orders of magnitude at ``p = 1e-6``.
    """
    pmf = np.zeros(len(depths) + 1, dtype=np.float64)
    pmf[0] = 1.0
    for depth in depths:
        if p <= 0 or depth <= 0:
            continue
        pi = 1.0 if p >= 1 else float(-np.expm1(depth * np.log1p(-p)))
        pmf[1:] = pmf[1:] * (1.0 - pi) + pmf[:-1] * pi
        pmf[0] *= 1.0 - pi
    return pmf


def naive_aggregate(model, probabilities, weights):
    """The implementation this module exists to prevent.

    Identical generating function, identical inversion, but the PMF is only
    clipped at zero instead of being truncated to the support its closed-form
    moments place the mass in. Used to prove the regression test has teeth.
    """
    depths, multiplicity = model.depth_groups()
    counts = np.zeros(model.k_fft, dtype=np.float64)
    g = np.empty((len(probabilities), model.k_fft), dtype=np.complex128)
    _c.publicness_pgf(np.ascontiguousarray(probabilities, dtype=np.float64),
                      depths, multiplicity, g.view(np.float64).reshape(-1))
    pmf = np.fft.fft(g, axis=-1).real / model.k_fft
    np.clip(pmf, 0.0, None, out=pmf)
    counts += (np.asarray(weights)[:, None] * pmf).sum(axis=0)
    return counts


UNEVEN_DEPTHS = [10, 50, 120, 300, 700, 1500, 4000, 9000, 12, 63, 250, 800]


@pytest.fixture
def uneven_model():
    return PublicnessModel(UNEVEN_DEPTHS)


# ── Exactness against brute force ─────────────────────────────

@pytest.mark.parametrize(
    "p", [0.0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0]
)
def test_matches_bruteforce_dp_convolution(uneven_model, p):
    np.testing.assert_allclose(
        uneven_model.pmf(p), reference_pmf(p, UNEVEN_DEPTHS),
        rtol=0, atol=1e-14,
    )


def test_pmf_is_normalised(uneven_model):
    for p in [0.0, 1e-7, 1e-3, 0.25, 1.0]:
        assert uneven_model.pmf(p).sum() == pytest.approx(1.0, abs=1e-12)


def test_equal_depths_reduce_to_a_binomial():
    depth, n_repertoires = 5_000, 24
    model = PublicnessModel([depth] * n_repertoires)
    assert model.n_depth_groups == 1
    assert model.depth_spread == 1.0

    for p in [1e-6, 1e-4, 1e-3]:
        pi = float(-np.expm1(depth * np.log1p(-p)))
        expected = np.array([
            comb(n_repertoires, k) * pi**k * (1 - pi) ** (n_repertoires - k)
            for k in range(n_repertoires + 1)
        ])
        np.testing.assert_allclose(model.pmf(p), expected, rtol=0, atol=1e-14)


def test_single_repertoire():
    model = PublicnessModel([2_000])
    assert model.n_repertoires == 1
    assert model.k_fft == 2
    pi = float(-np.expm1(2_000 * np.log1p(-1e-4)))
    np.testing.assert_allclose(model.pmf(1e-4), [1 - pi, pi], atol=1e-15)


def test_probability_zero_and_one_are_deltas(uneven_model):
    at_zero = uneven_model.pmf(0.0)
    assert at_zero[0] == 1.0
    assert at_zero[1:].max() == 0.0

    at_one = uneven_model.pmf(1.0)
    assert at_one[-1] == pytest.approx(1.0, abs=1e-14)
    assert at_one[:-1].max() == pytest.approx(0.0, abs=1e-14)


def test_zero_depth_repertoires_never_detect():
    """A repertoire that contributed nothing still counts toward the cohort."""
    model = PublicnessModel([0, 0, 5_000])
    assert model.n_repertoires == 3
    pi = float(-np.expm1(5_000 * np.log1p(-1e-4)))
    np.testing.assert_allclose(model.pmf(1e-4), [1 - pi, pi, 0.0, 0.0],
                               atol=1e-15)


# ── Closed-form moments ───────────────────────────────────────

def test_closed_form_moments_match_the_pmf(uneven_model):
    for p in [1e-8, 1e-5, 1e-3, 0.05, 0.4]:
        mean, variance = uneven_model.moments(p)
        pmf = uneven_model.pmf(p)
        levels = np.arange(pmf.size)
        assert mean == pytest.approx(float(np.dot(levels, pmf)), rel=1e-12)
        assert variance == pytest.approx(
            float(np.dot(levels * levels, pmf) - mean**2), rel=1e-10
        )


def test_moments_are_vectorised(uneven_model):
    probabilities = np.array([[1e-6, 1e-4], [1e-3, 1e-2]])
    mean, variance = uneven_model.moments(probabilities)
    assert mean.shape == variance.shape == probabilities.shape
    for index in np.ndindex(probabilities.shape):
        scalar_mean, scalar_variance = uneven_model.moments(
            probabilities[index]
        )
        assert mean[index] == pytest.approx(scalar_mean)
        assert variance[index] == pytest.approx(scalar_variance)


# ── The roundoff floor this module exists to prevent ──────────

def test_far_tail_does_not_floor_under_huge_multiplicities():
    """Regression test for the rectified-roundoff floor.

    A spectrum whose multiplicities reach 1e29 at probabilities whose
    occupancy mass sits within a couple of levels of zero. Every level above
    what the closed-form moments can reach must predict exactly nothing. The
    inversion roundoff there is around 1e-16 per level, which clipping alone
    turns into 1e13 spurious sequences, so a plain clip does not merely blur
    the tail: it replaces it.
    """
    rng = np.random.default_rng(20260808)
    depths = np.rint(np.exp(rng.normal(np.log(50_000), 0.6, 1024)))
    model = PublicnessModel(depths)

    # Mid-surprisal atoms carry nearly all of D0. Their occupancy mean is far
    # below one, so the true count above a handful of levels is zero.
    probabilities = np.logspace(-30, -9, 512)
    weights = np.logspace(20, 29, 512)
    result = model.expected_counts(probabilities, weights)
    counts = result["expected_counts"]

    mean, variance = model.moments(probabilities)
    reach = int(np.ceil(np.max(mean + 40.0 * np.sqrt(variance)))) + 257
    assert reach < model.n_repertoires, "fixture must leave an unreachable tail"

    # 1. Nothing at all above the support the closed-form moments allow.
    assert counts[reach:].max() == 0.0

    # 2. The prediction decays instead of flattening, over every level that
    #    carries mass a double can resolve.
    assert np.all(np.diff(counts[:11]) < 0)

    # 3. Inside the absolute floor, where roundoff is retained by design, the
    #    residue stays sixteen orders below the real counts. See the module
    #    docstring: it cannot be removed, only bounded.
    assert counts[16:reach].max() < 1e-15 * counts[0]

    # 4. Every atom's truncated PMF still holds all of its mass, so none of
    #    that decay came from throwing real mass away.
    assert result["min_retained_mass"] == pytest.approx(1.0, abs=1e-9)

    # 5. The same inversion with a plain clip really does floor: it predicts
    #    tens of trillions of sequences at every level, all the way to the top
    #    of the transform. Without this the assertions above prove nothing.
    naive = naive_aggregate(model, probabilities, weights)
    assert naive[reach:].min() > 1e12
    assert naive[-1] > 1e12


def test_truncation_survives_a_broad_probability_range():
    """Atoms spanning delta-at-zero to nearly-certain, aggregated together."""
    model = PublicnessModel(np.rint(np.linspace(1_000, 200_000, 512)))
    probabilities = np.concatenate([
        np.logspace(-90, -40, 64),      # true PMF is a delta at zero
        np.logspace(-12, -6, 64),       # a few levels of real spread
        np.array([1e-3, 1e-2, 0.5]),    # nearly every repertoire
    ])
    weights = np.concatenate([
        np.full(64, 1e28), np.full(64, 1e12), np.full(3, 1.0)
    ])
    result = model.expected_counts(probabilities, weights)

    assert result["min_retained_mass"] == pytest.approx(1.0, abs=1e-9)
    # Mass is conserved: every sequence lands in exactly one publicness level.
    assert result["expected_total"] == pytest.approx(
        float(weights.sum()), rel=1e-12
    )
    # The delta-at-zero atoms are seen in no repertoire at all, and they are
    # sixteen orders heavier than everything else, so level 0 is exactly them.
    assert result["expected_counts"][0] == pytest.approx(64e28, rel=1e-12)
    # Only the three high-probability atoms can reach the top level, and the
    # p = 0.5 one is certain to be in every repertoire.
    assert 1.0 <= result["expected_counts"][-1] <= 3.0


def test_mass_check_fails_loudly_when_the_window_is_too_narrow(uneven_model):
    model = PublicnessModel(UNEVEN_DEPTHS, tail_sigma=0.0, tail_floor=0.0)
    with pytest.raises(ValueError, match="occupancy mass"):
        model.pmf(1e-3)
    # The same atom is fine at the default window.
    assert uneven_model.pmf(1e-3).sum() == pytest.approx(1.0, abs=1e-12)


# ── Binning and aggregation ───────────────────────────────────

def test_aggregate_is_the_multiplicity_weighted_sum_of_atoms(uneven_model):
    probabilities = np.array([1e-5, 1e-3, 0.02])
    weights = np.array([1e9, 4.0, 2.5])
    counts = uneven_model.expected_counts(probabilities, weights)[
        "expected_counts"
    ]
    expected = sum(
        w * uneven_model.pmf(p) for p, w in zip(probabilities, weights)
    )
    np.testing.assert_allclose(counts, expected, rtol=1e-12)


def test_explicit_and_integer_bin_edges(uneven_model):
    probabilities = np.array([1e-4, 1e-3])
    weights = np.array([100.0, 7.0])
    per_level = uneven_model.expected_counts(probabilities, weights)
    coarse = uneven_model.expected_counts(
        probabilities, weights, levels=[0, 1, 4, 13]
    )
    np.testing.assert_allclose(coarse["edges"], [0, 1, 4, 13])
    np.testing.assert_allclose(
        coarse["expected_counts"],
        [
            per_level["expected_counts"][0],
            per_level["expected_counts"][1:4].sum(),
            per_level["expected_counts"][4:13].sum(),
        ],
        rtol=1e-12,
    )

    binned = uneven_model.expected_counts(probabilities, weights, levels=4)
    assert binned["edges"].size == 5
    assert binned["expected_total"] == pytest.approx(
        per_level["expected_total"], rel=1e-12
    )


def test_zero_multiplicity_atoms_are_dropped(uneven_model):
    probabilities = np.array([1e-4, 1e-3, 0.5])
    weights = np.array([3.0, 0.0, 0.0])
    result = uneven_model.expected_counts(probabilities, weights)
    assert result["n_atoms"] == 1
    np.testing.assert_allclose(
        result["expected_counts"], 3.0 * uneven_model.pmf(1e-4), rtol=1e-12
    )


def test_depth_binning_is_exact_below_the_bin_count():
    exact = PublicnessModel(UNEVEN_DEPTHS, depth_bins=64)
    assert exact.n_depth_groups == len(UNEVEN_DEPTHS)
    assert exact.depth_spread == 1.0

    binned = PublicnessModel(UNEVEN_DEPTHS, depth_bins=4)
    assert binned.n_depth_groups <= 4
    assert binned.depth_spread > 1.0
    assert binned.n_repertoires == exact.n_repertoires


# ── Input validation ──────────────────────────────────────────

def test_rejects_a_transform_too_short_to_hold_the_cohort():
    with pytest.raises(ValueError, match="wraps around"):
        PublicnessModel([100] * 32, k_fft=32)
    with pytest.raises(ValueError, match="even"):
        PublicnessModel([100] * 32, k_fft=33)


@pytest.mark.parametrize(
    "depths, message",
    [
        ([], "at least one repertoire"),
        ([100, -1], "non-negative"),
        ([100, np.inf], "finite"),
    ],
)
def test_rejects_bad_depths(depths, message):
    with pytest.raises(ValueError, match=message):
        PublicnessModel(depths)


def test_rejects_bad_spectra(uneven_model):
    with pytest.raises(ValueError, match="same length"):
        uneven_model.expected_counts([1e-3, 1e-4], [1.0])
    with pytest.raises(ValueError, match="non-negative"):
        uneven_model.expected_counts([1e-3], [-1.0])
    with pytest.raises(ValueError, match="no mass"):
        uneven_model.expected_counts([1e-3], [0.0])
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        uneven_model.expected_counts([1.5], [1.0])


def test_rejects_bad_bin_edges(uneven_model):
    for levels, message in [
        ([0], "at least two"),
        ([0.5, 3], "integers"),
        ([0, 5, 2], "non-decreasing"),
        ([0, 10_000], r"within \[0, 16\]"),
        (0, "positive bin count"),
    ]:
        with pytest.raises(ValueError, match=message):
            uneven_model.expected_counts([1e-3], [1.0], levels=levels)


def test_repr_reports_the_cohort(uneven_model):
    text = repr(uneven_model)
    assert "n_repertoires=12" in text
    assert "k_fft=16" in text


# ── Integration with the p-sequence spectrum ──────────────────

@pytest.fixture
def graph():
    return FlashBackGraph(
        [
            "CASSLGIRRT", "CASSLGYEQYF", "CASSLEPSGGTDTQYF",
            "CASSDTSGGTDTQYF", "CASSFGQGSYEQYF", "CASSQETQYF",
        ],
        abundances=[7, 5, 3, 2, 2, 1],
    )


def test_publicness_distribution_conserves_the_counting_spectrum(graph):
    analysis = graph.pseq_analysis()
    result = analysis.publicness_distribution([1_000, 5_000, 20_000, 50_000])

    assert result["method"] == "exact_atoms"
    assert result["n_repertoires"] == 4
    assert result["edges"].size == 6           # levels 0..4, so five edges
    assert result["min_retained_mass"] == pytest.approx(1.0, abs=1e-9)
    # Every path in the DAG lands in exactly one publicness level.
    assert result["expected_total"] == pytest.approx(
        float(graph.path_count), rel=1e-9
    )


def test_publicness_distribution_matches_a_direct_atom_sum(graph):
    analysis = graph.pseq_analysis()
    depths = [2_000, 9_000, 40_000]
    atoms = analysis.exact_atoms()
    model = PublicnessModel(depths)
    expected = sum(model.pmf(float(p)) for p in atoms.probabilities)

    result = analysis.publicness_distribution(depths)
    np.testing.assert_allclose(
        result["expected_counts"], expected, rtol=1e-11, atol=1e-12
    )


def test_publicness_rises_with_depth(graph):
    analysis = graph.pseq_analysis()
    shallow = analysis.publicness_distribution([1] * 8)["expected_counts"]
    deep = analysis.publicness_distribution([500] * 8)["expected_counts"]
    assert deep[-1] > shallow[-1]
    assert deep[0] < shallow[0]


def test_publicness_distribution_accepts_coarse_levels(graph):
    analysis = graph.pseq_analysis()
    depths = [1_000, 5_000, 20_000, 50_000]
    fine = analysis.publicness_distribution(depths)
    coarse = analysis.publicness_distribution(depths, levels=[0, 1, 5])
    assert coarse["expected_counts"][0] == pytest.approx(
        fine["expected_counts"][0], rel=1e-12
    )
    assert coarse["expected_counts"][1] == pytest.approx(
        fine["expected_counts"][1:].sum(), rel=1e-12
    )


def test_publicness_is_not_the_frequency_spectrum(graph):
    """The two answer different questions and must not be conflated.

    ``expected_frequency_spectrum`` counts how many times a sequence appears
    inside one pool of n draws. ``publicness_distribution`` counts how many
    separate repertoires contain it at all.
    """
    analysis = graph.pseq_analysis()
    frequency = analysis.expected_frequency_spectrum(4_000, 4)
    publicness = analysis.publicness_distribution([1_000] * 4)
    assert frequency["expected_counts"].shape == (5,)
    assert publicness["expected_counts"].shape == (5,)
    assert not np.allclose(
        frequency["expected_counts"], publicness["expected_counts"]
    )


# ── Atoms below the transform's precision ─────────────────────


def test_tiny_atoms_do_not_comb_across_odd_levels():
    """Regression test for the odd-harmonic quantisation comb.

    An atom at Pseq 1e-21 has a mean occupancy of order 1e-13 across a
    realistic cohort, so its generating function is 1 + mu*(z-1) with mu far
    below the double-precision epsilon of 1.0. The perturbation is destroyed
    by cancellation and the inverse transform returns rounding residue.

    That residue is not noise-like. Rounding is an odd nonlinearity applied to
    a cosine, so it concentrates on odd harmonics: the floor at odd occupancy
    levels runs about 1e12 times the floor at even ones. Scaled by a spectrum
    multiplicity of 1e17 it reached hundreds of predicted sequences at
    occupancy levels the atom cannot reach, in a comb across alternate levels
    that reads as structure in any (probability, occupancy) map.
    """
    rng = np.random.default_rng(20260813)
    depths = np.rint(np.exp(rng.normal(np.log(26_000), 0.9, 4096)))
    model = PublicnessModel(depths, depth_bins=64)

    counts = model.expected_counts([1e-21], [1.8e17],
                                   levels=np.arange(0, 40, dtype=np.int64))
    per_level = np.asarray(counts["expected_counts"])

    # Levels 2 and up are effectively unreachable: mu**2/2 times the
    # multiplicity is well under a millionth of a sequence. The old comb put
    # hundreds there.
    assert per_level[2:].max() < 1e-6, (
        f"unreachable levels predict up to {per_level[2:].max():.6g} "
        f"sequences"
    )

    # And no parity structure: the comb was ~1e12x heavier on odd levels.
    odd, even = per_level[3::2].sum(), per_level[4::2].sum()
    assert odd <= max(1e-6, 10.0 * even), (
        f"odd-level total {odd:.6g} against even-level total {even:.6g}: "
        f"the residue still has the parity signature"
    )


def test_tiny_atom_level_one_is_not_halved():
    """``pmf[1]/mu`` must stay 1, not decay to 1/2 in the deep tail.

    Separate failure from the comb above and it bites earlier. The real part
    of the perturbation, mu*(cos(theta) - 1), is annihilated against 1.0 while
    the imaginary part mu*sin(theta) keeps full precision; inverting an
    imaginary-only transform yields exactly mu/2. So the singleton column, the
    one that carries essentially all the mass of a rare atom, was under-
    predicted by up to a factor of two.
    """
    rng = np.random.default_rng(20260813)
    depths = np.rint(np.exp(rng.normal(np.log(26_000), 0.9, 2048)))
    model = PublicnessModel(depths, depth_bins=64)

    for p in (1e-18, 1e-21, 1e-24, 1e-27):
        mu, _ = model.moments(p)
        counts = model.expected_counts([p], [1.0],
                                       levels=np.arange(0, 8, dtype=np.int64))
        singleton = float(np.asarray(counts["expected_counts"])[1])
        assert singleton == pytest.approx(float(mu), rel=1e-9), (
            f"at Pseq {p:g} the singleton column is {singleton:.6g} against a "
            f"mean occupancy of {float(mu):.6g} "
            f"(ratio {singleton / float(mu):.6f})"
        )


def test_tiny_and_transform_paths_agree_at_the_crossover():
    """The two inversions must not disagree where they meet.

    SMALL_MU is chosen so both methods are accurate at the boundary. If that
    ever stops being true the aggregate develops a step at a threshold no
    caller can see.
    """
    from LZGraphs._publicness import SMALL_MU

    rng = np.random.default_rng(4242)
    depths = np.rint(np.exp(rng.normal(np.log(26_000), 0.9, 1024)))
    model = PublicnessModel(depths, depth_bins=32)
    levels = np.arange(0, 12, dtype=np.int64)

    # Bracket the crossover: find probabilities whose mean occupancy sits just
    # either side of SMALL_MU.
    lo, hi = 1e-30, 1e-3
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        if float(model.moments(mid)[0]) < SMALL_MU:
            lo = mid
        else:
            hi = mid

    # Straddle the crossover closely. A wide bracket compares two genuinely
    # different atoms and measures that difference rather than the seam.
    below = np.asarray(model.expected_counts([lo * (1 - 1e-4)], [1e12],
                                             levels=levels)["expected_counts"])
    above = np.asarray(model.expected_counts([hi * (1 + 1e-4)], [1e12],
                                             levels=levels)["expected_counts"])
    # Occupancy 0 and 1 carry all the representable mass at this mean; level 2
    # holds under one sequence in 1e12 and is the one place the transform is
    # itself imprecise, being only ~3 orders above epsilon there.
    assert below[:2] == pytest.approx(above[:2], rel=1e-3), (
        f"closed form gives {below[:2]} just below the crossover, transform "
        f"gives {above[:2]} just above"
    )
