"""Exhaustive validation of sampling-free FlashBack p-sequence analytics."""

from __future__ import annotations

import random
from math import comb

import numpy as np
import pytest

from LZGraphs import (
    FlashBackGraph,
    FlashBackPseqAnalysis,
    PseqAtoms,
    PseqHistogram,
    PseqSaddlepoint,
    flashback_reverse,
)


def reference_path_count(graph):
    """Independent pure-Python bigint DP, to check the native counter."""
    csr = graph.adjacency_csr()
    row = csr["row_offsets"]
    col = csr["col_indices"]
    n_nodes = len(row) - 1
    indegree = [0] * n_nodes
    for target in col:
        indegree[int(target)] += 1
    roots = [i for i, label in enumerate(graph.all_nodes) if label.startswith("@")]
    assert len(roots) == 1
    counts = [0] * n_nodes
    counts[roots[0]] = 1
    queue = [i for i in range(n_nodes) if indegree[i] == 0]
    order = []
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        order.append(u)
        for e in range(int(row[u]), int(row[u + 1])):
            v = int(col[e])
            counts[v] += counts[u]
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    assert len(order) == n_nodes, "graph is not a DAG"
    return sum(counts[u] for u in range(n_nodes) if row[u] == row[u + 1])

PATH_A = [
    "@$_1{0}",
    "AE_1{1}",
    "BD_1{2}",
    "CK_1{3}",
    "M_0{4}",
]
PATH_B = [
    "@$_1{0}",
    "AE_1{1}",
    "YZ_1{2}",
    "CK_1{3}",
    "NN_0{4}",
]


@pytest.fixture
def recombining_graph():
    # The two training paths merge at CK and then split again. The Markov DAG
    # therefore represents four sequences, including two recombinations.
    return FlashBackGraph(
        [flashback_reverse(PATH_A), flashback_reverse(PATH_B)],
        abundances=[3, 1],
    )


@pytest.fixture
def analysis(recombining_graph):
    return recombining_graph.pseq_analysis()


def enumerate_public_graph(graph):
    """Independent exhaustive walk enumeration through the public CSR API."""
    labels = graph.all_nodes
    csr = graph.adjacency_csr()
    row = csr["row_offsets"]
    col = csr["col_indices"]
    weights = csr["weights"]
    root = next(i for i, label in enumerate(labels) if label.startswith("@"))
    stack = [(root, 1.0, [labels[root]])]
    paths = []
    while stack:
        node, probability, tokens = stack.pop()
        start, end = int(row[node]), int(row[node + 1])
        if start == end:
            sequence = flashback_reverse(tokens)
            paths.append((probability, len(sequence), sequence))
            continue
        for edge in range(start, end):
            target = int(col[edge])
            stack.append(
                (
                    target,
                    probability * float(weights[edge]),
                    tokens + [labels[target]],
                )
            )
    return paths


class TestExactTransform:
    def test_public_entry_point_and_types(self, analysis):
        assert isinstance(analysis, FlashBackPseqAnalysis)
        assert isinstance(analysis.exact_atoms(), PseqAtoms)
        assert isinstance(analysis.histogram(), PseqHistogram)
        assert isinstance(analysis.saddlepoint(), PseqSaddlepoint)

    def test_graph_recombines_to_four_paths(self, recombining_graph):
        paths = enumerate_public_graph(recombining_graph)
        assert len(paths) == 4
        assert recombining_graph.path_count == 4
        assert isinstance(recombining_graph.path_count, int)
        assert len({sequence for _, _, sequence in paths}) == 4

    def test_arbitrary_precision_path_count_above_float_exactness(self):
        # A graph whose path count needs three base-2**32 limbs, so the
        # native counter has to carry across limb boundaries and cannot be
        # represented exactly by a double.
        random.seed(20240730)
        sequences = [
            "C" + "".join(random.choice("ACGT") for _ in range(40)) + "F"
            for _ in range(3000)
        ]
        graph = FlashBackGraph(sequences)

        result = graph.path_count
        assert isinstance(result, int)
        assert result > 2**53, "fixture must exceed exact double range"
        assert result.bit_length() > 64, "fixture must span multiple limbs"
        assert result == reference_path_count(graph)
        # Every digit is significant, unlike the double-precision analytics.
        assert int(float(result)) != result

    def test_path_count_matches_brute_force_enumeration(self, recombining_graph):
        paths = enumerate_public_graph(recombining_graph)
        assert recombining_graph.path_count == len(paths)

    @pytest.mark.parametrize("q", [-0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    def test_mellin_matches_all_enumerated_paths(self, recombining_graph, analysis, q):
        probabilities = np.array(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)]
        )
        expected = np.sum(probabilities**q)
        assert analysis.mellin(q) == pytest.approx(expected, rel=2e-14)
        assert analysis.log_mellin(q) == pytest.approx(np.log(expected), abs=2e-14)

    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0])
    def test_derivatives_match_direct_sums(self, recombining_graph, analysis, q):
        probabilities = np.array(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)]
        )
        log_probabilities = np.log(probabilities)
        expected = np.array(
            [
                np.sum(probabilities**q * log_probabilities**order)
                for order in range(5)
            ]
        )
        np.testing.assert_allclose(
            analysis.derivatives(q, 4), expected, rtol=2e-13, atol=2e-13
        )

    def test_hill_identites(self, recombining_graph, analysis):
        probabilities = np.array(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)]
        )
        d0 = len(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities))
        d1 = np.exp(entropy)
        d2 = 1.0 / np.sum(probabilities**2)
        assert analysis.mellin(0) == pytest.approx(d0)
        assert recombining_graph.hill_number(1) == pytest.approx(d1)
        assert recombining_graph.hill_number(2) == pytest.approx(d2)
        assert analysis.mellin(2) == pytest.approx(1.0 / d2)
        assert d0 >= d1 >= d2

    def test_cgf_is_shifted_mellin_transform(self, analysis):
        for t in [-0.5, 0.0, 0.25, 0.75]:
            assert analysis.cgf(t) == pytest.approx(
                analysis.log_mellin(1 - t), abs=1e-15
            )


class TestExactMomentsAndLengths:
    def test_atoms_match_independent_enumeration(self, recombining_graph, analysis):
        direct = enumerate_public_graph(recombining_graph)
        atoms = analysis.exact_atoms()
        assert atoms.n_sequences == len(direct)
        assert atoms.probability_mass == pytest.approx(1.0, abs=1e-15)
        np.testing.assert_allclose(
            np.sort(atoms.probabilities),
            np.sort([p for p, _, _ in direct]),
            rtol=1e-15,
            atol=1e-15,
        )
        assert sorted(atoms.lengths.tolist()) == sorted(length for _, length, _ in direct)

    def test_atoms_respect_limit(self, analysis):
        with pytest.raises(ValueError, match="exceeding"):
            analysis.exact_atoms(max_paths=3)

    def test_global_moments_and_cumulants(self, recombining_graph, analysis):
        p = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
        x = -np.log(p)
        mean = np.sum(p * x)
        variance = np.sum(p * (x - mean) ** 2)
        third = np.sum(p * (x - mean) ** 3)
        fourth_cumulant = np.sum(p * (x - mean) ** 4) - 3 * variance**2

        moments = analysis.moments()
        assert moments["mass"] == pytest.approx(1.0)
        assert moments["mean"] == pytest.approx(mean, rel=1e-14)
        assert moments["variance"] == pytest.approx(variance, rel=1e-14)

        cumulants = analysis.cumulants()
        assert cumulants["log_normalizer"] == pytest.approx(0.0, abs=1e-14)
        assert cumulants["kappa1"] == pytest.approx(mean, rel=1e-14)
        assert cumulants["kappa2"] == pytest.approx(variance, rel=1e-14)
        assert cumulants["kappa3"] == pytest.approx(third, rel=1e-13)
        assert cumulants["kappa4"] == pytest.approx(fourth_cumulant, rel=1e-12)

    def test_length_derivatives_partition_global_transform(
        self, recombining_graph, analysis
    ):
        direct = enumerate_public_graph(recombining_graph)
        for q in [0.0, 0.5, 1.0, 2.0]:
            by_length = analysis.length_derivatives(q, 4)
            summed = np.sum(list(by_length.values()), axis=0)
            np.testing.assert_allclose(
                summed, analysis.derivatives(q, 4), rtol=2e-14, atol=2e-14
            )
            for length, jet in by_length.items():
                selected = np.array(
                    [p for p, path_length, _ in direct if path_length == length]
                )
                expected = np.array(
                    [
                        np.sum(selected**q * np.log(selected) ** order)
                        for order in range(5)
                    ]
                )
                np.testing.assert_allclose(jet, expected, rtol=2e-13, atol=2e-13)

    def test_length_profile_matches_direct_conditional_moments(
        self, recombining_graph, analysis
    ):
        direct = enumerate_public_graph(recombining_graph)
        profile = analysis.length_profile()
        assert set(profile) == {7, 8}
        assert sum(item["mass"] for item in profile.values()) == pytest.approx(1.0)
        for length, item in profile.items():
            selected = [
                (p, -np.log(p))
                for p, path_length, _ in direct
                if path_length == length
            ]
            mass = sum(p for p, _ in selected)
            mean = sum(p * x for p, x in selected) / mass
            variance = sum(p * (x - mean) ** 2 for p, x in selected) / mass
            assert item["mass"] == pytest.approx(mass)
            assert item["mean"] == pytest.approx(mean)
            assert item["variance"] == pytest.approx(variance)

    def test_single_path_degenerate_distribution(self):
        sequence = flashback_reverse(PATH_A)
        analysis = FlashBackGraph([sequence]).pseq_analysis()
        assert analysis.mellin(0) == pytest.approx(1)
        assert analysis.mellin(1) == pytest.approx(1)
        assert analysis.moments()["variance"] == 0
        assert analysis.histogram(length=len(sequence)).total_mass == pytest.approx(1)
        assert analysis.histogram(length=len(sequence) + 1).total_mass == 0


class TestDeterministicReconstruction:
    @pytest.mark.parametrize(
        ("measure", "expected_mass"),
        [("generated", 1.0), ("counting", 4.0)],
    )
    def test_histogram_conserves_measure(self, analysis, measure, expected_mass):
        histogram = analysis.histogram(512, measure=measure)
        assert histogram.total_mass == pytest.approx(expected_mass, rel=2e-14)
        assert histogram.exact is False
        assert histogram.grid_spacing > 0
        assert histogram.max_rounding_error >= histogram.grid_spacing

    def test_length_histogram_conserves_conditional_mass(self, analysis):
        h7 = analysis.histogram(512, length=7)
        h8 = analysis.histogram(512, length=8)
        assert h7.total_mass == pytest.approx(0.75, rel=2e-14)
        assert h8.total_mass == pytest.approx(0.25, rel=2e-14)
        assert h7.total_mass + h8.total_mass == pytest.approx(1.0)

    def test_grid_mean_is_exact_and_variance_converges(self, analysis):
        exact = analysis.moments()
        coarse = analysis.histogram(128)
        fine = analysis.histogram(4096)
        # Linear grid transport preserves the first moment.
        assert fine.mean == pytest.approx(exact["mean"], abs=2e-13)
        assert abs(fine.variance - exact["variance"]) < abs(
            coarse.variance - exact["variance"]
        )
        assert abs(fine.variance - exact["variance"]) < 1e-5

    def test_histogram_cdf_pdf_and_quantiles(self, analysis):
        histogram = analysis.histogram(1024)
        x = np.linspace(0, histogram.surprisal[-1], 100)
        cdf = histogram.cdf(x)
        assert np.all(np.diff(cdf) >= -1e-15)
        assert 0 <= cdf[0] <= cdf[-1] <= 1
        assert np.all(histogram.pdf(x) >= 0)
        quantiles = histogram.quantile([0.1, 0.5, 0.9])
        assert np.all(np.diff(quantiles) >= 0)

    def test_histogram_is_reproducible_without_rng(self, analysis):
        first = analysis.histogram(300, measure="counting")
        second = analysis.histogram(300, measure="counting")
        np.testing.assert_array_equal(first.surprisal, second.surprisal)
        np.testing.assert_array_equal(first.weights, second.weights)

    def test_saddlepoint_is_bounded_and_monotone(self, analysis):
        saddlepoint = analysis.saddlepoint()
        x = np.linspace(
            analysis.true_min_surprisal - 0.1,
            analysis.true_max_surprisal + 0.1,
            100,
        )
        cdf = saddlepoint.cdf(x)
        pdf = saddlepoint.pdf(x)
        assert cdf[0] == 0
        assert cdf[-1] == 1
        assert np.all((cdf >= 0) & (cdf <= 1))
        assert np.all(np.diff(cdf) >= -1e-12)
        assert np.all(pdf >= 0)


class TestInterpretationAndSamplingDepth:
    def test_individual_position_is_exact_for_small_graph(
        self, recombining_graph, analysis
    ):
        direct = enumerate_public_graph(recombining_graph)
        sequence = max(direct, key=lambda item: item[0])[2]
        probability = max(item[0] for item in direct)
        result = analysis.position(sequence)
        assert result["method"] == "exact_atoms"
        assert result["pseq"] == pytest.approx(probability)
        assert result["number_of_sequences_at_least_as_probable"] == 1
        assert result["fraction_of_sequences_at_least_as_probable"] == pytest.approx(0.25)
        assert result["generated_mass_at_least_as_probable"] == pytest.approx(probability)
        assert result["relative_to_D1"] > 1

    def test_position_rejects_unsupported_sequence(self, analysis):
        with pytest.raises(ValueError, match="outside"):
            analysis.position("ZZZZZZ")

    @pytest.mark.parametrize("n", [0, 1, 2, 10, 100])
    def test_expected_richness_matches_direct_atom_sum(
        self, recombining_graph, analysis, n
    ):
        probabilities = np.array(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)]
        )
        expected = np.sum(1 - (1 - probabilities) ** n)
        result = analysis.expected_richness(n)
        assert result["method"] == "exact_atoms"
        assert result["expected_richness"] == pytest.approx(expected, rel=2e-14)

    def test_expected_frequency_spectrum_matches_binomial_sum(
        self, recombining_graph, analysis
    ):
        n = 12
        max_count = 6
        probabilities = np.array(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)]
        )
        expected = []
        for r in range(max_count + 1):
            coefficient = comb(n, r)
            expected.append(
                np.sum(coefficient * probabilities**r * (1 - probabilities) ** (n - r))
            )
        result = analysis.expected_frequency_spectrum(n, max_count)
        assert result["method"] == "exact_atoms"
        np.testing.assert_allclose(
            result["expected_counts"], expected, rtol=2e-14, atol=2e-14
        )

    def test_grid_occupancy_converges_to_exact(self, analysis):
        exact = analysis.expected_richness(100)["expected_richness"]
        grid = analysis.expected_richness(
            100, bins=8192, max_exact_paths=1
        )
        assert grid["method"] == "deterministic_grid"
        assert grid["expected_richness"] == pytest.approx(exact, rel=2e-5)

    def test_pair_collision_identity(self, recombining_graph, analysis):
        n = 20
        probabilities = np.array(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)]
        )
        expected_collisions = comb(n, 2) * np.sum(probabilities**2)
        d2 = recombining_graph.hill_number(2)
        assert expected_collisions == pytest.approx(comb(n, 2) / d2)
        assert analysis.mellin(2) == pytest.approx(1 / d2)


def test_invalid_parameters(analysis):
    with pytest.raises(ValueError):
        analysis.derivatives(1, 9)
    with pytest.raises(ValueError):
        analysis.histogram(3)
    with pytest.raises(ValueError):
        analysis.histogram(measure="invalid")
    with pytest.raises(ValueError):
        analysis.expected_frequency_spectrum(3, 4)
    with pytest.raises(ValueError):
        analysis.exact_atoms(0)


@pytest.mark.parametrize(
    "graph",
    [
        FlashBackGraph(["ABCDE", "ABXDE", "AYCZE"], abundances=[7, 2, 1]),
        FlashBackGraph(["ABCD", "ABEFG", "XYCD", "XYEFG"], smoothing=0.25),
        FlashBackGraph(
            [
                "CASSLGIRRT",
                "CASSLGYEQYF",
                "CASSLEPSGGTDTQYF",
                "CASSDTSGGTDTQYF",
            ],
            abundances=[5, 3, 2, 1],
        ),
    ],
    ids=["recombined-inner", "mixed-length-smoothed", "cdr3"],
)
def test_exact_dp_on_multiple_flashback_topologies(graph):
    """Cross-check the implementation beyond the hand-designed fixture."""
    analysis = graph.pseq_analysis()
    direct = enumerate_public_graph(graph)
    probabilities = np.array([p for p, _, _ in direct])
    assert len(direct) == graph.path_count
    for q in [0.0, 0.3, 1.0, 1.7, 3.0]:
        expected = np.array(
            [
                np.sum(probabilities**q * np.log(probabilities) ** order)
                for order in range(3)
            ]
        )
        np.testing.assert_allclose(
            analysis.derivatives(q, 2), expected, rtol=3e-13, atol=3e-13
        )
        assert analysis.log_mellin(q) == pytest.approx(
            np.log(expected[0]), abs=3e-13
        )
    assert sum(item["mass"] for item in analysis.length_profile().values()) == (
        pytest.approx(1.0, abs=2e-14)
    )


def test_fine_grid_cdf_matches_exact_atoms_between_jumps(analysis):
    atoms = analysis.exact_atoms()
    histogram = analysis.histogram(8192)
    support = np.sort(np.unique(atoms.surprisal))
    midpoints = 0.5 * (support[:-1] + support[1:])
    np.testing.assert_allclose(
        histogram.cdf(midpoints),
        atoms.cdf(midpoints),
        rtol=0,
        atol=2e-6,
    )
