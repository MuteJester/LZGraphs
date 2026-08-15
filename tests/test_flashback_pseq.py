"""Exhaustive validation of sampling-free FlashBack p-sequence analytics."""

from __future__ import annotations

import gc
import random
from math import comb, erf, exp, pi, sqrt

import numpy as np
import pytest

from LZGraphs import (
    FlashBackGraph,
    FlashBackPseqAnalysis,
    PseqAtoms,
    PseqAttribution,
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


def enumerate_edge_paths(graph):
    """Exhaustively return probability, visited nodes, and CSR edge IDs."""
    labels = graph.all_nodes
    csr = graph.adjacency_csr()
    row = np.asarray(csr["row_offsets"])
    col = np.asarray(csr["col_indices"])
    weights = np.asarray(csr["weights"])
    root = next(i for i, label in enumerate(labels) if label.startswith("@"))
    stack = [(root, 1.0, (root,), ())]
    paths = []
    while stack:
        node, probability, nodes, edges = stack.pop()
        start, end = int(row[node]), int(row[node + 1])
        if start == end:
            paths.append((probability, nodes, edges))
            continue
        for edge in range(start, end):
            target = int(col[edge])
            stack.append(
                (
                    target,
                    probability * float(weights[edge]),
                    nodes + (target,),
                    edges + (edge,),
                )
            )
    return paths


class TestExactTransform:
    def test_public_entry_point_and_types(self, analysis):
        assert isinstance(analysis, FlashBackPseqAnalysis)
        assert isinstance(analysis.exact_atoms(), PseqAtoms)
        assert isinstance(analysis.histogram(), PseqHistogram)
        assert isinstance(analysis.saddlepoint(), PseqSaddlepoint)
        assert isinstance(analysis.attribution(), PseqAttribution)

    def test_native_initialization_matches_python_reference(self, analysis):
        np.testing.assert_array_equal(
            analysis._topological_order, analysis._make_topological_order()
        )
        native_bounds = (
            analysis.true_min_surprisal,
            analysis.true_max_surprisal,
            analysis._max_edges,
        )
        np.testing.assert_allclose(native_bounds, analysis._path_bounds(), rtol=0, atol=0)

    def test_native_graph_views_are_read_only_and_zero_copy(self, analysis):
        for array in (analysis._row, analysis._col, analysis._weights, analysis._topological_order):
            assert not array.flags.writeable
            assert not array.flags.owndata

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
            "C" + "".join(random.choice("ACGT") for _ in range(40)) + "F" for _ in range(3000)
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
        probabilities = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
        expected = np.sum(probabilities**q)
        assert analysis.mellin(q) == pytest.approx(expected, rel=2e-14)
        assert analysis.log_mellin(q) == pytest.approx(np.log(expected), abs=2e-14)

    @pytest.mark.parametrize("q", [-50.0, -10.0, -1.0, 0.0, 1.0, 10.0, 50.0])
    def test_native_log_mellin_matches_python_logsumexp(self, analysis, q):
        assert analysis.log_mellin(q) == pytest.approx(
            analysis._log_mellin_python(q), rel=0, abs=2e-13
        )

    @pytest.mark.parametrize("q", [-10.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0])
    @pytest.mark.parametrize("order", [0, 1, 2, 4, 8])
    def test_native_tilted_moments_match_python_oracle(self, analysis, q, order):
        native_log_mass, native_raw = analysis._tilted_log_moments(q, order)
        python_log_mass, python_raw = analysis._tilted_log_moments_python(q, order)
        assert native_log_mass == pytest.approx(python_log_mass, abs=3e-13)
        np.testing.assert_allclose(native_raw, python_raw, rtol=3e-13, atol=3e-11)

    @pytest.mark.parametrize("q", [-1000.0, -100.0, 100.0, 1000.0])
    def test_log_domain_transform_remains_finite_at_extreme_tilts(self, analysis, q):
        atoms = analysis.exact_atoms()
        terms = q * np.log(atoms.probabilities)
        expected = float(np.logaddexp.reduce(terms))
        assert np.isfinite(analysis.log_mellin(q))
        assert analysis.log_mellin(q) == pytest.approx(expected, abs=2e-12)

    @pytest.mark.parametrize("t", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_native_cgf_cumulants_match_exact_atoms(self, analysis, t):
        atoms = analysis.exact_atoms()
        x = atoms.surprisal.astype(np.longdouble)
        log_p = -x
        q = np.longdouble(1.0 - t)
        log_weights = q * log_p
        maximum = np.max(log_weights)
        weights = np.exp(log_weights - maximum)
        weights /= np.sum(weights, dtype=np.longdouble)
        mean = np.sum(weights * x, dtype=np.longdouble)
        centered = x - mean
        central2 = np.sum(weights * centered**2, dtype=np.longdouble)
        central3 = np.sum(weights * centered**3, dtype=np.longdouble)
        central4 = np.sum(weights * centered**4, dtype=np.longdouble)
        expected = np.array(
            [
                maximum + np.log(np.sum(np.exp(log_weights - maximum))),
                mean,
                central2,
                central3,
                central4 - 3 * central2**2,
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(
            analysis._cgf_derivatives(t, 4),
            expected,
            rtol=2e-13,
            atol=2e-13,
        )

    def test_public_tilted_moments_reports_normalized_statistics(self, analysis):
        result = analysis.tilted_moments(0.5, order=4)
        assert result["q"] == 0.5
        assert result["log_mass"] == pytest.approx(analysis.log_mellin(0.5))
        raw = result["raw_log_probability_moments"]
        central = result["central_log_probability_moments"]
        cumulants = result["log_probability_cumulants"]
        assert raw.shape == central.shape == cumulants.shape == (5,)
        assert raw[0] == central[0] == 1.0
        assert central[1] == 0.0
        assert cumulants[0] == result["log_mass"]
        assert cumulants[1] == raw[1]
        assert cumulants[2] == central[2]
        assert cumulants[3] == central[3]
        assert cumulants[4] == pytest.approx(central[4] - 3 * central[2] ** 2)

    @pytest.mark.parametrize("q", [float("nan"), float("inf"), -float("inf")])
    def test_log_domain_transform_rejects_nonfinite_tilt(self, analysis, q):
        with pytest.raises(ValueError, match="finite"):
            analysis.log_mellin(q)

    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0])
    def test_derivatives_match_direct_sums(self, recombining_graph, analysis, q):
        probabilities = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
        log_probabilities = np.log(probabilities)
        expected = np.array(
            [np.sum(probabilities**q * log_probabilities**order) for order in range(5)]
        )
        np.testing.assert_allclose(analysis.derivatives(q, 4), expected, rtol=2e-13, atol=2e-13)

    @pytest.mark.parametrize("q", [-0.5, 0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("order", range(9))
    def test_native_derivatives_match_python_reference(self, analysis, q, order):
        np.testing.assert_allclose(
            analysis.derivatives(q, order),
            analysis._derivatives_python(q, order),
            rtol=2e-13,
            atol=2e-13,
        )

    def test_native_eighth_order_derivatives_match_atoms(self, recombining_graph, analysis):
        probabilities = np.asarray([p for p, _, _ in enumerate_public_graph(recombining_graph)])
        q = 0.5
        expected = np.asarray(
            [np.sum(probabilities**q * np.log(probabilities) ** order) for order in range(9)]
        )
        np.testing.assert_allclose(analysis.derivatives(q, 8), expected, rtol=2e-13, atol=2e-13)

    def test_native_derivatives_handle_graph_with_no_edges(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        np.testing.assert_array_equal(
            graph.pseq_analysis().derivatives(1.0, 8),
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_hill_identites(self, recombining_graph, analysis):
        probabilities = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
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
            assert analysis.cgf(t) == pytest.approx(analysis.log_mellin(1 - t), abs=1e-15)


class TestExactMomentsAndLengths:
    @pytest.mark.parametrize(
        "sequences",
        [
            [flashback_reverse(PATH_A), flashback_reverse(PATH_B)],
            ["CASS", "CASST", "CAT", "CATS"],
        ],
    )
    def test_length_marginals_match_brute_force_atoms(self, sequences):
        graph = FlashBackGraph(sequences)
        analysis = graph.pseq_analysis()
        atoms = analysis.exact_atoms()
        marginals = analysis.length_marginals()
        for length in np.unique(atoms.lengths):
            selected = atoms.lengths == length
            assert marginals[int(length)]["counting"] == pytest.approx(
                float(np.count_nonzero(selected)), rel=0, abs=0
            )
            assert marginals[int(length)]["generated"] == pytest.approx(
                float(np.sum(atoms.probabilities[selected], dtype=np.longdouble)),
                rel=2e-15,
                abs=2e-15,
            )

    def test_length_marginals_match_independent_native_calls(self, analysis):
        marginals = analysis.length_marginals()
        counting = analysis.length_derivatives(0.0, 0)
        generated = analysis.length_derivatives(1.0, 0)
        assert marginals.keys() == counting.keys() == generated.keys()
        for length, values in marginals.items():
            assert values["counting"] == counting[length][0]
            assert values["generated"] == pytest.approx(
                generated[length][0], rel=2e-15, abs=2e-15
            )

    def test_length_marginals_handle_graph_with_no_edges(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        assert graph.pseq_analysis().length_marginals() == {
            0: {"counting": 1.0, "generated": 1.0}
        }

    def test_path_count_by_length_matches_recombining_atoms(self, recombining_graph, analysis):
        atoms = analysis.exact_atoms()
        expected = {
            int(length): float(np.count_nonzero(atoms.lengths == length))
            for length in np.unique(atoms.lengths)
        }
        assert recombining_graph.path_count_by_length() == expected
        assert recombining_graph.path_count_by_length() is not expected
        by_derivative = analysis.length_derivatives(0.0, 0)
        assert {length: float(jet[0]) for length, jet in by_derivative.items()} == expected

    def test_path_count_by_length_matches_second_brute_force_graph(self):
        graph = FlashBackGraph(["CASS", "CASST", "CAT", "CATS"])
        atoms = graph.pseq_analysis().exact_atoms()
        expected = {
            int(length): float(np.count_nonzero(atoms.lengths == length))
            for length in np.unique(atoms.lengths)
        }
        assert graph.path_count_by_length() == expected
        assert sum(expected.values()) == graph.path_count

    @pytest.mark.parametrize("sequence", ["A", "CASS"])
    def test_path_count_by_length_handles_degenerate_single_path(self, sequence):
        graph = FlashBackGraph([sequence])
        assert graph.path_count_by_length() == {len(sequence): 1.0}

    def test_path_count_by_length_handles_graph_with_no_edges(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        assert graph.n_edges == 0
        assert graph.path_count_by_length() == {0: float(graph.path_count)}

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

    def test_length_derivatives_partition_global_transform(self, recombining_graph, analysis):
        direct = enumerate_public_graph(recombining_graph)
        for q in [0.0, 0.5, 1.0, 2.0]:
            by_length = analysis.length_derivatives(q, 4)
            summed = np.sum(list(by_length.values()), axis=0)
            np.testing.assert_allclose(summed, analysis.derivatives(q, 4), rtol=2e-14, atol=2e-14)
            for length, jet in by_length.items():
                selected = np.array([p for p, path_length, _ in direct if path_length == length])
                expected = np.array(
                    [np.sum(selected**q * np.log(selected) ** order) for order in range(5)]
                )
                np.testing.assert_allclose(jet, expected, rtol=2e-13, atol=2e-13)

    @pytest.mark.parametrize("q", [-0.5, 0.0, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("order", range(9))
    def test_native_length_derivatives_match_python_reference(self, analysis, q, order):
        native = analysis.length_derivatives(q, order)
        reference = analysis._length_derivatives_python(q, order)
        assert native.keys() == reference.keys()
        for length in native:
            np.testing.assert_allclose(native[length], reference[length], rtol=2e-13, atol=2e-13)

    def test_native_eighth_order_length_derivatives_match_atoms(self, recombining_graph, analysis):
        direct = enumerate_public_graph(recombining_graph)
        q = 0.5
        native = analysis.length_derivatives(q, 8)
        for length, jet in native.items():
            selected = np.asarray([p for p, path_length, _ in direct if path_length == length])
            expected = np.asarray(
                [np.sum(selected**q * np.log(selected) ** order) for order in range(9)]
            )
            np.testing.assert_allclose(jet, expected, rtol=2e-13, atol=2e-13)

    def test_native_length_derivatives_handle_graph_with_no_edges(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        result = graph.pseq_analysis().length_derivatives(1.0, 4)
        np.testing.assert_array_equal(result[0], [1.0, 0.0, 0.0, 0.0, 0.0])

    def test_length_profile_matches_direct_conditional_moments(self, recombining_graph, analysis):
        direct = enumerate_public_graph(recombining_graph)
        profile = analysis.length_profile()
        assert set(profile) == {7, 8}
        assert sum(item["mass"] for item in profile.values()) == pytest.approx(1.0)
        for length, item in profile.items():
            selected = [(p, -np.log(p)) for p, path_length, _ in direct if path_length == length]
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


class TestTiltedAttribution:
    @pytest.mark.parametrize("q", [-3.0, 0.0, 0.5, 1.0, 2.0, 5.0])
    def test_matches_exhaustive_path_attribution(self, recombining_graph, analysis, q):
        paths = enumerate_edge_paths(recombining_graph)
        log_weight = np.asarray([q * np.log(path[0]) for path in paths])
        maximum = float(np.max(log_weight))
        normalized = np.exp(log_weight - maximum)
        normalized /= np.sum(normalized)

        expected_nodes = np.zeros(recombining_graph.n_nodes)
        expected_edges = np.zeros(recombining_graph.n_edges)
        for path_weight, (_, nodes, edges) in zip(normalized, paths):
            expected_nodes[list(nodes)] += path_weight
            expected_edges[list(edges)] += path_weight

        result = analysis.attribution(q)
        expected_log_mass = maximum + np.log(np.sum(np.exp(log_weight - maximum)))
        assert result.log_mass == pytest.approx(expected_log_mass, abs=2e-14)
        np.testing.assert_allclose(result.node_probability, expected_nodes, rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result.edge_probability, expected_edges, rtol=2e-14, atol=2e-14)

    def test_flow_conservation_and_transform_invariants(self, recombining_graph, analysis):
        result = analysis.attribution(0.37)
        row, col = analysis._row, analysis._col
        incoming = np.bincount(
            col, weights=result.edge_probability, minlength=recombining_graph.n_nodes
        )
        outgoing = np.asarray(
            [
                np.sum(result.edge_probability[row[u] : row[u + 1]])
                for u in range(recombining_graph.n_nodes)
            ]
        )
        root = analysis._root
        sinks = analysis._sinks
        internal = np.ones(recombining_graph.n_nodes, dtype=bool)
        internal[root] = False
        internal[sinks] = False

        assert result.node_probability[root] == pytest.approx(1.0, abs=2e-15)
        assert outgoing[root] == pytest.approx(1.0, abs=2e-15)
        assert np.sum(result.node_probability[sinks]) == pytest.approx(1.0, abs=2e-15)
        np.testing.assert_allclose(
            incoming[internal], result.node_probability[internal], rtol=2e-14, atol=2e-14
        )
        np.testing.assert_allclose(
            outgoing[internal], result.node_probability[internal], rtol=2e-14, atol=2e-14
        )
        assert result.log_mass == pytest.approx(analysis.log_mellin(result.q), abs=2e-14)

    @pytest.mark.parametrize("q", [-2.0, 0.0, 0.5, 1.0, 3.0])
    def test_surprisal_and_path_size_match_exact_paths(self, recombining_graph, analysis, q):
        paths = enumerate_edge_paths(recombining_graph)
        unnormalized = np.asarray([p**q for p, _, _ in paths])
        normalized = unnormalized / np.sum(unnormalized)
        expected_surprisal = sum(
            weight * -np.log(path[0]) for weight, path in zip(normalized, paths)
        )
        expected_edges = sum(weight * len(path[2]) for weight, path in zip(normalized, paths))
        result = analysis.attribution(q)
        assert result.mean_surprisal == pytest.approx(expected_surprisal, rel=2e-14, abs=2e-14)
        assert np.sum(result.edge_surprisal_contribution) == pytest.approx(
            expected_surprisal, rel=2e-14, abs=2e-14
        )
        assert result.expected_path_edges == pytest.approx(expected_edges, rel=2e-14, abs=2e-14)

    @pytest.mark.parametrize("q", [-1.5, 0.0, 0.7, 2.0])
    def test_independent_edge_log_weight_gradient(self, recombining_graph, analysis, q):
        paths = enumerate_edge_paths(recombining_graph)
        probabilities = np.asarray([p for p, _, _ in paths])
        result = analysis.attribution(q)
        epsilon = 1e-6
        for edge in range(recombining_graph.n_edges):
            used = np.asarray([edge in path_edges for _, _, path_edges in paths])

            def perturbed_log_mass(delta):
                terms = q * (np.log(probabilities) + delta * used)
                maximum = np.max(terms)
                return maximum + np.log(np.sum(np.exp(terms - maximum)))

            numerical = (perturbed_log_mass(epsilon) - perturbed_log_mass(-epsilon)) / (2 * epsilon)
            assert result.edge_sensitivity[edge] == pytest.approx(numerical, rel=2e-8, abs=2e-10)

    def test_q_zero_is_supported_path_fraction(self, recombining_graph, analysis):
        paths = enumerate_edge_paths(recombining_graph)
        expected = np.zeros(recombining_graph.n_edges)
        for _, _, edges in paths:
            expected[list(edges)] += 1.0 / len(paths)
        np.testing.assert_allclose(
            analysis.attribution(0.0).edge_probability,
            expected,
            rtol=0,
            atol=2e-15,
        )

    def test_q_one_decomposes_generated_entropy(self, analysis):
        result = analysis.attribution(1.0)
        assert result.log_mass == pytest.approx(0.0, abs=2e-15)
        assert result.mean_surprisal == pytest.approx(
            analysis.moments()["mean"], rel=2e-14, abs=2e-14
        )

    @pytest.mark.parametrize("q", [-1000.0, 1000.0])
    def test_extreme_tilts_remain_finite_and_normalized(self, analysis, q):
        result = analysis.attribution(q)
        assert np.all(np.isfinite(result.node_probability))
        assert np.all(np.isfinite(result.edge_probability))
        assert np.all((result.node_probability >= 0) & (result.node_probability <= 1))
        assert np.all((result.edge_probability >= 0) & (result.edge_probability <= 1))
        assert result.node_probability[analysis._root] == pytest.approx(1.0)
        assert np.sum(result.node_probability[analysis._sinks]) == pytest.approx(1.0)

    def test_arrays_are_read_only_zero_copy_and_keep_storage_alive(self, analysis):
        result = analysis.attribution(1.0)
        nodes = result.node_probability
        edges = result.edge_probability
        expected_nodes = nodes.copy()
        expected_edges = edges.copy()
        assert not nodes.flags.writeable
        assert not edges.flags.writeable
        assert not nodes.flags.owndata
        assert not edges.flags.owndata
        with pytest.raises(ValueError, match="read-only"):
            edges[0] = 0.0
        del result
        gc.collect()
        np.testing.assert_array_equal(nodes, expected_nodes)
        np.testing.assert_array_equal(edges, expected_edges)

    def test_degenerate_no_edge_graph(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        analysis = graph.pseq_analysis()
        result = analysis.attribution(1.0)
        expected_nodes = np.zeros(graph.n_nodes)
        expected_nodes[analysis._root] = 1.0
        np.testing.assert_array_equal(result.node_probability, expected_nodes)
        assert result.edge_probability.size == 0
        assert result.log_mass == 0.0
        assert result.expected_path_edges == 0.0
        assert result.mean_surprisal == 0.0
        assert result.top_edges() == []

    def test_top_edges_are_sorted_and_validate_arguments(self, analysis):
        result = analysis.attribution(0.5)
        top = result.top_edges(3, by="occupancy")
        assert len(top) == 3
        assert all(top[i]["occupancy"] >= top[i + 1]["occupancy"] for i in range(len(top) - 1))
        assert top == result.top_edges(3, by="sensitivity")
        assert len(result.top_edges(2, by="surprisal")) == 2
        assert result.top_edges(0) == []
        with pytest.raises(ValueError, match="non-negative"):
            result.top_edges(-1)
        with pytest.raises(ValueError, match="occupancy"):
            result.top_edges(by="unknown")
        with pytest.raises(ValueError, match="finite"):
            analysis.attribution(np.inf)


class TestEdgeThresholdDiversity:
    @staticmethod
    def brute_force(graph, thresholds):
        paths = enumerate_edge_paths(graph)
        weights = np.asarray(graph.adjacency_csr()["weights"])
        output = []
        for threshold in thresholds:
            probabilities = np.asarray(
                [
                    probability
                    for probability, _, edges in paths
                    if all(weights[edge] > threshold for edge in edges)
                ]
            )
            if probabilities.size == 0:
                d0 = d1 = d2 = mass = 0.0
            else:
                mass = float(np.sum(probabilities))
                conditioned = probabilities / mass
                d0 = float(probabilities.size)
                d1 = float(np.exp(-np.sum(conditioned * np.log(conditioned))))
                d2 = float(1.0 / np.sum(conditioned**2))
            output.append(
                (
                    d0,
                    d1,
                    d2,
                    mass,
                    int(np.count_nonzero(weights > threshold)),
                )
            )
        return np.asarray(output)

    def test_matches_exhaustive_paths_with_strict_unsorted_thresholds(
        self, recombining_graph, analysis
    ):
        weights = np.asarray(recombining_graph.adjacency_csr()["weights"])
        thresholds = np.asarray(
            [
                np.inf,
                weights[0],
                -np.inf,
                np.median(weights),
                0.0,
                weights[0],
            ]
        )
        expected = self.brute_force(recombining_graph, thresholds)
        result = analysis.diversity_under_edge_thresholds(thresholds)
        np.testing.assert_array_equal(result["thresholds"], thresholds)
        np.testing.assert_allclose(result["D0"], expected[:, 0], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["D1"], expected[:, 1], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["D2"], expected[:, 2], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["surviving_mass"], expected[:, 3], rtol=2e-14, atol=2e-14)
        np.testing.assert_array_equal(result["kept_edges"], expected[:, 4])
        np.testing.assert_allclose(
            result["edge_fraction"],
            expected[:, 4] / recombining_graph.n_edges,
            rtol=0,
            atol=0,
        )

    def test_unpruned_endpoint_matches_public_diversity(self, analysis):
        result = analysis.diversity_under_edge_thresholds([-np.inf])
        assert result["D0"][0] == pytest.approx(analysis.graph.path_count, rel=2e-14)
        assert result["D1"][0] == pytest.approx(analysis.graph.hill_number(1), rel=2e-14)
        assert result["D2"][0] == pytest.approx(analysis.graph.hill_number(2), rel=2e-14)
        assert result["surviving_mass"][0] == pytest.approx(1.0, abs=2e-14)
        assert result["kept_edges"][0] == analysis.graph.n_edges

    def test_multiple_native_batches_match_second_enumerated_graph(self):
        graph = FlashBackGraph(
            ["CASS", "CASST", "CAT", "CATS"],
            abundances=[7, 3, 2, 1],
        )
        weights = np.asarray(graph.adjacency_csr()["weights"])
        # Nineteen cutoffs exercise two complete eight-lane native batches
        # and a partial final batch. Reversing them also tests restoration of
        # the caller's arbitrary input order.
        thresholds = np.linspace(-0.01, float(np.max(weights)) + 0.01, 19)[::-1]
        expected = self.brute_force(graph, thresholds)
        result = graph.pseq_analysis().diversity_under_edge_thresholds(thresholds)
        np.testing.assert_allclose(result["D0"], expected[:, 0], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["D1"], expected[:, 1], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["D2"], expected[:, 2], rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(
            result["surviving_mass"],
            expected[:, 3],
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_array_equal(result["kept_edges"], expected[:, 4])

    def test_log_outputs_and_empty_support(self, analysis):
        result = analysis.diversity_under_edge_thresholds([-np.inf, np.inf])
        for order in range(3):
            diversity = result[f"D{order}"]
            log_diversity = result[f"log_D{order}"]
            assert log_diversity[0] == pytest.approx(np.log(diversity[0]))
            assert diversity[1] == 0.0
            assert log_diversity[1] == -np.inf
        assert result["surviving_mass"][1] == 0.0
        assert result["kept_edges"][1] == 0

    def test_empty_thresholds_and_validation(self, analysis):
        result = analysis.diversity_under_edge_thresholds([])
        assert all(values.size == 0 for values in result.values())
        with pytest.raises(ValueError, match="one-dimensional"):
            analysis.diversity_under_edge_thresholds([[0.1]])
        with pytest.raises(ValueError, match="NaN"):
            analysis.diversity_under_edge_thresholds([np.nan])

    def test_degenerate_no_edge_graph(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        result = graph.pseq_analysis().diversity_under_edge_thresholds([-np.inf, 0.0, np.inf])
        np.testing.assert_array_equal(result["D0"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result["D1"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result["D2"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result["surviving_mass"], [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result["kept_edges"], [0, 0, 0])


class TestDeterministicReconstruction:
    @pytest.mark.parametrize("bins", [32, 127, 512])
    def test_paired_histograms_match_independent_native_calls(self, analysis, bins):
        paired = analysis.histogram_pair(bins)
        counting = analysis.histogram(bins, measure="counting")
        generated = analysis.histogram(bins, measure="generated")
        np.testing.assert_array_equal(paired["counting"].surprisal, counting.surprisal)
        np.testing.assert_array_equal(paired["generated"].surprisal, generated.surprisal)
        np.testing.assert_allclose(
            paired["counting"].weights,
            counting.weights,
            rtol=1e-12,
            atol=0.0,
        )
        np.testing.assert_allclose(
            paired["generated"].weights,
            generated.weights,
            rtol=1e-12,
            atol=1e-15,
        )
        assert paired["counting"].grid_spacing == counting.grid_spacing
        assert paired["generated"].max_rounding_error == generated.max_rounding_error

    def test_paired_histogram_validation_and_degenerate_graph(self):
        graph = FlashBackGraph(["CASS"]).without(["CASS"])
        paired = graph.pseq_analysis().histogram_pair()
        np.testing.assert_array_equal(paired["counting"].weights, [1.0])
        np.testing.assert_array_equal(paired["generated"].weights, [1.0])
        with pytest.raises(ValueError, match="bins must exceed"):
            FlashBackGraph(["CASS", "CATS"]).pseq_analysis().histogram_pair(2)

    @pytest.mark.parametrize("bins", [32, 127, 512])
    @pytest.mark.parametrize("measure", ["generated", "counting"])
    @pytest.mark.parametrize("length", [None, 7, 8, 99])
    def test_native_histogram_matches_python_reference(self, analysis, bins, measure, length):
        native = analysis.histogram(bins, measure=measure, length=length)
        reference = analysis._histogram_python(bins, measure=measure, length=length)
        np.testing.assert_array_equal(native.surprisal, reference.surprisal)
        np.testing.assert_allclose(native.weights, reference.weights, rtol=2e-14, atol=2e-15)
        assert native.grid_spacing == reference.grid_spacing
        assert native.max_rounding_error == reference.max_rounding_error

    @pytest.mark.parametrize("measure", ["generated", "counting"])
    def test_length_histograms_partition_global_grid(self, analysis, measure):
        global_histogram = analysis.histogram(512, measure=measure)
        by_length = np.sum(
            [
                analysis.histogram(512, measure=measure, length=length).weights
                for length in analysis.length_derivatives(0.0, 0)
            ],
            axis=0,
        )
        np.testing.assert_allclose(by_length, global_histogram.weights, rtol=2e-14, atol=2e-15)

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
        assert abs(fine.variance - exact["variance"]) < abs(coarse.variance - exact["variance"])
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

    def test_native_saddlepoint_batch_solves_exact_tilt_equation(self):
        random.seed(91827)
        graph = FlashBackGraph(
            ["C" + "".join(random.choice("ACGT") for _ in range(12)) + "F" for _ in range(80)]
        )
        analysis = graph.pseq_analysis()
        assert 64 < graph.path_count < 100_000
        saddlepoint = analysis.saddlepoint()
        saddlepoint.discrete_fallback_paths = 0
        x = np.linspace(
            analysis.true_min_surprisal + 0.01,
            analysis.true_max_surprisal - 0.01,
            21,
        )
        result = saddlepoint._native_evaluate(x)
        residuals = np.array(
            [
                analysis._cgf_derivatives(float(t), 2)[1] - point
                for point, t in zip(x, result["saddle"])
            ]
        )
        np.testing.assert_allclose(residuals, 0.0, rtol=0, atol=2e-12)
        assert np.all(result["iterations"] <= 20)
        assert np.all(result["pdf"] >= 0)
        assert np.all(np.diff(result["cdf"]) >= 0)

        atoms = analysis.exact_atoms()
        histogram = analysis.histogram(8192)
        np.testing.assert_allclose(result["cdf"], atoms.cdf(x), rtol=0, atol=0.03)
        np.testing.assert_allclose(result["cdf"], histogram.cdf(x), rtol=0, atol=0.03)

        # Independently reconstruct the saddlepoint and Lugannani-Rice
        # formulas from the explicitly enumerated probability atoms.
        atom_x = atoms.surprisal.astype(np.longdouble)
        atom_p = atoms.probabilities.astype(np.longdouble)
        base_log_mass = np.log(np.sum(atom_p, dtype=np.longdouble))
        base_mean = np.sum(atom_p * atom_x) / np.sum(atom_p)
        base_variance = np.sum(atom_p * (atom_x - base_mean) ** 2) / np.sum(atom_p)
        expected_pdf = []
        expected_cdf = []
        for point, t in zip(x, result["saddle"]):
            log_weights = (np.longdouble(1) - t) * np.log(atom_p)
            maximum = np.max(log_weights)
            weights = np.exp(log_weights - maximum)
            z = np.sum(weights, dtype=np.longdouble)
            normalized = weights / z
            tilted_mean = np.sum(normalized * atom_x, dtype=np.longdouble)
            variance = np.sum(
                normalized * (atom_x - tilted_mean) ** 2,
                dtype=np.longdouble,
            )
            k = maximum + np.log(z) - base_log_mass
            expected_pdf.append(exp(float(k - t * point)) / sqrt(2 * pi * float(variance)))
            if abs(t) < 1e-8:
                normal_z = (point - float(base_mean)) / sqrt(float(base_variance))
                expected_cdf.append(0.5 * (1 + erf(normal_z / sqrt(2))))
            else:
                w = np.copysign(sqrt(max(2 * float(t * point - k), 0)), t)
                u = t * sqrt(float(variance))
                normal_cdf = 0.5 * (1 + erf(w / sqrt(2)))
                normal_pdf = exp(-0.5 * w * w) / sqrt(2 * pi)
                expected_cdf.append(normal_cdf + normal_pdf * (1 / w - 1 / u))
        np.testing.assert_allclose(result["pdf"], expected_pdf, rtol=3e-13)
        np.testing.assert_allclose(result["cdf"], expected_cdf, rtol=3e-13)

    def test_saddlepoint_pdf_cdf_is_fused_and_shape_preserving(self):
        random.seed(87231)
        graph = FlashBackGraph(
            ["C" + "".join(random.choice("ACGT") for _ in range(10)) + "F" for _ in range(60)]
        )
        saddlepoint = graph.pseq_analysis().saddlepoint()
        saddlepoint.discrete_fallback_paths = 0
        x = np.linspace(
            saddlepoint.analysis.true_min_surprisal - 0.1,
            saddlepoint.analysis.true_max_surprisal + 0.1,
            12,
        ).reshape(3, 4)
        pdf, cdf = saddlepoint.pdf_cdf(x)
        assert pdf.shape == x.shape
        assert cdf.shape == x.shape
        np.testing.assert_array_equal(pdf, saddlepoint.pdf(x))
        np.testing.assert_array_equal(cdf, saddlepoint.cdf(x))
        scalar_pdf, scalar_cdf = saddlepoint.pdf_cdf(float(x[1, 1]))
        assert isinstance(scalar_pdf, float)
        assert isinstance(scalar_cdf, float)


class TestInterpretationAndSamplingDepth:
    def test_individual_position_is_exact_for_small_graph(self, recombining_graph, analysis):
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
    def test_expected_richness_matches_direct_atom_sum(self, recombining_graph, analysis, n):
        probabilities = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
        expected = np.sum(1 - (1 - probabilities) ** n)
        result = analysis.expected_richness(n)
        assert result["method"] == "exact_atoms"
        assert result["expected_richness"] == pytest.approx(expected, rel=2e-14)

    def test_native_discovery_curve_matches_exhaustive_atoms(self, recombining_graph, analysis):
        probabilities = np.asarray(
            [p for p, _, _ in enumerate_public_graph(recombining_graph)],
            dtype=np.longdouble,
        )
        draws = np.asarray([1.0, 1.5, 2.0, 10.0, 1e6])
        expected_richness = []
        expected_novelty = []
        for n in draws.astype(np.longdouble):
            log_survival = np.log1p(-probabilities)
            expected_richness.append(np.sum(-np.expm1(n * log_survival), dtype=np.longdouble))
            expected_novelty.append(
                np.sum(
                    probabilities * np.exp((n - 1) * log_survival),
                    dtype=np.longdouble,
                )
            )
        result = analysis.discovery_curve(draws)
        assert result["method"] == "exact_atoms"
        assert result["spectrum_normalized"] is False
        assert result["spectrum_mass_before_normalization"] == pytest.approx(1.0)
        np.testing.assert_allclose(
            result["expected_richness"],
            expected_richness,
            rtol=2e-14,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            result["novelty_probability"],
            expected_novelty,
            rtol=2e-14,
            atol=2e-14,
        )
        assert np.all(np.diff(result["expected_richness"]) >= 0)
        assert np.all(np.diff(result["novelty_probability"]) <= 0)

    def test_discovery_curve_shape_scalar_grid_and_validation(self, analysis):
        draws = np.asarray([[1.0, 2.0], [10.0, 100.0]])
        exact = analysis.discovery_curve(draws)
        assert exact["expected_richness"].shape == draws.shape
        assert exact["novelty_probability"].shape == draws.shape
        scalar = analysis.discovery_curve(10.0)
        assert isinstance(scalar["expected_richness"], float)
        assert isinstance(scalar["novelty_probability"], float)
        grid = analysis.discovery_curve(draws, bins=8192, max_exact_paths=1)
        assert grid["method"] == "deterministic_grid"
        assert grid["spectrum_normalized"] is True
        assert grid["spectrum_mass_before_normalization"] > 0
        assert grid["expected_richness"][0, 0] == pytest.approx(1.0, rel=2e-14)
        assert grid["novelty_probability"][0, 0] == pytest.approx(1.0, rel=2e-14)
        assert np.all(grid["novelty_probability"] <= 1.0 + 2e-14)
        np.testing.assert_allclose(grid["expected_richness"], exact["expected_richness"], rtol=2e-5)
        empty = analysis.discovery_curve(np.empty((0, 2)))
        assert empty["method"] == "empty"
        assert empty["spectrum_normalized"] is False
        assert empty["expected_richness"].shape == (0, 2)
        for bad in (0.0, -1.0, np.inf, np.nan):
            with pytest.raises(ValueError, match="at least one"):
                analysis.discovery_curve([bad])

    def test_expected_frequency_spectrum_matches_binomial_sum(self, recombining_graph, analysis):
        n = 12
        max_count = 6
        probabilities = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
        expected = []
        for r in range(max_count + 1):
            coefficient = comb(n, r)
            expected.append(np.sum(coefficient * probabilities**r * (1 - probabilities) ** (n - r)))
        result = analysis.expected_frequency_spectrum(n, max_count)
        assert result["method"] == "exact_atoms"
        np.testing.assert_allclose(result["expected_counts"], expected, rtol=2e-14, atol=2e-14)

    def test_grid_occupancy_converges_to_exact(self, analysis):
        exact = analysis.expected_richness(100)["expected_richness"]
        grid = analysis.expected_richness(100, bins=8192, max_exact_paths=1)
        assert grid["method"] == "deterministic_grid"
        assert grid["spectrum_normalized"] is True
        assert grid["expected_richness"] == pytest.approx(exact, rel=2e-5)

        first_draw = analysis.expected_richness(1, bins=8192, max_exact_paths=1)
        assert first_draw["expected_richness"] == pytest.approx(1.0, rel=2e-14)
        assert first_draw["spectrum_mass_before_normalization"] > 0

    def test_pair_collision_identity(self, recombining_graph, analysis):
        n = 20
        probabilities = np.array([p for p, _, _ in enumerate_public_graph(recombining_graph)])
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
    with pytest.raises(TypeError):
        analysis.histogram(length=7.5)
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
            [np.sum(probabilities**q * np.log(probabilities) ** order) for order in range(3)]
        )
        np.testing.assert_allclose(analysis.derivatives(q, 2), expected, rtol=3e-13, atol=3e-13)
        assert analysis.log_mellin(q) == pytest.approx(np.log(expected[0]), abs=3e-13)
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
