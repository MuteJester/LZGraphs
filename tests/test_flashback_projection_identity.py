"""Exact entropy/KL identity for an unsmoothed fitted FlashBackGraph.

For a weighted empirical sequence distribution P and the unsmoothed graph
fitted to exactly those weighted paths, the graph distribution Q preserves
the empirical edge-use marginals.  Consequently

    H(Q) = H(P, Q)

and therefore

    H(Q) - H(P) = KL(P || Q).

These tests enumerate every graph path on deliberately small examples rather
than relying only on the analytical entropy implementation.
"""

from __future__ import annotations

import math
from collections import defaultdict

import pytest

from LZGraphs import FlashBackGraph, flashback_decompose, flashback_reverse


def _enumerate_paths(
    graph: FlashBackGraph,
) -> list[tuple[tuple[str, ...], str, float]]:
    """Return every (token path, reconstructed sequence, probability)."""

    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    nodes = set(graph.all_nodes)
    for source, target, weight, _count in graph.all_edges:
        adjacency[source].append((target, weight))

    roots = [node for node in nodes if node.startswith("@")]
    assert roots == ["@$_1{0}"]
    root = roots[0]
    out: list[tuple[tuple[str, ...], str, float]] = []

    def visit(node: str, path: tuple[str, ...], probability: float) -> None:
        successors = adjacency.get(node, [])
        if not successors:
            sequence = flashback_reverse(list(path))
            # Every graph-generated path must still be the canonical,
            # deterministic FlashBack decomposition of its sequence.
            assert tuple(flashback_decompose(sequence)) == path
            out.append((path, sequence, probability))
            return
        for target, weight in successors:
            visit(target, path + (target,), probability * weight)

    visit(root, (root,), 1.0)
    return out


def _empirical_metrics(
    graph: FlashBackGraph,
    sequences: list[str],
    counts: list[int],
) -> tuple[float, float, float]:
    total = float(sum(counts))
    probabilities = [count / total for count in counts]
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
    )
    cross_entropy = -sum(
        probability * float(graph.pgen(sequence))
        for sequence, probability in zip(sequences, probabilities)
    )
    kl_divergence = cross_entropy - entropy
    return entropy, cross_entropy, kl_divergence


def _path_entropy(
    paths: list[tuple[tuple[str, ...], str, float]],
) -> float:
    return -sum(
        probability * math.log(probability)
        for _path, _sequence, probability in paths
        if probability > 0.0
    )


def test_weighted_recombining_graph_gap_is_exact_kl() -> None:
    # The two long observed paths merge at BB_1{2} and then split.  The graph
    # therefore supports two additional recombinant paths (five versus three
    # source identities), making the entropy gap non-zero.
    sequences = ["AA", "ABABA", "ABCBC"]
    counts = [2, 3, 5]
    graph = FlashBackGraph(sequences, abundances=counts, smoothing=0.0)

    paths = _enumerate_paths(graph)
    path_sequences = [sequence for _path, sequence, _probability in paths]
    path_probabilities = [
        probability for _path, _sequence, probability in paths
    ]
    assert len(paths) == 5
    assert len(set(path_sequences)) == len(paths)
    assert sum(path_probabilities) == pytest.approx(1.0)
    assert {"ABABA", "ABCBC"} < set(path_sequences)

    h_p, cross_entropy, kl_divergence = _empirical_metrics(
        graph,
        sequences,
        counts,
    )
    h_q = _path_entropy(paths)
    analytical_h_q = graph.diversity_profile()["entropy_nats"]

    assert h_q == pytest.approx(analytical_h_q, abs=1e-13)
    assert cross_entropy == pytest.approx(h_q, abs=1e-13)
    assert h_q - h_p == pytest.approx(kl_divergence, abs=1e-13)
    assert kl_divergence > 0.0


def test_no_recombination_has_zero_projection_gap() -> None:
    sequences = ["ABBA", "ACCA", "DBBE", "DCCE"]
    counts = [8, 2, 3, 7]
    graph = FlashBackGraph(sequences, abundances=counts, smoothing=0.0)

    paths = _enumerate_paths(graph)
    h_p, cross_entropy, kl_divergence = _empirical_metrics(
        graph,
        sequences,
        counts,
    )
    h_q = _path_entropy(paths)

    assert len(paths) == len(sequences)
    assert cross_entropy == pytest.approx(h_q, abs=1e-13)
    assert h_q == pytest.approx(h_p, abs=1e-13)
    assert kl_divergence == pytest.approx(0.0, abs=1e-13)


def test_smoothing_breaks_cross_entropy_equals_graph_entropy() -> None:
    sequences = ["AA", "ABABA", "ABCBC"]
    counts = [2, 3, 5]
    graph = FlashBackGraph(sequences, abundances=counts, smoothing=1.0)

    paths = _enumerate_paths(graph)
    h_p, cross_entropy, kl_divergence = _empirical_metrics(
        graph,
        sequences,
        counts,
    )
    h_q = _path_entropy(paths)

    assert cross_entropy != pytest.approx(h_q, abs=1e-6)
    assert h_q - h_p != pytest.approx(kl_divergence, abs=1e-6)

