from __future__ import annotations

import numpy as np

from fscd.graphs import enumerate_topological_orders
from fscd.metrics import exact_permutation_distance, kendall_tau_distance, skeleton_distance


def brute_force_permutation_distance(adjacency_a: np.ndarray, adjacency_b: np.ndarray) -> int:
    best_distance: int | None = None
    for order_a in enumerate_topological_orders(adjacency_a):
        for order_b in enumerate_topological_orders(adjacency_b):
            distance = kendall_tau_distance(order_a, order_b)
            if best_distance is None or distance < best_distance:
                best_distance = distance
    assert best_distance is not None
    return best_distance


def test_skeleton_distance_counts_undirected_edge_mismatches() -> None:
    graph_a = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    graph_b = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    assert skeleton_distance(graph_a, graph_b) == 2


def test_kendall_tau_distance_matches_adjacent_swap_count() -> None:
    assert kendall_tau_distance([0, 1, 2, 3], [3, 2, 1, 0]) == 6
    assert kendall_tau_distance([3, 0, 1, 2], [3, 0, 2, 1]) == 1


def test_exact_permutation_distance_uses_best_topological_orders() -> None:
    graph_a = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    graph_b = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    assert exact_permutation_distance(graph_a, graph_b) == 0


def test_exact_permutation_distance_matches_bruteforce_on_small_dags() -> None:
    graph_a = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )
    graph_b = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )
    assert exact_permutation_distance(graph_a, graph_b) == brute_force_permutation_distance(graph_a, graph_b)

