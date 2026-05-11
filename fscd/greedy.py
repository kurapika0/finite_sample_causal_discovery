from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass

import numpy as np

from fscd.algorithms import run_boss, score_dag_bic_from_cov
from fscd.graphs import adjacency_to_skeleton_upper, is_dag, topological_order


@dataclass(frozen=True)
class TopKStep:
    k: int
    adjacency: np.ndarray
    bic_score: float
    frontier_size: int
    permutation: tuple[int, ...]


@dataclass(frozen=True)
class GreedySearchResult:
    seed_graph: np.ndarray
    initial_skeleton: np.ndarray
    initial_permutation: tuple[int, ...]
    steps: tuple[TopKStep, ...]


def adjacency_key(adjacency: object) -> tuple[int, ...]:
    matrix = np.asarray(adjacency, dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square adjacency matrix.")
    return tuple(int(value) for value in matrix.reshape(-1).tolist())


def upper_triangle_pairs(num_nodes: int) -> tuple[tuple[int, int], ...]:
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive.")
    return tuple((left, right) for left in range(num_nodes) for right in range(left + 1, num_nodes))


def skeleton_upper_to_bits(skeleton_upper: object) -> tuple[int, ...]:
    matrix = np.asarray(skeleton_upper, dtype=int)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square upper-triangular skeleton matrix.")

    num_nodes = matrix.shape[0]
    bits: list[int] = []
    for left, right in upper_triangle_pairs(num_nodes):
        bits.append(int(matrix[left, right] != 0))
    return tuple(bits)


def bits_to_skeleton_upper(bits: tuple[int, ...], num_nodes: int) -> np.ndarray:
    skeleton = np.zeros((num_nodes, num_nodes), dtype=int)
    pairs = upper_triangle_pairs(num_nodes)
    if len(bits) != len(pairs):
        raise ValueError("Bit-vector length does not match the number of possible skeleton edges.")

    for bit, (left, right) in zip(bits, pairs):
        skeleton[left, right] = int(bit != 0)
    return skeleton


def orient_skeleton_by_permutation(skeleton_upper: object, permutation: tuple[int, ...] | list[int]) -> np.ndarray:
    skeleton = np.asarray(skeleton_upper, dtype=int)
    if skeleton.ndim != 2 or skeleton.shape[0] != skeleton.shape[1]:
        raise ValueError("Expected a square upper-triangular skeleton matrix.")

    order = tuple(int(node) for node in permutation)
    num_nodes = skeleton.shape[0]
    if len(order) != num_nodes or set(order) != set(range(num_nodes)):
        raise ValueError("Permutation must contain each node exactly once.")

    position = {node: index for index, node in enumerate(order)}
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    for left, right in upper_triangle_pairs(num_nodes):
        if skeleton[left, right] == 0:
            continue
        if position[left] < position[right]:
            adjacency[left, right] = 1
        else:
            adjacency[right, left] = 1

    if not is_dag(adjacency):
        raise ValueError("Orienting a skeleton by a permutation must produce a DAG.")
    return adjacency


def skeleton_neighbors_within_distance(skeleton_upper: object, epsilon: int) -> list[np.ndarray]:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    skeleton = np.asarray(skeleton_upper, dtype=int)
    num_nodes = skeleton.shape[0]
    base_bits = skeleton_upper_to_bits(skeleton)
    neighbors: set[tuple[int, ...]] = set()

    for flips in range(epsilon + 1):
        for changed_indices in itertools.combinations(range(len(base_bits)), flips):
            candidate_bits = list(base_bits)
            for index in changed_indices:
                candidate_bits[index] = 1 - candidate_bits[index]
            neighbors.add(tuple(candidate_bits))

    return [bits_to_skeleton_upper(bits, num_nodes) for bits in sorted(neighbors)]


def permutations_within_adjacent_transpositions(permutation: tuple[int, ...] | list[int], delta: int) -> list[tuple[int, ...]]:
    if delta < 0:
        raise ValueError("delta must be non-negative.")

    base = tuple(int(node) for node in permutation)
    if len(base) == 0:
        raise ValueError("Permutation must be non-empty.")

    seen = {base}
    queue: deque[tuple[tuple[int, ...], int]] = deque([(base, 0)])
    while queue:
        current, distance = queue.popleft()
        if distance == delta:
            continue

        for index in range(len(current) - 1):
            swapped = list(current)
            swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
            swapped_tuple = tuple(swapped)
            if swapped_tuple in seen:
                continue
            seen.add(swapped_tuple)
            queue.append((swapped_tuple, distance + 1))

    return sorted(seen)


def rank_candidate_graphs(candidates: list[np.ndarray], score_fn: callable) -> list[np.ndarray]:
    deduplicated: dict[tuple[int, ...], np.ndarray] = {}
    for candidate in candidates:
        deduplicated[adjacency_key(candidate)] = np.asarray(candidate, dtype=int)

    return sorted(
        deduplicated.values(),
        key=lambda adjacency: (-float(score_fn(adjacency)), adjacency_key(adjacency)),
    )


def select_next_frontier(
    frontier: list[np.ndarray],
    neighborhood: list[np.ndarray],
    selected_graphs: list[np.ndarray],
    k: int,
    score_fn: callable,
) -> list[np.ndarray]:
    selected_keys = {adjacency_key(graph) for graph in selected_graphs}
    merged = [
        np.asarray(graph, dtype=int)
        for graph in list(frontier) + list(neighborhood)
        if adjacency_key(graph) not in selected_keys
    ]
    return rank_candidate_graphs(merged, score_fn)[:k]


def run_greedy_search(
    samples: np.ndarray,
    epsilon: int,
    delta: int,
    k_max: int,
    seed_graph: np.ndarray | None = None,
) -> GreedySearchResult:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if delta < 0:
        raise ValueError("delta must be non-negative.")
    if k_max <= 0:
        raise ValueError("k_max must be positive.")

    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError("Expected samples with shape (n_samples, d).")

    current_seed_graph = run_boss(samples) if seed_graph is None else np.asarray(seed_graph, dtype=int)
    initial_skeleton = adjacency_to_skeleton_upper(current_seed_graph)
    initial_permutation = tuple(topological_order(current_seed_graph))
    covariance = np.cov(samples.T, ddof=0)
    n_samples = int(samples.shape[0])

    score_cache: dict[tuple[int, ...], float] = {}

    def score_fn(adjacency: np.ndarray) -> float:
        key = adjacency_key(adjacency)
        if key not in score_cache:
            score_cache[key] = score_dag_bic_from_cov(adjacency, covariance, n_samples)
        return score_cache[key]

    initial_graph = orient_skeleton_by_permutation(initial_skeleton, initial_permutation)
    frontier = [initial_graph]
    top_graphs = [initial_graph]
    steps = [
        TopKStep(
            k=1,
            adjacency=initial_graph.copy(),
            bic_score=score_fn(initial_graph),
            frontier_size=1,
            permutation=initial_permutation,
        )
    ]

    for current_k in range(2, k_max + 1):
        previous_graph = top_graphs[-1]
        previous_skeleton = adjacency_to_skeleton_upper(previous_graph)
        # Candidates are graphs, so we use the graph's lexicographically smallest
        # topological order as its canonical permutation representative.
        previous_permutation = tuple(topological_order(previous_graph))

        neighborhood: list[np.ndarray] = []
        for skeleton in skeleton_neighbors_within_distance(previous_skeleton, epsilon):
            for permutation in permutations_within_adjacent_transpositions(previous_permutation, delta):
                neighborhood.append(orient_skeleton_by_permutation(skeleton, permutation))

        frontier = select_next_frontier(frontier, neighborhood, top_graphs, current_k, score_fn)
        if not frontier:
            break

        next_graph = frontier[0]
        top_graphs.append(next_graph)
        steps.append(
            TopKStep(
                k=current_k,
                adjacency=next_graph.copy(),
                bic_score=score_fn(next_graph),
                frontier_size=len(frontier),
                permutation=tuple(topological_order(next_graph)),
            )
        )

    return GreedySearchResult(
        seed_graph=np.asarray(current_seed_graph, dtype=int),
        initial_skeleton=np.asarray(initial_skeleton, dtype=int),
        initial_permutation=initial_permutation,
        steps=tuple(steps),
    )
