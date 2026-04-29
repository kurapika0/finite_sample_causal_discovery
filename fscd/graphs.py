from __future__ import annotations

import heapq
from collections.abc import Iterator

import numpy as np


def graph_to_matrix(graph_or_matrix: object) -> np.ndarray:
    return np.asarray(getattr(graph_or_matrix, "graph", graph_or_matrix), dtype=int)


def _is_adjacency_matrix(matrix: np.ndarray) -> bool:
    return bool(np.all((matrix == 0) | (matrix == 1)))


def weighted_sem_to_adjacency(weighted_adjacency: np.ndarray) -> np.ndarray:
    weighted_adjacency = np.asarray(weighted_adjacency, dtype=float)
    return (np.abs(weighted_adjacency) > 1e-12).astype(int)


def general_graph_to_adjacency(graph_or_matrix: object) -> np.ndarray:
    matrix = graph_to_matrix(graph_or_matrix)
    if _is_adjacency_matrix(matrix):
        return matrix.copy()
    assert_supported_general_graph(matrix)

    num_nodes = matrix.shape[0]
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)

    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src == dst:
                continue
            if matrix[dst, src] == 1 and matrix[src, dst] == -1:
                adjacency[src, dst] = 1

    return adjacency


def adjacency_to_skeleton_upper(graph_or_matrix: object) -> np.ndarray:
    matrix = graph_to_matrix(graph_or_matrix)
    if not _is_adjacency_matrix(matrix):
        assert_supported_general_graph(matrix)
    skeleton = ((matrix != 0) | (matrix.T != 0)).astype(int)
    return np.triu(skeleton, k=1)


def adjacency_edge_list(adjacency: np.ndarray) -> list[tuple[int, int]]:
    adjacency = np.asarray(adjacency, dtype=int)
    rows, cols = np.nonzero(adjacency)
    return [(int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist())]


def topological_order(graph_or_matrix: object) -> list[int]:
    adjacency = general_graph_to_adjacency(graph_or_matrix)
    indegree = adjacency.sum(axis=0).astype(int).tolist()
    children = [np.flatnonzero(adjacency[node]).astype(int).tolist() for node in range(adjacency.shape[0])]
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(queue)

    order: list[int] = []
    while queue:
        node = heapq.heappop(queue)
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(queue, child)

    if len(order) != adjacency.shape[0]:
        raise ValueError("Adjacency matrix is not a DAG.")

    return order


def is_dag(graph_or_matrix: object) -> bool:
    try:
        topological_order(graph_or_matrix)
    except ValueError:
        return False
    return True


def enumerate_topological_orders(graph_or_matrix: object) -> Iterator[list[int]]:
    adjacency = general_graph_to_adjacency(graph_or_matrix)
    num_nodes = adjacency.shape[0]
    indegree = adjacency.sum(axis=0).astype(int)
    available = [node for node in range(num_nodes) if indegree[node] == 0]

    def backtrack(order: list[int], current_indegree: np.ndarray, current_available: list[int]) -> Iterator[list[int]]:
        if len(order) == num_nodes:
            yield list(order)
            return

        for node in sorted(current_available):
            next_order = order + [node]
            next_indegree = current_indegree.copy()
            next_available = [candidate for candidate in current_available if candidate != node]

            for child in np.flatnonzero(adjacency[node]):
                next_indegree[child] -= 1
                if next_indegree[child] == 0:
                    next_available.append(int(child))

            yield from backtrack(next_order, next_indegree, next_available)

    yield from backtrack([], indegree, available)


def assert_supported_general_graph(graph_or_matrix: object) -> None:
    matrix = graph_to_matrix(graph_or_matrix)
    if _is_adjacency_matrix(matrix):
        return

    supported_pairs = {(0, 0), (-1, -1), (-1, 1), (1, -1)}

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            pair = (int(matrix[i, j]), int(matrix[j, i]))
            if pair not in supported_pairs:
                raise ValueError(f"Unsupported graph endpoint pair {pair} between nodes {i} and {j}.")


def pdag_to_dag_adjacency(graph: object) -> np.ndarray:
    general_graph = getattr(graph, "G", graph)
    assert_supported_general_graph(general_graph)
    matrix = graph_to_matrix(general_graph)
    adjacency = general_graph_to_adjacency(matrix)
    order = topological_order(adjacency)
    order_index = {node: index for index, node in enumerate(order)}

    # Orient every undirected edge according to a topological order that already
    # respects the compelled directions, which avoids the infinite loop in
    # causallearn's PDAG2DAG helper on some sparse 4-cycle structures.
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] == -1 and matrix[j, i] == -1:
                if order_index[i] < order_index[j]:
                    adjacency[i, j] = 1
                else:
                    adjacency[j, i] = 1

    if not is_dag(adjacency):
        raise ValueError("Normalized graph is not acyclic.")
    return adjacency
