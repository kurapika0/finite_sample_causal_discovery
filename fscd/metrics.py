from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fscd.graphs import adjacency_edge_list, adjacency_to_skeleton_upper, general_graph_to_adjacency, is_dag


def skeleton_distance(graph_a: object, graph_b: object) -> int:
    skeleton_a = adjacency_to_skeleton_upper(graph_a)
    skeleton_b = adjacency_to_skeleton_upper(graph_b)
    return int(np.count_nonzero(skeleton_a != skeleton_b))


def kendall_tau_distance(order_a: list[int], order_b: list[int]) -> int:
    if len(order_a) != len(order_b):
        raise ValueError("Orders must have the same length.")

    position_in_b = {node: idx for idx, node in enumerate(order_b)}
    inversions = 0
    for idx, left_node in enumerate(order_a):
        left_pos = position_in_b[left_node]
        for right_node in order_a[idx + 1 :]:
            if left_pos > position_in_b[right_node]:
                inversions += 1
    return inversions


def exact_permutation_distance(graph_a: object, graph_b: object) -> int:
    adjacency_a = general_graph_to_adjacency(graph_a)
    adjacency_b = general_graph_to_adjacency(graph_b)

    if adjacency_a.shape != adjacency_b.shape:
        raise ValueError("Both graphs must have the same shape.")
    if not is_dag(adjacency_a) or not is_dag(adjacency_b):
        raise ValueError("Permutation distance requires acyclic directed constraints.")

    num_nodes = adjacency_a.shape[0]
    if num_nodes <= 1:
        return 0

    pairs = [(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)]
    num_pairs = len(pairs)

    rank_a_start = 0
    pair_a_start = rank_a_start + num_nodes
    rank_b_start = pair_a_start + num_pairs
    pair_b_start = rank_b_start + num_nodes
    diff_start = pair_b_start + num_pairs
    num_variables = diff_start + num_pairs

    objective = np.zeros(num_variables, dtype=float)
    objective[diff_start:] = 1.0

    lower_bounds = np.zeros(num_variables, dtype=float)
    upper_bounds = np.ones(num_variables, dtype=float)

    lower_bounds[rank_a_start:pair_a_start] = 0.0
    upper_bounds[rank_a_start:pair_a_start] = float(num_nodes - 1)
    lower_bounds[rank_b_start:pair_b_start] = 0.0
    upper_bounds[rank_b_start:pair_b_start] = float(num_nodes - 1)

    integrality = np.ones(num_variables, dtype=int)

    rows: list[np.ndarray] = []
    lb: list[float] = []
    ub: list[float] = []

    for pair_index, (left, right) in enumerate(pairs):
        pair_a_var = pair_a_start + pair_index
        pair_b_var = pair_b_start + pair_index
        diff_var = diff_start + pair_index

        row = np.zeros(num_variables, dtype=float)
        row[rank_a_start + left] = 1.0
        row[rank_a_start + right] = -1.0
        row[pair_a_var] = float(num_nodes)
        rows.append(row)
        lb.append(-math.inf)
        ub.append(float(num_nodes - 1))

        row = np.zeros(num_variables, dtype=float)
        row[rank_a_start + right] = 1.0
        row[rank_a_start + left] = -1.0
        row[pair_a_var] = -float(num_nodes)
        rows.append(row)
        lb.append(-math.inf)
        ub.append(-1.0)

        row = np.zeros(num_variables, dtype=float)
        row[rank_b_start + left] = 1.0
        row[rank_b_start + right] = -1.0
        row[pair_b_var] = float(num_nodes)
        rows.append(row)
        lb.append(-math.inf)
        ub.append(float(num_nodes - 1))

        row = np.zeros(num_variables, dtype=float)
        row[rank_b_start + right] = 1.0
        row[rank_b_start + left] = -1.0
        row[pair_b_var] = -float(num_nodes)
        rows.append(row)
        lb.append(-math.inf)
        ub.append(-1.0)

        row = np.zeros(num_variables, dtype=float)
        row[pair_a_var] = 1.0
        row[pair_b_var] = -1.0
        row[diff_var] = -1.0
        rows.append(row)
        lb.append(-math.inf)
        ub.append(0.0)

        row = np.zeros(num_variables, dtype=float)
        row[pair_a_var] = -1.0
        row[pair_b_var] = 1.0
        row[diff_var] = -1.0
        rows.append(row)
        lb.append(-math.inf)
        ub.append(0.0)

    for src, dst in adjacency_edge_list(adjacency_a):
        row = np.zeros(num_variables, dtype=float)
        row[rank_a_start + src] = 1.0
        row[rank_a_start + dst] = -1.0
        rows.append(row)
        lb.append(-math.inf)
        ub.append(-1.0)

    for src, dst in adjacency_edge_list(adjacency_b):
        row = np.zeros(num_variables, dtype=float)
        row[rank_b_start + src] = 1.0
        row[rank_b_start + dst] = -1.0
        rows.append(row)
        lb.append(-math.inf)
        ub.append(-1.0)

    constraints = LinearConstraint(np.vstack(rows), np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options={"presolve": True},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed with status {result.status}: {result.message}")

    return int(round(float(result.fun)))
