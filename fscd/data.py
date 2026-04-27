from __future__ import annotations

import numpy as np

LOWER_BOUND_WEIGHT_VALUE = 0.5
UPPER_BOUND_WEIGHT_VALUE = 2.0
LOWER_BOUND_VARIANCE_VALUE = 0.1
UPPER_BOUND_VARIANCE_VALUE = 1.0


def generate_random_adjacency_matrix(
    nodes: int,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a weighted upper-triangular DAG matrix B where i -> j iff B[i, j] != 0."""
    adjacency = np.zeros((nodes, nodes), dtype=float)
    possible_edges = [(i, j) for i in range(nodes) for j in range(i + 1, nodes)]
    num_possible_edges = len(possible_edges)
    num_edges = int(num_possible_edges * density)

    if num_edges == 0:
        return adjacency

    chosen_indices = rng.choice(num_possible_edges, size=num_edges, replace=False)
    for idx in chosen_indices:
        i, j = possible_edges[int(idx)]
        magnitude = rng.uniform(LOWER_BOUND_WEIGHT_VALUE, UPPER_BOUND_WEIGHT_VALUE)
        sign = -1.0 if rng.random() < 0.5 else 1.0
        adjacency[i, j] = sign * magnitude

    return adjacency


def generate_dataset_from_adjacency_matrix(
    weighted_adjacency: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate X with shape (n_samples, d) from a linear Gaussian SEM."""
    weighted_adjacency = np.asarray(weighted_adjacency, dtype=float)
    num_nodes = weighted_adjacency.shape[0]
    identity = np.eye(num_nodes)
    variances = rng.uniform(
        LOWER_BOUND_VARIANCE_VALUE,
        UPPER_BOUND_VARIANCE_VALUE,
        size=num_nodes,
    )
    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(variances)[:, None],
        size=(num_nodes, n_samples),
    )
    samples = np.linalg.solve(identity - weighted_adjacency.T, noise)
    return samples.T


def generate_synthetic_instance(
    nodes: int,
    density: float,
    n_samples: int,
    graph_seed: int,
    noise_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    graph_rng = np.random.default_rng(graph_seed)
    noise_rng = np.random.default_rng(noise_seed)
    weighted_adjacency = generate_random_adjacency_matrix(nodes, density, graph_rng)
    samples = generate_dataset_from_adjacency_matrix(weighted_adjacency, n_samples, noise_rng)
    return weighted_adjacency, samples

