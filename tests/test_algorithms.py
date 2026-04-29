from __future__ import annotations

import pytest

from fscd.algorithms import run_pc
from fscd.data import generate_synthetic_instance
from fscd.graphs import is_dag


@pytest.mark.parametrize(
    ("nodes", "density", "n_samples", "graph_seed", "noise_seed"),
    [
        (15, 0.8, 10000, 4224628626, 3592345146),
        (6, 0.8, 500, 405600534, 1862250330),
    ],
)
def test_run_pc_avoids_known_cyclic_orientation_cases(
    nodes: int,
    density: float,
    n_samples: int,
    graph_seed: int,
    noise_seed: int,
) -> None:
    _, samples = generate_synthetic_instance(
        nodes=nodes,
        density=density,
        n_samples=n_samples,
        graph_seed=graph_seed,
        noise_seed=noise_seed,
    )

    predicted = run_pc(samples)

    assert is_dag(predicted)
