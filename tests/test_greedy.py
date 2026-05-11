from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fscd.algorithms import run_boss
from fscd.data import generate_synthetic_instance
from fscd.graphs import is_dag
from fscd.greedy import (
    adjacency_key,
    orient_skeleton_by_permutation,
    permutations_within_adjacent_transpositions,
    rank_candidate_graphs,
    run_greedy_search,
    select_next_frontier,
    skeleton_neighbors_within_distance,
)
from fscd.metrics import kendall_tau_distance, skeleton_distance
from fscd.reporting import summarize_recovery_results
from fscd.run_greedy import GreedyBenchmarkConfig, benchmark_greedy


def test_orient_skeleton_by_permutation_builds_a_dag() -> None:
    skeleton = np.array(
        [
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    adjacency = orient_skeleton_by_permutation(skeleton, (1, 0, 3, 2))

    expected = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(adjacency, expected)
    assert is_dag(adjacency)


def test_skeleton_neighbors_respect_epsilon() -> None:
    skeleton = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ]
    )

    neighbors = skeleton_neighbors_within_distance(skeleton, epsilon=1)

    assert any(np.array_equal(neighbor, skeleton) for neighbor in neighbors)
    for neighbor in neighbors:
        assert skeleton_distance(skeleton, neighbor) <= 1


def test_permutation_neighbors_respect_adjacent_transposition_budget() -> None:
    base = (0, 1, 2, 3)
    neighbors = permutations_within_adjacent_transpositions(base, delta=2)

    assert base in neighbors
    for neighbor in neighbors:
        assert kendall_tau_distance(list(base), list(neighbor)) <= 2


def test_candidate_ranking_deduplicates_by_final_dag() -> None:
    candidate_a = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    candidate_b = candidate_a.copy()
    candidate_c = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )

    ranked = rank_candidate_graphs(
        [candidate_a, candidate_b, candidate_c],
        score_fn=lambda adjacency: float(np.count_nonzero(adjacency)),
    )

    assert len(ranked) == 2
    assert adjacency_key(ranked[0]) == adjacency_key(candidate_c)


def test_frontier_selection_removes_previously_selected_graphs_and_truncates() -> None:
    graph_1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    graph_2 = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    graph_3 = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )
    graph_4 = np.array(
        [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ]
    )

    frontier = select_next_frontier(
        frontier=[graph_1, graph_2],
        neighborhood=[graph_1, graph_3, graph_4],
        selected_graphs=[graph_2],
        k=2,
        score_fn=lambda adjacency: float(10 * np.count_nonzero(adjacency) + adjacency[0, 1]),
    )

    assert len(frontier) == 2
    assert all(adjacency_key(graph) != adjacency_key(graph_2) for graph in frontier)
    assert adjacency_key(frontier[0]) == adjacency_key(graph_4)
    assert adjacency_key(frontier[1]) == adjacency_key(graph_3)


def test_recovery_summary_matches_raw_topk_flags() -> None:
    raw_results = pd.DataFrame(
        [
            {"d": 5, "density": 0.4, "n_samples": 20, "epsilon": 1, "delta": 1, "run_id": 0, "k": 1, "is_true_at_k": 0, "status": "ok"},
            {"d": 5, "density": 0.4, "n_samples": 20, "epsilon": 1, "delta": 1, "run_id": 0, "k": 2, "is_true_at_k": 1, "status": "ok"},
            {"d": 5, "density": 0.4, "n_samples": 20, "epsilon": 1, "delta": 1, "run_id": 0, "k": 3, "is_true_at_k": 0, "status": "ok"},
            {"d": 5, "density": 0.4, "n_samples": 20, "epsilon": 1, "delta": 1, "run_id": 1, "k": 1, "is_true_at_k": 0, "status": "ok"},
            {"d": 5, "density": 0.4, "n_samples": 20, "epsilon": 1, "delta": 1, "run_id": 1, "k": 2, "is_true_at_k": 0, "status": "ok"},
            {"d": 5, "density": 0.4, "n_samples": 20, "epsilon": 1, "delta": 1, "run_id": 1, "k": 3, "is_true_at_k": 0, "status": "ok"},
        ]
    )

    summary = summarize_recovery_results(raw_results, k_max=3)

    assert summary.shape[0] == 1
    row = summary.iloc[0]
    assert int(row["n_success"]) == 1
    assert float(row["success_rate_within_kmax"]) == 0.5
    assert float(row["recover_k_mean_found_only"]) == 2.0


def test_run_greedy_search_uses_boss_seed_graph() -> None:
    _, samples = generate_synthetic_instance(
        nodes=5,
        density=0.4,
        n_samples=200,
        graph_seed=123,
        noise_seed=456,
    )
    boss_graph = run_boss(samples)

    result = run_greedy_search(samples=samples, epsilon=1, delta=1, k_max=3, seed_graph=boss_graph)

    np.testing.assert_array_equal(result.seed_graph, np.asarray(boss_graph, dtype=int))
    assert len(result.steps) == 3


def test_benchmark_greedy_smoke(monkeypatch, tmp_path: Path) -> None:
    def fake_run_boss(samples: np.ndarray) -> np.ndarray:
        del samples
        return np.array(
            [
                [0, -1, 0],
                [-1, 0, -1],
                [0, -1, 0],
            ]
        )

    monkeypatch.setattr("fscd.run_greedy.run_boss", fake_run_boss)
    monkeypatch.setattr("fscd.greedy.run_boss", fake_run_boss)

    config = GreedyBenchmarkConfig(
        nodes=(3,),
        densities=(0.5,),
        sample_sizes=(20,),
        epsilons=(1,),
        deltas=(1,),
        k_max=3,
        runs=1,
        seed=0,
        output_dir=tmp_path / "greedy_smoke",
        plot_sample_sizes=(20,),
        plot_epsilon=1,
        plot_delta=1,
    )

    raw_results, summary_results, recovery_results = benchmark_greedy(config)

    assert not raw_results.empty
    assert set(raw_results["k"].tolist()) == {1, 2, 3}
    assert (config.output_dir / "raw_topk_results.csv").exists()
    assert (config.output_dir / "summary_topk_results.csv").exists()
    assert (config.output_dir / "recovery_summary.csv").exists()
    assert (config.output_dir / "plots" / "topk_distances_d3_density0p5_eps1_delta1_n20.png").exists()
    assert not summary_results.empty
    assert not recovery_results.empty

    found_true_by_k = raw_results.sort_values("k")["found_true_by_k"].tolist()
    assert found_true_by_k == sorted(found_true_by_k)
