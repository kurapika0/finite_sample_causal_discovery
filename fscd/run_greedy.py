from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fscd.algorithms import run_boss
from fscd.data import generate_synthetic_instance
from fscd.graphs import weighted_sem_to_adjacency
from fscd.greedy import run_greedy_search
from fscd.metrics import exact_permutation_distance, skeleton_distance
from fscd.reporting import plot_topk_distances, summarize_recovery_results, summarize_topk_results

DEFAULT_NODES = (5,)
DEFAULT_DENSITIES = (0.4,)
DEFAULT_SAMPLE_SIZES = (20, 500, 10000)
DEFAULT_EPSILONS = (1, 2)
DEFAULT_DELTAS = (1, 2)
DEFAULT_K_MAX = 10
DEFAULT_RUNS = 50
DEFAULT_SEED = 0
DEFAULT_OUTPUT = "results/greedy_default"
DEFAULT_PLOT_SAMPLE_SIZES = (20, 500, 10000)
DEFAULT_PLOT_EPSILON = 1
DEFAULT_PLOT_DELTA = 1


@dataclass
class GreedyBenchmarkConfig:
    nodes: tuple[int, ...]
    densities: tuple[float, ...]
    sample_sizes: tuple[int, ...]
    epsilons: tuple[int, ...]
    deltas: tuple[int, ...]
    k_max: int
    runs: int
    seed: int
    output_dir: Path
    plot_sample_sizes: tuple[int, ...]
    plot_epsilon: int
    plot_delta: int

    @classmethod
    def from_namespace(cls, namespace: object) -> "GreedyBenchmarkConfig":
        if namespace.plot_sample_sizes is not None:
            plot_sample_sizes = tuple(int(value) for value in namespace.plot_sample_sizes)
        elif namespace.plot_sample_size_legacy is not None:
            plot_sample_sizes = tuple(int(value) for value in namespace.plot_sample_size_legacy)
        else:
            plot_sample_sizes = tuple(DEFAULT_PLOT_SAMPLE_SIZES)

        config = cls(
            nodes=tuple(int(value) for value in namespace.nodes),
            densities=tuple(float(value) for value in namespace.densities),
            sample_sizes=tuple(int(value) for value in namespace.sample_sizes),
            epsilons=tuple(int(value) for value in namespace.epsilons),
            deltas=tuple(int(value) for value in namespace.deltas),
            k_max=int(namespace.k_max),
            runs=int(namespace.runs),
            seed=int(namespace.seed),
            output_dir=Path(namespace.output),
            plot_sample_sizes=plot_sample_sizes,
            plot_epsilon=int(namespace.plot_epsilon),
            plot_delta=int(namespace.plot_delta),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.nodes or any(node <= 0 for node in self.nodes):
            raise ValueError("All node counts must be positive.")
        if not self.densities or any(density < 0.0 or density > 1.0 for density in self.densities):
            raise ValueError("All densities must be in [0, 1].")
        if not self.sample_sizes or any(size <= 0 for size in self.sample_sizes):
            raise ValueError("All sample sizes must be positive.")
        if not self.epsilons or any(epsilon < 0 for epsilon in self.epsilons):
            raise ValueError("All epsilons must be non-negative.")
        if not self.deltas or any(delta < 0 for delta in self.deltas):
            raise ValueError("All deltas must be non-negative.")
        if self.k_max <= 0:
            raise ValueError("k_max must be positive.")
        if self.runs <= 0:
            raise ValueError("runs must be positive.")
        if not self.plot_sample_sizes or any(size <= 0 for size in self.plot_sample_sizes):
            raise ValueError("plot_sample_sizes must contain positive values.")
        if self.plot_epsilon < 0:
            raise ValueError("plot_epsilon must be non-negative.")
        if self.plot_delta < 0:
            raise ValueError("plot_delta must be non-negative.")

    def to_metadata(self) -> dict[str, object]:
        metadata = asdict(self)
        metadata["output_dir"] = str(self.output_dir)
        return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BOSS-initialized greedy Top-K search benchmark")
    parser.add_argument("--nodes", nargs="+", type=int, default=list(DEFAULT_NODES))
    parser.add_argument("--densities", nargs="+", type=float, default=list(DEFAULT_DENSITIES))
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=list(DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--epsilons", nargs="+", type=int, default=list(DEFAULT_EPSILONS))
    parser.add_argument("--deltas", nargs="+", type=int, default=list(DEFAULT_DELTAS))
    parser.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-sample-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--plot-sample-size", dest="plot_sample_size_legacy", action="append", type=int, default=None)
    parser.add_argument("--plot-epsilon", type=int, default=DEFAULT_PLOT_EPSILON)
    parser.add_argument("--plot-delta", type=int, default=DEFAULT_PLOT_DELTA)
    return parser


def save_metadata(config: GreedyBenchmarkConfig, output_dir: Path) -> None:
    metadata = config.to_metadata()
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["python_version"] = sys.version
    metadata["platform"] = platform.platform()
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _empty_row(
    *,
    run_id: int,
    nodes: int,
    density: float,
    n_samples: int,
    epsilon: int,
    delta: int,
    k: int,
    graph_seed: int,
    noise_seed: int,
    status: str,
    error_message: str,
    found_true_by_k: int = 0,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "d": nodes,
        "density": density,
        "n_samples": n_samples,
        "epsilon": epsilon,
        "delta": delta,
        "k": k,
        "ds": math.nan,
        "dp": math.nan,
        "bic_score": math.nan,
        "is_true_at_k": 0,
        "found_true_by_k": found_true_by_k,
        "graph_seed": graph_seed,
        "noise_seed": noise_seed,
        "frontier_size": 0,
        "status": status,
        "error_message": error_message,
    }


def benchmark_greedy(
    config: GreedyBenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config.validate()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(config, output_dir)

    raw_results_path = output_dir / "raw_topk_results.csv"
    summary_results_path = output_dir / "summary_topk_results.csv"
    recovery_results_path = output_dir / "recovery_summary.csv"

    seed_rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    total_jobs = (
        len(config.nodes)
        * len(config.densities)
        * len(config.sample_sizes)
        * config.runs
        * len(config.epsilons)
        * len(config.deltas)
    )
    completed_jobs = 0

    for nodes in config.nodes:
        for density in config.densities:
            for n_samples in config.sample_sizes:
                for run_id in range(config.runs):
                    graph_seed = int(seed_rng.integers(0, 2**32 - 1))
                    noise_seed = int(seed_rng.integers(0, 2**32 - 1))
                    weighted_true_graph, samples = generate_synthetic_instance(
                        nodes=nodes,
                        density=density,
                        n_samples=n_samples,
                        graph_seed=graph_seed,
                        noise_seed=noise_seed,
                    )
                    true_adjacency = weighted_sem_to_adjacency(weighted_true_graph)

                    seed_graph: np.ndarray | None = None
                    seed_error_message = ""
                    try:
                        seed_graph = run_boss(samples)
                    except Exception as exc:  # pragma: no cover - integration path
                        seed_error_message = str(exc)

                    for epsilon in config.epsilons:
                        for delta in config.deltas:
                            completed_jobs += 1
                            print(
                                f"[{completed_jobs}/{total_jobs}] "
                                f"d={nodes} density={density} n={n_samples} run={run_id + 1}/{config.runs} "
                                f"epsilon={epsilon} delta={delta}",
                                flush=True,
                            )

                            if seed_graph is None:
                                for k in range(1, config.k_max + 1):
                                    rows.append(
                                        _empty_row(
                                            run_id=run_id,
                                            nodes=nodes,
                                            density=density,
                                            n_samples=n_samples,
                                            epsilon=epsilon,
                                            delta=delta,
                                            k=k,
                                            graph_seed=graph_seed,
                                            noise_seed=noise_seed,
                                            status="error:boss_initialization",
                                            error_message=seed_error_message,
                                        )
                                    )
                                continue

                            try:
                                search_result = run_greedy_search(
                                    samples=samples,
                                    epsilon=epsilon,
                                    delta=delta,
                                    k_max=config.k_max,
                                    seed_graph=seed_graph,
                                )
                            except Exception as exc:  # pragma: no cover - integration path
                                for k in range(1, config.k_max + 1):
                                    rows.append(
                                        _empty_row(
                                            run_id=run_id,
                                            nodes=nodes,
                                            density=density,
                                            n_samples=n_samples,
                                            epsilon=epsilon,
                                            delta=delta,
                                            k=k,
                                            graph_seed=graph_seed,
                                            noise_seed=noise_seed,
                                            status=f"error:{type(exc).__name__}",
                                            error_message=str(exc),
                                        )
                                    )
                                continue

                            found_true_by_k = 0
                            steps_by_k = {step.k: step for step in search_result.steps}
                            for k in range(1, config.k_max + 1):
                                step = steps_by_k.get(k)
                                if step is None:
                                    rows.append(
                                        _empty_row(
                                            run_id=run_id,
                                            nodes=nodes,
                                            density=density,
                                            n_samples=n_samples,
                                            epsilon=epsilon,
                                            delta=delta,
                                            k=k,
                                            graph_seed=graph_seed,
                                            noise_seed=noise_seed,
                                            status="exhausted_candidates",
                                            error_message="Search frontier exhausted before reaching the requested K.",
                                            found_true_by_k=found_true_by_k,
                                        )
                                    )
                                    continue

                                is_true_at_k = int(np.array_equal(step.adjacency, true_adjacency))
                                found_true_by_k = int(found_true_by_k or is_true_at_k)
                                rows.append(
                                    {
                                        "run_id": run_id,
                                        "d": nodes,
                                        "density": density,
                                        "n_samples": n_samples,
                                        "epsilon": epsilon,
                                        "delta": delta,
                                        "k": k,
                                        "ds": int(skeleton_distance(true_adjacency, step.adjacency)),
                                        "dp": int(exact_permutation_distance(true_adjacency, step.adjacency)),
                                        "bic_score": float(step.bic_score),
                                        "is_true_at_k": is_true_at_k,
                                        "found_true_by_k": found_true_by_k,
                                        "graph_seed": graph_seed,
                                        "noise_seed": noise_seed,
                                        "frontier_size": int(step.frontier_size),
                                        "status": "ok",
                                        "error_message": "",
                                    }
                                )

    raw_results = pd.DataFrame(rows).sort_values(
        ["d", "density", "n_samples", "run_id", "epsilon", "delta", "k"]
    ).reset_index(drop=True)
    raw_results.to_csv(raw_results_path, index=False)

    summary_results = summarize_topk_results(raw_results)
    summary_results.to_csv(summary_results_path, index=False)

    recovery_results = summarize_recovery_results(raw_results, config.k_max)
    recovery_results.to_csv(recovery_results_path, index=False)

    for plot_sample_size in dict.fromkeys(config.plot_sample_sizes):
        plot_topk_distances(
            summary_results=summary_results,
            output_dir=output_dir,
            nodes=config.nodes[0],
            density=config.densities[0],
            epsilon=config.plot_epsilon,
            delta=config.plot_delta,
            n_samples=int(plot_sample_size),
        )

    return raw_results, summary_results, recovery_results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = GreedyBenchmarkConfig.from_namespace(args)
    benchmark_greedy(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
