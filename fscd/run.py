from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fscd.algorithms import SUPPORTED_ALGORITHMS, run_algorithm
from fscd.config import (
    DEFAULT_ALGORITHMS,
    DEFAULT_CHECKPOINT_INTERVAL,
    DEFAULT_DENSITIES,
    DEFAULT_NODES,
    DEFAULT_OUTPUT,
    DEFAULT_RUNS,
    DEFAULT_SAMPLE_SIZES,
    DEFAULT_SEED,
    BenchmarkConfig,
)
from fscd.data import generate_synthetic_instance
from fscd.graphs import weighted_sem_to_adjacency
from fscd.metrics import exact_permutation_distance, skeleton_distance
from fscd.reporting import plot_results, summarize_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finite-sample causal discovery benchmark")
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--nodes", nargs="+", type=int, default=list(DEFAULT_NODES))
    parser.add_argument("--densities", nargs="+", type=float, default=list(DEFAULT_DENSITIES))
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=list(DEFAULT_SAMPLE_SIZES))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    return parser


def save_metadata(config: BenchmarkConfig, output_dir: Path) -> None:
    metadata = config.to_metadata()
    metadata["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["python_version"] = sys.version
    metadata["platform"] = platform.platform()
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_checkpoint(rows: list[dict[str, object]], raw_results_path: Path) -> None:
    raw_results = pd.DataFrame(rows)
    raw_results.to_csv(raw_results_path, index=False)


def benchmark(config: BenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    config.validate()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_dir / "raw_results.csv"
    summary_results_path = output_dir / "summary_results.csv"

    save_metadata(config, output_dir)

    seed_rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    total_jobs = (
        len(config.nodes)
        * len(config.densities)
        * len(config.sample_sizes)
        * config.runs
        * len(config.algorithms)
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

                    for algorithm in config.algorithms:
                        completed_jobs += 1
                        print(
                            f"[{completed_jobs}/{total_jobs}] "
                            f"algorithm={algorithm} d={nodes} density={density} n={n_samples} run={run_id + 1}/{config.runs}",
                            flush=True,
                        )

                        start_time = time.perf_counter()
                        status = "ok"
                        ds = float("nan")
                        dp = float("nan")
                        error_message = ""

                        try:
                            predicted_adjacency = run_algorithm(algorithm, samples)
                            ds = float(skeleton_distance(true_adjacency, predicted_adjacency))
                            dp = float(exact_permutation_distance(true_adjacency, predicted_adjacency))
                        except Exception as exc:  # pragma: no cover - exercised in smoke runs
                            status = f"error:{type(exc).__name__}"
                            error_message = str(exc)

                        runtime_sec = time.perf_counter() - start_time
                        rows.append(
                            {
                                "run_id": run_id,
                                "algorithm": algorithm,
                                "d": nodes,
                                "density": density,
                                "n_samples": n_samples,
                                "ds": ds,
                                "dp": dp,
                                "runtime_sec": runtime_sec,
                                "graph_seed": graph_seed,
                                "noise_seed": noise_seed,
                                "status": status,
                                "error_message": error_message,
                            }
                        )

                        if len(rows) % config.checkpoint_interval == 0:
                            write_checkpoint(rows, raw_results_path)

    raw_results = pd.DataFrame(rows)
    raw_results.to_csv(raw_results_path, index=False)

    summary_results = summarize_results(raw_results)
    summary_results.to_csv(summary_results_path, index=False)
    plot_results(summary_results, output_dir, config.algorithms)
    return raw_results, summary_results


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = BenchmarkConfig.from_namespace(args)
    benchmark(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
