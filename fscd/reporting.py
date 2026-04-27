from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def confidence_interval(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().astype(float)
    n_valid = int(clean.shape[0])
    if n_valid == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "n_valid": 0,
        }

    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n_valid > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(n_valid) if n_valid > 1 else 0.0
    ci95_low = max(0.0, mean - half_width)
    ci95_high = mean + half_width
    return {
        "mean": mean,
        "std": std,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "n_valid": n_valid,
    }


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    successful = raw_results.loc[raw_results["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "d",
                "density",
                "n_samples",
                "n_valid",
                "ds_mean",
                "ds_std",
                "ds_ci95_low",
                "ds_ci95_high",
                "dp_mean",
                "dp_std",
                "dp_ci95_low",
                "dp_ci95_high",
                "runtime_sec_mean",
                "runtime_sec_std",
            ]
        )

    summary_rows: list[dict[str, float | int | str]] = []
    group_columns = ["algorithm", "d", "density", "n_samples"]
    for keys, group in successful.groupby(group_columns, sort=True):
        ds_stats = confidence_interval(group["ds"])
        dp_stats = confidence_interval(group["dp"])
        runtime_stats = confidence_interval(group["runtime_sec"])
        algorithm, nodes, density, n_samples = keys
        summary_rows.append(
            {
                "algorithm": algorithm,
                "d": int(nodes),
                "density": float(density),
                "n_samples": int(n_samples),
                "n_valid": int(group.shape[0]),
                "ds_mean": ds_stats["mean"],
                "ds_std": ds_stats["std"],
                "ds_ci95_low": ds_stats["ci95_low"],
                "ds_ci95_high": ds_stats["ci95_high"],
                "dp_mean": dp_stats["mean"],
                "dp_std": dp_stats["std"],
                "dp_ci95_low": dp_stats["ci95_low"],
                "dp_ci95_high": dp_stats["ci95_high"],
                "runtime_sec_mean": runtime_stats["mean"],
                "runtime_sec_std": runtime_stats["std"],
            }
        )

    return pd.DataFrame(summary_rows).sort_values(group_columns).reset_index(drop=True)


def plot_results(summary_results: pd.DataFrame, output_dir: Path, algorithms: tuple[str, ...]) -> list[Path]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_paths: list[Path] = []
    if summary_results.empty:
        return plot_paths

    grouped = summary_results.groupby(["d", "density"], sort=True)
    for (nodes, density), group in grouped:
        figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        metrics = [
            ("ds", "Skeleton Distance"),
            ("dp", "Permutation Distance"),
        ]

        for axis, (metric_prefix, title) in zip(axes, metrics):
            for algorithm in algorithms:
                algorithm_rows = group.loc[group["algorithm"] == algorithm].sort_values("n_samples")
                if algorithm_rows.empty:
                    continue

                x_values = algorithm_rows["n_samples"].to_numpy(dtype=float)
                mean_values = algorithm_rows[f"{metric_prefix}_mean"].to_numpy(dtype=float)
                low_values = algorithm_rows[f"{metric_prefix}_ci95_low"].to_numpy(dtype=float)
                high_values = algorithm_rows[f"{metric_prefix}_ci95_high"].to_numpy(dtype=float)

                axis.plot(x_values, mean_values, marker="o", label=algorithm.upper())
                axis.fill_between(x_values, low_values, high_values, alpha=0.2)

            axis.set_title(title)
            axis.set_ylabel("Distance")
            axis.grid(True, alpha=0.3)
            axis.legend()

        axes[-1].set_xlabel("Sample Size")
        axes[-1].set_xscale("log")
        figure.suptitle(f"Finite-Sample Distances (d={nodes}, density={density})")
        figure.tight_layout()

        density_label = str(density).replace(".", "p")
        plot_path = plots_dir / f"distances_d{nodes}_density{density_label}.png"
        figure.savefig(plot_path, dpi=200)
        plt.close(figure)
        plot_paths.append(plot_path)

    return plot_paths
