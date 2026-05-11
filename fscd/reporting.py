from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARKER_STYLES = ("o", "s", "^", "D", "P", "X", "v", "<", ">")


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
            for algorithm_index, algorithm in enumerate(algorithms):
                algorithm_rows = group.loc[group["algorithm"] == algorithm].sort_values("n_samples")
                if algorithm_rows.empty:
                    continue

                x_values = algorithm_rows["n_samples"].to_numpy(dtype=float)
                mean_values = algorithm_rows[f"{metric_prefix}_mean"].to_numpy(dtype=float)
                low_values = algorithm_rows[f"{metric_prefix}_ci95_low"].to_numpy(dtype=float)
                high_values = algorithm_rows[f"{metric_prefix}_ci95_high"].to_numpy(dtype=float)
                marker_style = MARKER_STYLES[algorithm_index % len(MARKER_STYLES)]

                line = axis.plot(x_values, mean_values, label=algorithm.upper(), alpha=0.8)[0]
                axis.scatter(
                    x_values,
                    mean_values,
                    marker=marker_style,
                    s=48,
                    color=line.get_color(),
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                axis.fill_between(x_values, low_values, high_values, alpha=0.15)

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


def summarize_topk_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    successful = raw_results.loc[raw_results["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "d",
                "density",
                "n_samples",
                "epsilon",
                "delta",
                "k",
                "n_valid",
                "ds_mean",
                "ds_std",
                "ds_ci95_low",
                "ds_ci95_high",
                "dp_mean",
                "dp_std",
                "dp_ci95_low",
                "dp_ci95_high",
                "bic_score_mean",
                "bic_score_std",
                "bic_score_ci95_low",
                "bic_score_ci95_high",
                "found_true_by_k_mean",
                "found_true_by_k_std",
                "found_true_by_k_ci95_low",
                "found_true_by_k_ci95_high",
                "is_true_at_k_mean",
                "is_true_at_k_std",
                "is_true_at_k_ci95_low",
                "is_true_at_k_ci95_high",
            ]
        )

    summary_rows: list[dict[str, float | int]] = []
    group_columns = ["d", "density", "n_samples", "epsilon", "delta", "k"]
    metric_names = ("ds", "dp", "bic_score", "found_true_by_k", "is_true_at_k")
    for keys, group in successful.groupby(group_columns, sort=True):
        row: dict[str, float | int] = {
            "d": int(keys[0]),
            "density": float(keys[1]),
            "n_samples": int(keys[2]),
            "epsilon": int(keys[3]),
            "delta": int(keys[4]),
            "k": int(keys[5]),
            "n_valid": int(group.shape[0]),
        }
        for metric_name in metric_names:
            stats = confidence_interval(group[metric_name])
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]
            row[f"{metric_name}_ci95_low"] = stats["ci95_low"]
            row[f"{metric_name}_ci95_high"] = stats["ci95_high"]
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values(group_columns).reset_index(drop=True)


def summarize_recovery_results(raw_results: pd.DataFrame, k_max: int) -> pd.DataFrame:
    successful = raw_results.loc[raw_results["status"] == "ok"].copy()
    if successful.empty:
        return pd.DataFrame(
            columns=[
                "d",
                "density",
                "n_samples",
                "epsilon",
                "delta",
                "n_runs",
                "n_success",
                "success_rate_within_kmax",
                "recover_k_mean_found_only",
                "recover_k_std_found_only",
                "recover_k_ci95_low_found_only",
                "recover_k_ci95_high_found_only",
            ]
        )

    run_level_rows: list[dict[str, float | int]] = []
    run_columns = ["d", "density", "n_samples", "epsilon", "delta", "run_id"]
    for keys, group in successful.groupby(run_columns, sort=True):
        recovered = group.loc[group["is_true_at_k"] == 1, "k"]
        recover_k = float(recovered.min()) if not recovered.empty else math.nan
        run_level_rows.append(
            {
                "d": int(keys[0]),
                "density": float(keys[1]),
                "n_samples": int(keys[2]),
                "epsilon": int(keys[3]),
                "delta": int(keys[4]),
                "run_id": int(keys[5]),
                "recover_k": recover_k,
                "found_within_kmax": int(not math.isnan(recover_k) and recover_k <= k_max),
            }
        )

    run_level = pd.DataFrame(run_level_rows)
    summary_rows: list[dict[str, float | int]] = []
    group_columns = ["d", "density", "n_samples", "epsilon", "delta"]
    for keys, group in run_level.groupby(group_columns, sort=True):
        found_only = group["recover_k"].dropna()
        recover_stats = confidence_interval(found_only)
        n_runs = int(group.shape[0])
        n_success = int(group["found_within_kmax"].sum())
        summary_rows.append(
            {
                "d": int(keys[0]),
                "density": float(keys[1]),
                "n_samples": int(keys[2]),
                "epsilon": int(keys[3]),
                "delta": int(keys[4]),
                "n_runs": n_runs,
                "n_success": n_success,
                "success_rate_within_kmax": float(n_success / n_runs) if n_runs else math.nan,
                "recover_k_mean_found_only": recover_stats["mean"],
                "recover_k_std_found_only": recover_stats["std"],
                "recover_k_ci95_low_found_only": recover_stats["ci95_low"],
                "recover_k_ci95_high_found_only": recover_stats["ci95_high"],
            }
        )

    return pd.DataFrame(summary_rows).sort_values(group_columns).reset_index(drop=True)


def plot_topk_distances(
    summary_results: pd.DataFrame,
    output_dir: Path,
    nodes: int,
    density: float,
    epsilon: int,
    delta: int,
    n_samples: int,
) -> Path | None:
    selected = summary_results.loc[
        (summary_results["d"] == nodes)
        & (summary_results["density"] == density)
        & (summary_results["epsilon"] == epsilon)
        & (summary_results["delta"] == delta)
        & (summary_results["n_samples"] == n_samples)
    ].sort_values("k")

    if selected.empty:
        return None

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for axis, (metric_name, title) in zip(axes, (("ds", "Skeleton Distance"), ("dp", "Permutation Distance"))):
        x_values = selected["k"].to_numpy(dtype=float)
        mean_values = selected[f"{metric_name}_mean"].to_numpy(dtype=float)
        low_values = selected[f"{metric_name}_ci95_low"].to_numpy(dtype=float)
        high_values = selected[f"{metric_name}_ci95_high"].to_numpy(dtype=float)
        line = axis.plot(x_values, mean_values, alpha=0.85)[0]
        axis.scatter(
            x_values,
            mean_values,
            marker="o",
            s=48,
            color=line.get_color(),
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
        axis.fill_between(x_values, low_values, high_values, alpha=0.15)
        axis.set_title(title)
        axis.set_ylabel("Distance")
        axis.grid(True, alpha=0.3)

    axes[-1].set_xlabel("K")
    figure.suptitle(
        f"BOSS Greedy Top-K Distances (d={nodes}, density={density}, epsilon={epsilon}, delta={delta}, n={n_samples})"
    )
    figure.tight_layout()

    density_label = str(density).replace(".", "p")
    plot_path = (
        plots_dir / f"topk_distances_d{nodes}_density{density_label}_eps{epsilon}_delta{delta}_n{n_samples}.png"
    )
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return plot_path
