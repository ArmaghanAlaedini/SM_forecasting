#!/usr/bin/env python3
"""Visualize aligned 2024 validation and independent 2025 test results.

This script deliberately reads exact files from the corrected workflow. It does
not use ``rglob('*metrics*.csv')`` because that would mix pooled summaries,
daily GI metrics, method-specific metrics, and common-support metrics.

Primary comparisons
-------------------
* 2024: ``comparison/combined_validation_metrics.csv`` on ``common_support``.
* 2025: ``stacking/stacking_selected_test_metrics.csv`` on the common six-model
  support, including the frozen ridge stack.

Outputs
-------
``09_final_visualization/05_model_evaluation``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization_common import (
    VISUALIZATION_ROOT,
    cfg,
    model_family,
    pretty_method,
    safe_name,
    save_figure,
)


OUT_DIR = VISUALIZATION_ROOT / "05_model_evaluation"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_METRICS_PATH = cfg.COMPARISON_DIR / "combined_validation_metrics.csv"
VALIDATION_COVERAGE_PATH = cfg.COMPARISON_DIR / "validation_prediction_coverage.csv"
VALIDATION_ML_PRED_PATH = cfg.ML_VALIDATION_DIR / "ml_validation_predictions.csv"
VALIDATION_GI_PRED_PATH = cfg.INTERP_VALIDATION_DIR / "interpolation_validation_predictions.csv"
META_COVERAGE_PATH = cfg.STACKING_DIR / "meta_training_coverage.csv"
COEFFICIENT_PATH = cfg.STACKING_DIR / "meta_model_coefficients.csv"

TEST_METRICS_PATH = cfg.STACKING_TEST_DIR / "stacking_selected_test_metrics.csv"
TEST_PRED_PATH = cfg.STACKING_TEST_DIR / "stacking_selected_test_predictions.csv"
TEST_COVERAGE_PATH = cfg.STACKING_TEST_DIR / "stacking_selected_test_coverage.csv"

COMBINED_METRICS_PATH = OUT_DIR / "model_evaluation_metrics.csv"
COMBINED_PREDICTION_SAMPLE_PATH = OUT_DIR / "model_evaluation_prediction_sample.csv"

METRICS = ["rmse", "mae", "bias", "r2"]
PRIMARY_SUPPORT = {
    "validation_2024": "common_support",
    "independent_test_2025": "common_six_model_support",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-scatter-rows", type=int, default=6000)
    parser.add_argument("--top-methods", type=int, default=8)
    return parser.parse_args()


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required model-result file is missing: {path}")


def load_metrics() -> pd.DataFrame:
    require(VALIDATION_METRICS_PATH)
    require(TEST_METRICS_PATH)

    validation = pd.read_csv(VALIDATION_METRICS_PATH)
    validation["stage"] = "validation_2024"
    if "method_family" not in validation.columns:
        validation["method_family"] = validation["method"].map(model_family)

    test = pd.read_csv(TEST_METRICS_PATH)
    test["stage"] = "independent_test_2025"
    test["method_family"] = test["method"].map(model_family)
    if "n_targets" not in test.columns:
        test["n_targets"] = test.get("n", np.nan)

    columns = [
        "stage", "split", "holdout_mode", "support", "method_family", "method",
        "n_targets", "rmse", "mae", "bias", "r2", "n", "coverage",
    ]
    for frame in [validation, test]:
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        for column in ["rmse", "mae", "bias", "r2", "n", "coverage", "n_targets"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    metrics = pd.concat([validation[columns], test[columns]], ignore_index=True)
    metrics["method_label"] = metrics["method"].map(pretty_method)
    metrics.to_csv(COMBINED_METRICS_PATH, index=False)
    return metrics


def primary_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    keep = []
    for stage, support in PRIMARY_SUPPORT.items():
        keep.append(metrics[metrics["stage"].eq(stage) & metrics["support"].eq(support)])
    return pd.concat(keep, ignore_index=True)


def metric_sort(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric == "r2":
        return frame.sort_values(metric, ascending=False)
    if metric == "bias":
        return frame.assign(abs_bias=frame[metric].abs()).sort_values("abs_bias")
    return frame.sort_values(metric)


def plot_metric_bars(metrics: pd.DataFrame, top_n: int) -> None:
    primary = primary_metrics(metrics)
    for (stage, holdout), sub in primary.groupby(["stage", "holdout_mode"], sort=True):
        for metric in METRICS:
            plot_df = sub[np.isfinite(sub[metric])].copy()
            if plot_df.empty:
                continue
            plot_df = metric_sort(plot_df, metric).head(top_n).iloc[::-1]
            labels = [
                f"{pretty_method(method)} ({family})"
                for method, family in zip(plot_df["method"], plot_df["method_family"])
            ]
            fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.48 * len(plot_df) + 1.5)))
            positions = np.arange(len(plot_df))
            ax.barh(positions, plot_df[metric])
            ax.set_yticks(positions)
            ax.set_yticklabels(labels)
            ax.set_xlabel(metric.upper())
            ax.set_title(f"{stage.replace('_', ' ').title()} — {holdout.replace('_', ' ')}")
            if metric == "bias":
                ax.axvline(0, color="0.25", linewidth=1)
            ax.grid(axis="x", alpha=0.25)
            for position, value, n_rows in zip(positions, plot_df[metric], plot_df["n"]):
                ax.text(value, position, f" {value:.4f}  (n={int(n_rows):,})", va="center", fontsize=8)
            fig.tight_layout()
            save_figure(
                fig,
                FIG_DIR / f"metric_{safe_name(stage)}_{safe_name(holdout)}_{metric}",
            )


def load_validation_predictions() -> pd.DataFrame:
    require(VALIDATION_ML_PRED_PATH)
    require(VALIDATION_GI_PRED_PATH)
    ml = pd.read_csv(VALIDATION_ML_PRED_PATH, low_memory=False)
    ml = ml[
        ml["feature_group"].eq(cfg.FINAL_ML_FEATURE_GROUP)
        & ml["model"].isin(cfg.CANDIDATE_ML_MODELS)
    ].copy()
    ml["method"] = ml["model"]
    gi = pd.read_csv(VALIDATION_GI_PRED_PATH, low_memory=False)
    gi = gi[gi["method"].isin(cfg.SELECTED_INTERPOLATION_METHODS)].copy()

    columns = ["holdout_mode", "date", "pass", cfg.KEY, "observed", "prediction", "method"]
    out = pd.concat([ml[columns], gi[columns]], ignore_index=True)
    out["stage"] = "validation_2024"
    out["support"] = "common_manifest_targets"
    return out


def load_test_predictions() -> pd.DataFrame:
    require(TEST_PRED_PATH)
    out = pd.read_csv(TEST_PRED_PATH, low_memory=False)
    out["stage"] = "independent_test_2025"
    out["support"] = "common_six_model_support"
    return out


def load_predictions(max_rows_per_method: int) -> pd.DataFrame:
    predictions = pd.concat(
        [load_validation_predictions(), load_test_predictions()], ignore_index=True
    )
    predictions["observed"] = pd.to_numeric(predictions["observed"], errors="coerce")
    predictions["prediction"] = pd.to_numeric(predictions["prediction"], errors="coerce")
    predictions = predictions.dropna(subset=["observed", "prediction"]).copy()
    predictions["error"] = predictions["prediction"] - predictions["observed"]
    predictions["method_family"] = predictions["method"].map(model_family)
    predictions["method_label"] = predictions["method"].map(pretty_method)

    sampled = []
    for _, group in predictions.groupby(["stage", "holdout_mode", "method"], sort=False):
        if len(group) > max_rows_per_method:
            group = group.sample(n=max_rows_per_method, random_state=cfg.RANDOM_SEED)
        sampled.append(group)
    output = pd.concat(sampled, ignore_index=True)
    output.to_csv(COMBINED_PREDICTION_SAMPLE_PATH, index=False)
    return output


def selected_scatter_methods(metrics: pd.DataFrame) -> set[tuple[str, str, str]]:
    selected: set[tuple[str, str, str]] = set()
    primary = primary_metrics(metrics)
    for (stage, holdout), sub in primary.groupby(["stage", "holdout_mode"], sort=True):
        best = sub.sort_values("rmse").head(4)
        for method in best["method"]:
            selected.add((stage, holdout, method))
    return selected


def plot_scatter_and_errors(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    for stage, holdout, method in sorted(selected_scatter_methods(metrics)):
        sub = predictions[
            predictions["stage"].eq(stage)
            & predictions["holdout_mode"].eq(holdout)
            & predictions["method"].eq(method)
        ]
        if sub.empty:
            continue

        rmse = float(np.sqrt(np.mean(sub["error"] ** 2)))
        bias = float(sub["error"].mean())
        lo = float(min(sub["observed"].min(), sub["prediction"].min()))
        hi = float(max(sub["observed"].max(), sub["prediction"].max()))

        fig, ax = plt.subplots(figsize=(7.3, 7.0))
        ax.scatter(sub["observed"], sub["prediction"], s=10, alpha=0.35, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="0.20", linewidth=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed SMAP SM")
        ax.set_ylabel("Predicted SM")
        ax.set_title(
            f"{pretty_method(method)}\n{stage.replace('_', ' ')} | {holdout}\n"
            f"RMSE={rmse:.4f}, bias={bias:.4f}, n={len(sub):,}"
        )
        ax.grid(alpha=0.25)
        fig.tight_layout()
        stem = f"scatter_{stage}_{holdout}_{method}"
        save_figure(fig, FIG_DIR / safe_name(stem))

        fig, ax = plt.subplots(figsize=(8.0, 5.4))
        ax.hist(sub["error"], bins=50, alpha=0.85)
        ax.axvline(0, color="0.20", linewidth=1.2)
        ax.axvline(bias, color="0.35", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Prediction error (predicted - observed)")
        ax.set_ylabel("Count")
        ax.set_title(f"{pretty_method(method)} error distribution\n{stage} | {holdout}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, FIG_DIR / safe_name(f"error_{stage}_{holdout}_{method}"))


def plot_validation_coverage() -> None:
    if not VALIDATION_COVERAGE_PATH.exists():
        print(f"[warning] Coverage file missing: {VALIDATION_COVERAGE_PATH}")
        return
    coverage = pd.read_csv(VALIDATION_COVERAGE_PATH)
    for holdout, sub in coverage.groupby("holdout_mode"):
        sub = sub.sort_values("coverage")
        fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.42 * len(sub))))
        labels = [pretty_method(method) for method in sub["method"]]
        ax.barh(labels, sub["coverage"] * 100)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Prediction coverage (%)")
        ax.set_title(f"2024 validation prediction coverage — {holdout}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, FIG_DIR / f"coverage_validation_{safe_name(holdout)}")


def plot_stacking_coverage() -> None:
    paths = [
        (META_COVERAGE_PATH, "2024 meta-training coverage", "coverage_meta_training"),
        (TEST_COVERAGE_PATH, "2025 stacking-test coverage", "coverage_stacking_test"),
    ]
    for path, title, filename in paths:
        if not path.exists():
            print(f"[warning] Coverage file missing: {path}")
            continue
        coverage = pd.read_csv(path).sort_values("coverage")
        labels = [pretty_method(str(feature).removeprefix("pred_")) for feature in coverage["feature"]]
        fig, ax = plt.subplots(figsize=(9, max(4.2, 0.48 * len(coverage))))
        ax.barh(labels, coverage["coverage"] * 100)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Available rows (%)")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, FIG_DIR / filename)


def plot_coefficients() -> None:
    if not COEFFICIENT_PATH.exists():
        print(f"[warning] Coefficient file missing: {COEFFICIENT_PATH}")
        return
    coefficients = pd.read_csv(COEFFICIENT_PATH).sort_values("coefficient")
    labels = [pretty_method(str(feature).removeprefix("pred_")) for feature in coefficients["feature"]]
    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.48 * len(coefficients))))
    ax.barh(labels, coefficients["coefficient"])
    ax.axvline(0, color="0.20", linewidth=1)
    ax.set_xlabel("Standardized ridge coefficient")
    ax.set_title("Final 2024 ridge meta-model coefficients")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / "ridge_meta_model_coefficients")


def main() -> None:
    args = parse_args()
    print("12a: Visualize aligned validation and independent-test results")
    print(f"Validation metrics: {VALIDATION_METRICS_PATH}")
    print(f"Test metrics:       {TEST_METRICS_PATH}")
    print(f"Project seed:       {cfg.RANDOM_SEED}")
    print(f"Output:             {OUT_DIR}")

    metrics = load_metrics()
    primary = primary_metrics(metrics)
    print("\nPrimary comparable results:")
    print(
        primary.sort_values(["stage", "holdout_mode", "rmse"])[
            ["stage", "holdout_mode", "method", "rmse", "mae", "bias", "r2", "n"]
        ].to_string(index=False)
    )

    plot_metric_bars(metrics, args.top_methods)
    predictions = load_predictions(args.max_scatter_rows)
    plot_scatter_and_errors(metrics, predictions)
    plot_validation_coverage()
    plot_stacking_coverage()
    plot_coefficients()
    print(f"\nSaved figures to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
