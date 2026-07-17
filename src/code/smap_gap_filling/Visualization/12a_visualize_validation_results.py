#!/usr/bin/env python3
"""
12a_visualize_validation_results.py

Visualize SMAP gap-filling validation/test result CSV files.

This script reads metric and prediction files from:
  - 05_gapfill_model_validation/
  - 06_selected_methods_test/

It creates:
  - combined metric CSV
  - RMSE/MAE/R2/Bias bar plots
  - observed-vs-predicted scatterplots
  - error histograms

Outputs:
  src/data/processed/smap_gap_filling/09_final_visualization/validation/

This script does not modify model outputs.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER CONTROLS
# ============================================================

TOP_N_METHODS_PER_PLOT = 15
MAX_SCATTER_ROWS_PER_METHOD = 6000
MAKE_PDF_TOO = True

METRICS_TO_PLOT = ["rmse", "mae", "bias", "r2"]

# Make labels shorter in figures.
MAX_LABEL_LEN = 55


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR.parent / "11_gapfilling_setting.py"


if SETTINGS_PATH.exists():
    spec = importlib.util.spec_from_file_location("gapfill_settings", SETTINGS_PATH)
    settings = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load settings file: {SETTINGS_PATH}")
    spec.loader.exec_module(settings)
    PROJECT_ROOT = settings.PROJECT_ROOT
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_DIR = PROJECT_ROOT / "src/data/processed/smap_gap_filling"

VALIDATION_DIR = BASE_DIR / "05_gapfill_model_validation"
TEST_DIR = BASE_DIR / "06_selected_methods_test"

OUT_DIR = BASE_DIR / "09_final_visualization/validation"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x.strip("_")[:120]


def short_label(x: str, max_len: int = MAX_LABEL_LEN) -> str:
    x = str(x)
    if len(x) <= max_len:
        return x
    return x[: max_len - 3] + "..."


def save_figure(fig: plt.Figure, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    if MAKE_PDF_TOO:
        fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def infer_family(path: Path) -> str:
    s = str(path).lower()
    if "/ml/" in s or path.name.lower().startswith("ml_"):
        return "ML"
    if "interpolation" in s or "interp" in s:
        return "Interpolation"
    if "kriging" in s:
        return "Interpolation"
    return "Unknown"


def infer_result_set(path: Path) -> str:
    s = str(path).lower()
    if "06_selected_methods_test" in s or "_test_" in s or "selected_test" in s:
        return "test_2025"
    return "validation"


def method_label_from_row(row: pd.Series) -> str:
    family = str(row.get("family", ""))

    if family == "ML":
        model = str(row.get("model", "")).strip()
        feature_group = str(row.get("feature_group", "")).strip()

        if feature_group and feature_group.lower() != "nan":
            return f"{model} | {feature_group}"
        return model

    method = str(row.get("method", "")).strip()
    if method and method.lower() != "nan":
        return method

    model = str(row.get("model", "")).strip()
    if model and model.lower() != "nan":
        return model

    return "unknown_method"


# ============================================================
# LOAD METRICS
# ============================================================

def candidate_metric_files() -> list[Path]:
    files = []

    if VALIDATION_DIR.exists():
        files.extend(VALIDATION_DIR.rglob("*metrics*.csv"))

    if TEST_DIR.exists():
        files.extend(TEST_DIR.rglob("*metrics*.csv"))

    # Avoid reading our own outputs if rerun.
    files = [
        p for p in files
        if "09_final_visualization" not in str(p)
        and p.is_file()
    ]

    return sorted(set(files))


def normalize_metrics(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping unreadable metric file: {path}\n  {e}")
        return None

    if df.empty:
        return None

    df = df.loc[:, ~df.columns.duplicated()].copy()

    family = infer_family(path)
    result_set = infer_result_set(path)

    df["family"] = family
    df["result_set"] = result_set
    df["source_metric_file"] = str(path)

    if "method" not in df.columns:
        df["method"] = np.nan
    if "model" not in df.columns:
        df["model"] = np.nan
    if "feature_group" not in df.columns:
        df["feature_group"] = np.nan
    if "holdout_mode" not in df.columns:
        df["holdout_mode"] = "unknown_holdout"
    if "split" not in df.columns:
        df["split"] = result_set

    for c in ["rmse", "mae", "bias", "r2", "n"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["method_label"] = df.apply(method_label_from_row, axis=1)

    keep = [
        "result_set",
        "split",
        "holdout_mode",
        "family",
        "method",
        "model",
        "feature_group",
        "method_label",
        "rmse",
        "mae",
        "bias",
        "r2",
        "n",
        "source_metric_file",
    ]

    return df[keep].copy()


def load_all_metrics() -> pd.DataFrame:
    files = candidate_metric_files()

    print("\nMetric files found:")
    for p in files:
        print(f"  - {p}")

    parts = []
    for p in files:
        x = normalize_metrics(p)
        if x is not None and not x.empty:
            parts.append(x)

    if not parts:
        raise FileNotFoundError(
            f"No usable metric CSV files found under:\n"
            f"  {VALIDATION_DIR}\n"
            f"  {TEST_DIR}"
        )

    out = pd.concat(parts, ignore_index=True)
    out = out.dropna(subset=["method_label"], how="all").copy()

    combined_path = OUT_DIR / "combined_validation_and_test_metrics.csv"
    out.to_csv(combined_path, index=False)

    print(f"\nSaved combined metrics:\n  {combined_path}")

    return out


# ============================================================
# METRIC BAR PLOTS
# ============================================================

def sort_for_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    x = df.copy()

    if metric == "r2":
        x = x.sort_values(metric, ascending=False)
    elif metric == "bias":
        x["_abs_bias"] = x[metric].abs()
        x = x.sort_values("_abs_bias", ascending=True)
    else:
        x = x.sort_values(metric, ascending=True)

    return x


def plot_metric_bars(metrics: pd.DataFrame) -> None:
    for result_set in sorted(metrics["result_set"].dropna().unique()):
        for holdout in sorted(metrics["holdout_mode"].dropna().unique()):
            sub = metrics[
                (metrics["result_set"] == result_set)
                & (metrics["holdout_mode"] == holdout)
            ].copy()

            if sub.empty:
                continue

            for metric in METRICS_TO_PLOT:
                if metric not in sub.columns:
                    continue

                plot_df = sub[np.isfinite(sub[metric])].copy()
                if plot_df.empty:
                    continue

                plot_df = sort_for_metric(plot_df, metric).head(TOP_N_METHODS_PER_PLOT)
                plot_df = plot_df.iloc[::-1].copy()

                labels = [
                    short_label(f"{r.family}: {r.method_label}")
                    for r in plot_df.itertuples()
                ]

                fig_height = max(5.5, 0.36 * len(plot_df) + 1.5)
                fig, ax = plt.subplots(figsize=(11, fig_height))

                y = np.arange(len(plot_df))
                ax.barh(y, plot_df[metric].values)

                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=8.5)
                ax.set_xlabel(metric.upper())
                ax.set_title(
                    f"{metric.upper()} by method\n"
                    f"{result_set} | {holdout}"
                )

                if metric == "bias":
                    ax.axvline(0, linewidth=1)

                ax.grid(axis="x", alpha=0.25)

                for yi, val in zip(y, plot_df[metric].values):
                    if np.isfinite(val):
                        ax.text(
                            val,
                            yi,
                            f" {val:.4f}",
                            va="center",
                            fontsize=8,
                        )

                out = FIG_DIR / f"bar_{safe_name(result_set)}_{safe_name(holdout)}_{metric}.png"
                save_figure(fig, out)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def candidate_prediction_files() -> list[Path]:
    files = []

    if VALIDATION_DIR.exists():
        files.extend(VALIDATION_DIR.rglob("*prediction*.csv"))

    if TEST_DIR.exists():
        files.extend(TEST_DIR.rglob("*prediction*.csv"))

    files = [
        p for p in files
        if "09_final_visualization" not in str(p)
        and p.is_file()
    ]

    return sorted(set(files))


def normalize_predictions(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping unreadable prediction file: {path}\n  {e}")
        return None

    if df.empty:
        return None

    df = df.loc[:, ~df.columns.duplicated()].copy()

    if "observed" not in df.columns or "prediction" not in df.columns:
        return None

    family = infer_family(path)
    result_set = infer_result_set(path)

    df["family"] = family
    df["result_set"] = result_set
    df["source_prediction_file"] = str(path)

    if "method" not in df.columns:
        df["method"] = np.nan
    if "model" not in df.columns:
        df["model"] = np.nan
    if "feature_group" not in df.columns:
        df["feature_group"] = np.nan
    if "holdout_mode" not in df.columns:
        df["holdout_mode"] = "unknown_holdout"

    df["observed"] = pd.to_numeric(df["observed"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["error"] = df["prediction"] - df["observed"]
    df["method_label"] = df.apply(method_label_from_row, axis=1)

    keep = [
        "result_set",
        "holdout_mode",
        "family",
        "method",
        "model",
        "feature_group",
        "method_label",
        "observed",
        "prediction",
        "error",
        "source_prediction_file",
    ]

    return df[keep].dropna(subset=["observed", "prediction"]).copy()


def load_all_predictions() -> pd.DataFrame:
    files = candidate_prediction_files()

    print("\nPrediction files found:")
    for p in files:
        print(f"  - {p}")

    parts = []
    for p in files:
        x = normalize_predictions(p)
        if x is not None and not x.empty:
            parts.append(x)

    if not parts:
        print("\nNo usable prediction files found for scatter/error plots.")
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)

    combined_path = OUT_DIR / "combined_validation_and_test_predictions_sample.csv"
    out.to_csv(combined_path, index=False)

    print(f"\nSaved combined predictions sample:\n  {combined_path}")

    return out


# ============================================================
# SCATTER AND ERROR PLOTS
# ============================================================

def selected_methods_for_scatter(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for result_set in sorted(metrics["result_set"].dropna().unique()):
        for holdout in sorted(metrics["holdout_mode"].dropna().unique()):
            sub = metrics[
                (metrics["result_set"] == result_set)
                & (metrics["holdout_mode"] == holdout)
                & np.isfinite(metrics["rmse"])
            ].copy()

            if sub.empty:
                continue

            best = sub.sort_values("rmse").head(4)
            rows.append(best)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def plot_scatter_and_error(metrics: pd.DataFrame, preds: pd.DataFrame) -> None:
    if preds.empty:
        return

    chosen = selected_methods_for_scatter(metrics)
    if chosen.empty:
        return

    chosen_keys = set(
        zip(
            chosen["result_set"],
            chosen["holdout_mode"],
            chosen["family"],
            chosen["method_label"],
        )
    )

    for result_set, holdout, family, method_label in sorted(chosen_keys):
        sub = preds[
            (preds["result_set"] == result_set)
            & (preds["holdout_mode"] == holdout)
            & (preds["family"] == family)
            & (preds["method_label"] == method_label)
        ].copy()

        if sub.empty:
            continue

        if len(sub) > MAX_SCATTER_ROWS_PER_METHOD:
            sub = sub.sample(
                n=MAX_SCATTER_ROWS_PER_METHOD,
                random_state=42,
            )

        # Scatter plot
        fig, ax = plt.subplots(figsize=(7.5, 7))

        ax.scatter(
            sub["observed"],
            sub["prediction"],
            s=9,
            alpha=0.35,
            linewidths=0,
        )

        lo = np.nanmin([sub["observed"].min(), sub["prediction"].min()])
        hi = np.nanmax([sub["observed"].max(), sub["prediction"].max()])

        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [lo, hi], linewidth=1.2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        rmse = np.sqrt(np.mean((sub["prediction"] - sub["observed"]) ** 2))
        bias = np.mean(sub["prediction"] - sub["observed"])

        ax.set_xlabel("Observed SMAP soil moisture")
        ax.set_ylabel("Predicted soil moisture")
        ax.set_title(
            f"Observed vs predicted\n"
            f"{result_set} | {holdout}\n"
            f"{family}: {short_label(method_label, 70)}\n"
            f"RMSE={rmse:.4f}, Bias={bias:.4f}, n={len(sub):,}"
        )
        ax.grid(alpha=0.25)

        out = FIG_DIR / (
            f"scatter_{safe_name(result_set)}_{safe_name(holdout)}_"
            f"{safe_name(family)}_{safe_name(method_label)}.png"
        )
        save_figure(fig, out)

        # Error histogram
        fig, ax = plt.subplots(figsize=(8, 5.5))

        ax.hist(sub["error"].dropna(), bins=50, alpha=0.85)
        ax.axvline(0, linewidth=1.2)
        ax.axvline(bias, linestyle="--", linewidth=1.2)

        ax.set_xlabel("Prediction error = predicted - observed")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Prediction error distribution\n"
            f"{result_set} | {holdout}\n"
            f"{family}: {short_label(method_label, 70)}"
        )
        ax.grid(axis="y", alpha=0.25)

        out = FIG_DIR / (
            f"error_hist_{safe_name(result_set)}_{safe_name(holdout)}_"
            f"{safe_name(family)}_{safe_name(method_label)}.png"
        )
        save_figure(fig, out)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("12a: Visualize validation/test results")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Validation dir: {VALIDATION_DIR}")
    print(f"Test dir:       {TEST_DIR}")
    print(f"Output dir:     {OUT_DIR}")
    print("=" * 80)

    metrics = load_all_metrics()

    print("\nBest rows by RMSE:")
    cols = ["result_set", "holdout_mode", "family", "method_label", "rmse", "mae", "bias", "r2", "n"]
    print(
        metrics[cols]
        .dropna(subset=["rmse"])
        .sort_values(["result_set", "holdout_mode", "rmse"])
        .groupby(["result_set", "holdout_mode"])
        .head(8)
        .to_string(index=False)
    )

    plot_metric_bars(metrics)

    preds = load_all_predictions()
    plot_scatter_and_error(metrics, preds)

    print("\nSaved figures to:")
    print(f"  {FIG_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()