#!/usr/bin/env python3
"""
10c_compare_validation_results.py

Purpose
-------
Compare validation results from:

  10a_ML_validation.py
  10b_interpolation_validation.R

This script does NOT train models, test models, or fill SMAP gaps.
It only reads validation metrics, ranks methods, and writes clean summary tables.

Inputs
------
src/data/processed/smap_gap_filling/05_gapfill_model_validation/ml/ml_validation_metrics.csv
src/data/processed/smap_gap_filling/05_gapfill_model_validation/interpolation/interpolation_validation_metrics.csv

Outputs
-------
src/data/processed/smap_gap_filling/05_gapfill_model_validation/comparison/
    combined_validation_metrics.csv
    best_methods_by_holdout.csv
    recommended_base_models.csv
    summary_report.txt
    figures/
        combined_rmse_random_cell.pdf
        combined_rmse_spatial_block.pdf
"""

from __future__ import annotations

from pathlib import Path
import textwrap
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path("/home/armaghan/projects/SM_forecasting")

VALIDATION_DIR = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling/05_gapfill_model_validation"
)

ML_METRICS_PATH = VALIDATION_DIR / "ml/ml_validation_metrics.csv"
INTERP_METRICS_PATH = (
    VALIDATION_DIR / "interpolation/interpolation_validation_metrics.csv"
)

OUT_DIR = VALIDATION_DIR / "comparison"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Use spatial_block as the main decision criterion because it is harder
# and closer to realistic clustered missingness.
PRIMARY_HOLDOUT = "spatial_block"

# FFNN was unstable/weak in validation, so exclude it from first recommended set.
EXCLUDE_MODELS_FROM_RECOMMENDATION = {"ffnn_mlp", "baseline", "baseline_train_mean"}

# How many ML models to keep for script 11 candidate predictions.
N_ML_MODELS_TO_RECOMMEND = 3

# Keep the two interpolation methods: centroid OK + nearest neighbor baseline.
N_INTERP_METHODS_TO_RECOMMEND = 2


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")


def read_ml_metrics(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    required = {
        "split",
        "holdout_mode",
        "feature_group",
        "model",
        "n_features",
        "features",
        "rmse",
        "mae",
        "bias",
        "r2",
        "n",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"ML metrics missing required columns: {missing}")

    out = df.copy()
    out["method_family"] = "ML"
    out["method"] = out["model"].astype(str)
    out["method_name"] = (
        out["model"].astype(str) + " | " + out["feature_group"].astype(str)
    )
    out["source_file"] = str(path)
    return out


def read_interpolation_metrics(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)

    required = {
        "split",
        "holdout_mode",
        "method",
        "rmse",
        "mae",
        "bias",
        "r2",
        "n",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Interpolation metrics missing required columns: {missing}")

    out = df.copy()
    out["method_family"] = "Interpolation"
    out["model"] = out["method"].astype(str)
    out["feature_group"] = ""
    out["n_features"] = out.get("n_features", pd.Series([pd.NA] * len(out)))
    out["features"] = out.get("features", pd.Series([""] * len(out)))
    out["method_name"] = out["method"].astype(str)
    out["source_file"] = str(path)

    return out


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "split",
        "holdout_mode",
        "method_family",
        "method",
        "method_name",
        "feature_group",
        "model",
        "n_features",
        "features",
        "rmse",
        "mae",
        "bias",
        "r2",
        "n",
        "source_file",
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    out = df[cols].copy()

    numeric_cols = ["n_features", "rmse", "mae", "bias", "r2", "n"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.sort_values(["holdout_mode", "rmse", "mae"]).reset_index(drop=True)
    return out


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_within_holdout"] = (
        out.groupby("holdout_mode")["rmse"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )
    out["rank_within_family_holdout"] = (
        out.groupby(["holdout_mode", "method_family"])["rmse"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )
    return out


def make_best_by_holdout(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for holdout, sub in df.groupby("holdout_mode"):
        sub = sub.sort_values(["rmse", "mae"]).copy()
        rows.append(sub.head(1))

    return pd.concat(rows, ignore_index=True)


def make_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    primary = df[df["holdout_mode"].eq(PRIMARY_HOLDOUT)].copy()
    if primary.empty:
        raise ValueError(f"No rows found for PRIMARY_HOLDOUT={PRIMARY_HOLDOUT}")

    recommendations = []

    # 1. Interpolation recommendations
    interp = primary[primary["method_family"].eq("Interpolation")].copy()
    interp = interp.sort_values(["rmse", "mae"])

    if not interp.empty:
        keep = interp.head(N_INTERP_METHODS_TO_RECOMMEND).copy()
        keep["recommendation_role"] = [
            "primary_spatial_gap_filler" if i == 0 else "interpolation_baseline"
            for i in range(len(keep))
        ]
        recommendations.append(keep)

    # 2. ML recommendations: choose best feature group per model first
    ml = primary[primary["method_family"].eq("ML")].copy()
    ml = ml[~ml["model"].isin(EXCLUDE_MODELS_FROM_RECOMMENDATION)]
    ml = ml.sort_values(["model", "rmse", "mae"])
    ml_best_per_model = ml.groupby("model", as_index=False).head(1)
    ml_best_per_model = ml_best_per_model.sort_values(["rmse", "mae"])

    if not ml_best_per_model.empty:
        keep = ml_best_per_model.head(N_ML_MODELS_TO_RECOMMEND).copy()
        keep["recommendation_role"] = "ml_auxiliary_baseline"
        recommendations.append(keep)

    if not recommendations:
        return pd.DataFrame()

    out = pd.concat(recommendations, ignore_index=True)
    out = out.sort_values(
        ["recommendation_role", "rmse", "mae"]
    ).reset_index(drop=True)

    out["use_in_script_11"] = True
    return out


def write_report(
    combined: pd.DataFrame,
    best: pd.DataFrame,
    recs: pd.DataFrame,
    path: Path,
) -> None:
    lines = []

    lines.append("SMAP Gap-Filling Validation Comparison")
    lines.append("=" * 45)
    lines.append("")
    lines.append(f"Primary decision holdout: {PRIMARY_HOLDOUT}")
    lines.append("")
    lines.append("Interpretation rule:")
    lines.append(
        "  Lower RMSE/MAE is better. Spatial-block validation should be treated "
        "as more important than random-cell validation because it is harder and "
        "more realistic for clustered missingness."
    )
    lines.append("")

    lines.append("Best method by holdout mode:")
    lines.append("-" * 45)
    if not best.empty:
        lines.append(
            best[
                [
                    "holdout_mode",
                    "method_family",
                    "method_name",
                    "rmse",
                    "mae",
                    "bias",
                    "r2",
                    "n",
                ]
            ].to_string(index=False)
        )
    lines.append("")

    lines.append("Recommended base models for next stage:")
    lines.append("-" * 45)
    if not recs.empty:
        lines.append(
            recs[
                [
                    "recommendation_role",
                    "method_family",
                    "method_name",
                    "rmse",
                    "mae",
                    "bias",
                    "r2",
                    "n",
                ]
            ].to_string(index=False)
        )
    else:
        lines.append("No recommendations generated.")
    lines.append("")

    lines.append("Top 15 methods under spatial-block validation:")
    lines.append("-" * 45)
    spatial = combined[combined["holdout_mode"].eq("spatial_block")]
    if not spatial.empty:
        lines.append(
            spatial.sort_values(["rmse", "mae"])
            [
                [
                    "method_family",
                    "method_name",
                    "rmse",
                    "mae",
                    "bias",
                    "r2",
                    "n",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )
    lines.append("")

    path.write_text("\n".join(lines))


def plot_rmse(df: pd.DataFrame, holdout_mode: str, path: Path, top_n: int = 20) -> None:
    sub = df[df["holdout_mode"].eq(holdout_mode)].copy()
    sub = sub.sort_values(["rmse", "mae"]).head(top_n)
    if sub.empty:
        return

    labels = sub["method_name"].astype(str).tolist()
    rmse = sub["rmse"].tolist()

    height = max(5, 0.35 * len(sub))
    plt.figure(figsize=(10, height))
    plt.barh(labels[::-1], rmse[::-1])
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.title(f"Top {len(sub)} methods by RMSE: {holdout_mode}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("Comparing SMAP gap-filling validation results")
    print("=" * 70)
    print(f"ML metrics:            {ML_METRICS_PATH}")
    print(f"Interpolation metrics: {INTERP_METRICS_PATH}")
    print(f"Output folder:         {OUT_DIR}")
    print("=" * 70)

    ml = read_ml_metrics(ML_METRICS_PATH)
    interp = read_interpolation_metrics(INTERP_METRICS_PATH)

    combined = pd.concat([ml, interp], ignore_index=True)
    combined = standardize_columns(combined)
    combined = add_ranks(combined)

    best = make_best_by_holdout(combined)
    recs = make_recommendations(combined)

    combined_path = OUT_DIR / "combined_validation_metrics.csv"
    best_path = OUT_DIR / "best_methods_by_holdout.csv"
    recs_path = OUT_DIR / "recommended_base_models.csv"
    report_path = OUT_DIR / "summary_report.txt"

    combined.to_csv(combined_path, index=False)
    best.to_csv(best_path, index=False)
    recs.to_csv(recs_path, index=False)

    write_report(combined, best, recs, report_path)

    for holdout in sorted(combined["holdout_mode"].dropna().unique()):
        plot_rmse(
            combined,
            holdout_mode=holdout,
            path=FIG_DIR / f"combined_rmse_{holdout}.pdf",
            top_n=20,
        )

    print("\nSaved:")
    print(f"  {combined_path}")
    print(f"  {best_path}")
    print(f"  {recs_path}")
    print(f"  {report_path}")
    print(f"  {FIG_DIR}")

    print("\nRecommended base models:")
    if not recs.empty:
        print(
            recs[
                [
                    "recommendation_role",
                    "method_family",
                    "method_name",
                    "rmse",
                    "mae",
                    "bias",
                    "r2",
                    "n",
                ]
            ].to_string(index=False)
        )
    else:
        print("No recommendations generated.")

    print("\nDone.")


if __name__ == "__main__":
    main()