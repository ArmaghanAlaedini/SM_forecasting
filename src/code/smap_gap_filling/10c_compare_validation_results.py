#!/usr/bin/env python3
"""Compare 2024 ML and GI predictions on common target support.

Unlike the old script, this version reads prediction-level files rather than
concatenating metrics computed with different aggregation rules.  It reports:

* method-specific pooled metrics using each method's finite predictions;
* common-support pooled metrics using only target keys predicted by every
  candidate method; and
* prediction coverage.

Recommendations are based on spatial-block RMSE on common support.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapfill_workflow_common import cfg, compute_metrics


ML_PATH = cfg.ML_VALIDATION_DIR / "ml_validation_predictions.csv"
GI_PATH = cfg.INTERP_VALIDATION_DIR / "interpolation_validation_predictions.csv"
OUT_DIR = cfg.COMPARISON_DIR
FIG_DIR = OUT_DIR / "figures"

COMBINED_PATH = OUT_DIR / "combined_validation_metrics.csv"
COVERAGE_PATH = OUT_DIR / "validation_prediction_coverage.csv"
RECOMMENDATION_PATH = OUT_DIR / "recommended_base_models.csv"
SUMMARY_PATH = OUT_DIR / "summary_report.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

KEY_COLUMNS = ["split", "holdout_mode", "date", "pass", cfg.KEY]


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_predictions() -> pd.DataFrame:
    require(ML_PATH)
    require(GI_PATH)

    ml = pd.read_csv(ML_PATH, low_memory=False)
    ml = ml[
        ml["feature_group"].eq(cfg.FINAL_ML_FEATURE_GROUP)
        & ml["model"].isin(cfg.CANDIDATE_ML_MODELS)
    ].copy()
    ml["method"] = ml["model"]
    ml["method_family"] = "ML"

    gi = pd.read_csv(GI_PATH, low_memory=False)
    gi = gi[gi["method"].isin(cfg.SELECTED_INTERPOLATION_METHODS)].copy()
    gi["method_family"] = "GI"

    common_columns = KEY_COLUMNS + [
        "observed",
        "prediction",
        "method",
        "method_family",
    ]
    for frame in [ml, gi]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame[cfg.KEY] = frame[cfg.KEY].astype(str)
        frame["observed"] = pd.to_numeric(frame["observed"], errors="coerce")
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")

    combined = pd.concat([ml[common_columns], gi[common_columns]], ignore_index=True)
    duplicated = combined.duplicated(KEY_COLUMNS + ["method"], keep=False)
    if duplicated.any():
        examples = combined.loc[duplicated, KEY_COLUMNS + ["method"]].head().to_dict("records")
        raise ValueError(f"Duplicate method/target predictions found: {examples}")
    return combined


def metric_rows(predictions: pd.DataFrame, support_name: str) -> list[dict]:
    rows: list[dict] = []
    for (holdout_mode, family, method), sub in predictions.groupby(
        ["holdout_mode", "method_family", "method"], sort=True
    ):
        row = {
            "split": "validation",
            "holdout_mode": holdout_mode,
            "support": support_name,
            "method_family": family,
            "method": method,
            "n_targets": len(sub),
        }
        row.update(compute_metrics(sub["observed"], sub["prediction"]))
        row["coverage"] = row["n"] / len(sub) if len(sub) else np.nan
        rows.append(row)
    return rows


def common_support_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    expected_methods = sorted(predictions["method"].unique())
    finite = predictions[np.isfinite(predictions["prediction"])].copy()
    counts = finite.groupby(KEY_COLUMNS)["method"].nunique()
    complete_keys = counts[counts.eq(len(expected_methods))].reset_index()[KEY_COLUMNS]
    common = predictions.merge(complete_keys, on=KEY_COLUMNS, how="inner", validate="many_to_one")
    if common.empty:
        raise RuntimeError(
            "No validation target has predictions from every candidate method. "
            "Inspect 10a/10b prediction coverage."
        )
    return common


def make_recommendations(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        metrics["support"].eq("common_support")
        & metrics["holdout_mode"].eq(cfg.STACKING_HOLDOUT_MODE)
    ].copy()

    ml = (
        primary[
            primary["method_family"].eq("ML")
            & ~primary["method"].eq("ffnn_mlp")
        ]
        .sort_values(["rmse", "mae"])
        .head(len(cfg.SELECTED_ML_MODELS))
        .copy()
    )
    ml["recommendation_role"] = "selected_ml_base_learner"

    gi = primary[
        primary["method_family"].eq("GI")
        & primary["method"].isin(cfg.SELECTED_INTERPOLATION_METHODS)
    ].sort_values(["rmse", "mae"]).copy()
    gi["recommendation_role"] = "selected_gi_base_learner"

    recs = pd.concat([ml, gi], ignore_index=True)
    recs["configured_for_stacking"] = recs["method"].isin(
        cfg.SELECTED_ML_MODELS + cfg.SELECTED_INTERPOLATION_METHODS
    )
    return recs


def plot_rmse(metrics: pd.DataFrame) -> None:
    for support in ["common_support", "method_specific"]:
        for mode in cfg.HOLDOUT_MODES:
            sub = metrics[
                metrics["support"].eq(support)
                & metrics["holdout_mode"].eq(mode)
            ].sort_values("rmse")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(sub))))
            labels = sub["method"] + " | " + sub["method_family"]
            ax.barh(labels, sub["rmse"])
            ax.invert_yaxis()
            ax.set_xlabel("Pooled RMSE")
            ax.set_title(f"2024 validation: {mode}, {support.replace('_', ' ')}")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            fig.savefig(FIG_DIR / f"combined_rmse_{mode}_{support}.pdf", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    print("10c: Compare validation methods on aligned 2024 targets")
    print("=" * 78)
    predictions = load_predictions()
    common = common_support_predictions(predictions)

    rows = metric_rows(predictions, "method_specific")
    rows.extend(metric_rows(common, "common_support"))
    metrics = pd.DataFrame(rows).sort_values(
        ["support", "holdout_mode", "rmse", "mae", "method"]
    )

    coverage = (
        predictions.assign(finite=np.isfinite(predictions["prediction"]))
        .groupby(["holdout_mode", "method_family", "method"], as_index=False)
        .agg(n_targets=(cfg.KEY, "size"), n_predictions=("finite", "sum"))
    )
    coverage["coverage"] = coverage["n_predictions"] / coverage["n_targets"]

    recommendations = make_recommendations(metrics)
    recommended_ml = set(
        recommendations.loc[
            recommendations["recommendation_role"].eq("selected_ml_base_learner"),
            "method",
        ]
    )
    configured_ml = set(cfg.SELECTED_ML_MODELS)
    if recommended_ml != configured_ml:
        recommendations.to_csv(RECOMMENDATION_PATH, index=False)
        raise RuntimeError(
            "The top three 2024 ML models differ from SELECTED_ML_MODELS in "
            "00_config.py. Review recommended_base_models.csv, update the "
            "configuration deliberately, and rerun 10c onward. "
            f"Recommended={sorted(recommended_ml)}, configured={sorted(configured_ml)}"
        )

    metrics.to_csv(COMBINED_PATH, index=False)
    coverage.to_csv(COVERAGE_PATH, index=False)
    recommendations.to_csv(RECOMMENDATION_PATH, index=False)
    plot_rmse(metrics)

    report_lines = [
        "SMAP gap-filling aligned validation comparison",
        "=" * 52,
        f"Project seed: {cfg.RANDOM_SEED}",
        f"Final ML feature group: {cfg.FINAL_ML_FEATURE_GROUP}",
        f"Candidate methods: {sorted(predictions['method'].unique())}",
        f"Common-support prediction rows: {len(common):,}",
        "",
        "Spatial-block common-support metrics",
        "-" * 52,
        metrics[
            metrics["support"].eq("common_support")
            & metrics["holdout_mode"].eq("spatial_block")
        ].to_string(index=False),
        "",
        "Recommendations",
        "-" * 52,
        recommendations.to_string(index=False),
    ]
    SUMMARY_PATH.write_text("\n".join(report_lines))

    print("\nSaved:")
    print(f"  {COMBINED_PATH}")
    print(f"  {COVERAGE_PATH}")
    print(f"  {RECOMMENDATION_PATH}")
    print(f"  {SUMMARY_PATH}")
    print("\nSpatial-block common-support metrics:")
    print(
        metrics[
            metrics["support"].eq("common_support")
            & metrics["holdout_mode"].eq("spatial_block")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
