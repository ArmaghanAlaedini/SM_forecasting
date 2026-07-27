#!/usr/bin/env python3
"""Evaluate the frozen 2024 ridge stack on common 2025 artificial gaps.

This is the independent test of the main stacked model.  The script aligns the
selected ML and GI predictions on the same 2025 target keys, requires all six
base predictions, applies the saved 2024 ridge bundle, and compares every base
learner with the stack on identical rows.
"""

from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapfill_workflow_common import cfg, compute_metrics


ML_PATH = cfg.ML_TEST_DIR / "ml_selected_test_predictions.csv"
GI_PATH = cfg.INTERP_TEST_DIR / "interpolation_selected_test_predictions.csv"
MODEL_PATH = cfg.META_MODEL_PATH
OUT_DIR = cfg.STACKING_TEST_DIR
PREDICTION_PATH = OUT_DIR / "stacking_selected_test_predictions.csv"
METRICS_PATH = OUT_DIR / "stacking_selected_test_metrics.csv"
COVERAGE_PATH = OUT_DIR / "stacking_selected_test_coverage.csv"
REPORT_PATH = OUT_DIR / "stacking_selected_test_report.txt"
FIGURE_PATH = OUT_DIR / "stacking_selected_test_rmse.pdf"

OUT_DIR.mkdir(parents=True, exist_ok=True)
KEY_COLUMNS = ["split", "holdout_mode", "date", "pass", cfg.KEY]


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_model_bundle() -> dict:
    require(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            "The saved meta-model is not the expected bundle. Re-run 10g with the fixed script."
        )
    if bundle.get("feature_columns") != list(cfg.META_FEATURE_COLUMNS):
        raise ValueError(
            "Meta-model feature contract differs from 00_config.py. Re-run 10f and 10g."
        )
    return bundle


def load_wide_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(ML_PATH)
    require(GI_PATH)

    ml = pd.read_csv(ML_PATH, low_memory=False)
    ml = ml[ml["model"].isin(cfg.SELECTED_ML_MODELS)].copy()
    gi = pd.read_csv(GI_PATH, low_memory=False)
    gi = gi[gi["method"].isin(cfg.SELECTED_INTERPOLATION_METHODS)].copy()

    for frame in [ml, gi]:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame[cfg.KEY] = frame[cfg.KEY].astype(str)
        frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
        frame["observed"] = pd.to_numeric(frame["observed"], errors="coerce")

    context = (
        ml[[*KEY_COLUMNS, "observed", "x", "y"]]
        .drop_duplicates(KEY_COLUMNS)
        .copy()
    )
    ml_wide = ml.pivot(index=KEY_COLUMNS, columns="model", values="prediction").reset_index()
    ml_wide.columns.name = None
    ml_wide = ml_wide.rename(
        columns={model: f"pred_{model}" for model in cfg.SELECTED_ML_MODELS}
    )

    gi_wide = gi.pivot(index=KEY_COLUMNS, columns="method", values="prediction").reset_index()
    gi_wide.columns.name = None
    gi_wide = gi_wide.rename(
        columns={
            method: f"pred_{method}" for method in cfg.SELECTED_INTERPOLATION_METHODS
        }
    )
    return ml_wide.merge(gi_wide, on=KEY_COLUMNS, how="outer", validate="one_to_one"), context


def add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    doy = out["date"].dt.dayofyear.astype(float)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 366.0)
    out["pass_pm"] = out["pass"].astype(str).str.lower().eq("pm").astype(int)
    return out


def main() -> None:
    print("10h: Independent 2025 evaluation of the ridge stack")
    print("=" * 78)
    bundle = load_model_bundle()
    union, context = load_wide_predictions()

    coverage_rows = []
    for column in cfg.BASE_PREDICTION_COLUMNS:
        if column not in union.columns:
            union[column] = np.nan
        coverage_rows.append(
            {
                "feature": column,
                "available_rows": int(union[column].notna().sum()),
                "union_rows": len(union),
                "coverage": float(union[column].notna().mean()),
            }
        )

    aligned = union.merge(context, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    complete = np.isfinite(aligned[cfg.BASE_PREDICTION_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    aligned = aligned.loc[complete].copy()
    if aligned.empty:
        raise RuntimeError("No 2025 test row contains all six base predictions.")

    aligned = add_context_features(aligned)
    X_meta = aligned[cfg.META_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(X_meta.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite 2025 meta-features remain after alignment.")

    aligned["pred_stacking"] = np.asarray(
        bundle["pipeline"].predict(X_meta), dtype=float
    )

    method_columns = {
        **{column.removeprefix("pred_"): column for column in cfg.BASE_PREDICTION_COLUMNS},
        "stacking": "pred_stacking",
    }
    metric_rows: list[dict] = []
    long_parts: list[pd.DataFrame] = []

    base_context = aligned[[*KEY_COLUMNS, "observed", "x", "y"]]
    for method, column in method_columns.items():
        out = base_context.copy()
        out["method"] = method
        out["prediction"] = aligned[column].to_numpy(dtype=float)
        long_parts.append(out)
        for holdout_mode, sub in out.groupby("holdout_mode", sort=True):
            metric_rows.append(
                {
                    "split": "test",
                    "holdout_mode": holdout_mode,
                    "support": "common_six_model_support",
                    "method": method,
                    **compute_metrics(sub["observed"], sub["prediction"]),
                    "coverage": 1.0,
                }
            )

    long_predictions = pd.concat(long_parts, ignore_index=True).sort_values(
        ["holdout_mode", "date", "pass", cfg.KEY, "method"]
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["holdout_mode", "rmse", "mae", "method"]
    )

    long_predictions.to_csv(PREDICTION_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    pd.DataFrame(coverage_rows).to_csv(COVERAGE_PATH, index=False)

    spatial = metrics[metrics["holdout_mode"].eq(cfg.STACKING_HOLDOUT_MODE)].copy()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(spatial))))
    spatial = spatial.sort_values("rmse")
    ax.barh(spatial["method"], spatial["rmse"])
    ax.invert_yaxis()
    ax.set_xlabel("Pooled RMSE")
    ax.set_title("2025 independent test on common six-model support")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)

    report = [
        "Independent 2025 stacking test",
        "=" * 48,
        f"Project seed: {cfg.RANDOM_SEED}",
        f"Meta-model training year: {bundle.get('training_year')}",
        f"Ridge alpha: {bundle.get('best_alpha')}",
        f"Union target rows: {len(union):,}",
        f"Complete six-model rows: {len(aligned):,}",
        "",
        metrics.to_string(index=False),
    ]
    REPORT_PATH.write_text("\n".join(report))

    print("\n".join(report))
    print("\nSaved:")
    print(f"  {PREDICTION_PATH}")
    print(f"  {METRICS_PATH}")
    print(f"  {COVERAGE_PATH}")
    print(f"  {REPORT_PATH}")
    print(f"  {FIGURE_PATH}")


if __name__ == "__main__":
    main()
