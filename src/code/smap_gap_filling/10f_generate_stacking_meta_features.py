#!/usr/bin/env python3
"""
10f_generate_stacking_meta_features.py

Collect per-pixel base-model predictions from the 2024 validation run
and combine them into a single meta-training table for the stacking meta-model.

This script does NOT train any model.
It only reads outputs already produced by 10a and 10b, then joins them
on (date, pass, smap_pixel_key) so each row contains:
    - the true soil_moisture (observed)
    - predictions from every base model / interpolation method
    - spatial and temporal covariates (x, y, sin_doy, cos_doy, pass_pm)

The output is used by 10g to train the Ridge meta-model.

Inputs
------
05_gapfill_model_validation/ml/ml_validation_predictions_sample.csv
    Produced by 10a.  Must contain columns:
        date, pass, smap_pixel_key, soil_moisture, model, feature_group, prediction

05_gapfill_model_validation/interpolation/interpolation_validation_predictions_sample.csv
    Produced by 10b.  Must contain columns:
        date, pass, smap_pixel_key, soil_moisture, method, prediction

Outputs
-------
05_gapfill_model_validation/stacking/
    meta_training_table.csv
        One row per (date, pass, smap_pixel_key).
        Columns: soil_moisture, x, y, sin_doy, cos_doy, pass_pm,
                 pred_xgboost, pred_hist_gbdt, pred_random_forest,
                 pred_centroid_ordinary_kriging, pred_nearest_neighbor_same_day
    meta_training_summary.txt
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():
    config_path = Path(__file__).resolve().with_name("00_config.py")
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# SETTINGS
# ============================================================

TARGET = "soil_moisture"
KEY = "smap_pixel_key"

# The ML feature_group that was selected as best in 10c.
# Change this if 10c recommends a different group.
BEST_ML_FEATURE_GROUP = "all_iem_spatiotemporal"

# Which ML models to include in the meta table.
ML_MODELS_TO_STACK = ["xgboost", "hist_gbdt", "random_forest"]

# Which interpolation methods to include.
INTERP_METHODS_TO_STACK = ["centroid_ordinary_kriging", "nearest_neighbor_same_day"]

# Only use spatial_block holdouts for the meta-training table.
# These are harder and more realistic — a meta-model trained on them
# will generalise better to true clustered gaps.
HOLDOUT_MODE_FOR_META = "spatial_block"

# Validation split only — do NOT include test predictions here.
SPLIT_FOR_META = "validation"

RANDOM_STATE = 42


# ============================================================
# PATHS
# ============================================================

GAP_FILLING_DIR = cfg.GAP_FILLING_DIR

VAL_DIR = GAP_FILLING_DIR / "05_gapfill_model_validation"
ML_PREDS_PATH = VAL_DIR / "ml" / "ml_validation_predictions_sample.csv"
INTERP_PREDS_PATH = (
    VAL_DIR / "interpolation" / "interpolation_validation_predictions_sample.csv"
)

OUT_DIR = VAL_DIR / "stacking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_TABLE_PATH = OUT_DIR / "meta_training_table.csv"
SUMMARY_PATH = OUT_DIR / "meta_training_summary.txt"


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Make sure 10a and 10b have been run first."
        )


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Re-derive sin_doy / cos_doy / pass_pm from date and pass columns."""
    out = df.copy()
    if "date" in out.columns:
        dt = pd.to_datetime(out["date"], errors="coerce")
        doy = dt.dt.dayofyear.fillna(1).astype(float)
        out["sin_doy"] = np.sin(2.0 * np.pi * doy / 366.0)
        out["cos_doy"] = np.cos(2.0 * np.pi * doy / 366.0)
    if "pass" in out.columns and "pass_pm" not in out.columns:
        out["pass_pm"] = (out["pass"].str.lower() == "pm").astype(int)
    return out


# ============================================================
# LOAD ML PREDICTIONS
# ============================================================

def load_ml_predictions() -> pd.DataFrame:
    """
    Read 10a output and pivot so each ML model becomes a column.

    Returns a DataFrame indexed by (date, pass, smap_pixel_key) with
    columns: soil_moisture, x, y, sin_doy, cos_doy, pass_pm,
             pred_xgboost, pred_hist_gbdt, pred_random_forest
    """
    require_file(ML_PREDS_PATH)
    print(f"Reading ML predictions: {ML_PREDS_PATH}")
    df = pd.read_csv(ML_PREDS_PATH, low_memory=False)

    required = {"date", "pass", KEY, TARGET, "model", "feature_group", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ML predictions file missing columns: {sorted(missing)}")

    # Keep only the selected feature group and split
    df = df[
        df["feature_group"].eq(BEST_ML_FEATURE_GROUP)
        & df.get("split", pd.Series(SPLIT_FOR_META, index=df.index)).eq(SPLIT_FOR_META)
        & df.get("holdout_mode", pd.Series(HOLDOUT_MODE_FOR_META, index=df.index)).eq(HOLDOUT_MODE_FOR_META)
    ].copy()

    df = df[df["model"].isin(ML_MODELS_TO_STACK)].copy()

    if df.empty:
        raise RuntimeError(
            f"No ML predictions found for feature_group='{BEST_ML_FEATURE_GROUP}', "
            f"split='{SPLIT_FOR_META}', holdout_mode='{HOLDOUT_MODE_FOR_META}'. "
            "Check that 10a saved predictions with these labels."
        )

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    # Pivot models into columns
    wide = df.pivot_table(
        index=["date", "pass", KEY],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={m: f"pred_{m}" for m in ML_MODELS_TO_STACK if m in wide.columns})

    # Attach true target + spatial covariates from the first model's rows
    first_model = ML_MODELS_TO_STACK[0]
    base_cols = ["date", "pass", KEY, TARGET]
    for c in ["x", "y", "sin_doy", "cos_doy", "pass_pm"]:
        if c in df.columns:
            base_cols.append(c)

    anchor = (
        df[df["model"].eq(first_model)][base_cols]
        .drop_duplicates(["date", "pass", KEY])
        .copy()
    )

    out = anchor.merge(wide, on=["date", "pass", KEY], how="left")
    out = add_temporal_features(out)

    print(f"  ML rows after pivot: {len(out):,}")
    return out


# ============================================================
# LOAD INTERPOLATION PREDICTIONS
# ============================================================

def load_interp_predictions() -> pd.DataFrame:
    """
    Read 10b output and pivot so each interpolation method becomes a column.

    Returns a DataFrame indexed by (date, pass, smap_pixel_key) with
    columns: pred_centroid_ordinary_kriging, pred_nearest_neighbor_same_day
    """
    require_file(INTERP_PREDS_PATH)
    print(f"Reading interpolation predictions: {INTERP_PREDS_PATH}")
    df = pd.read_csv(INTERP_PREDS_PATH, low_memory=False)

    required = {"date", "pass", KEY, "method", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Interpolation predictions file missing columns: {sorted(missing)}")

    # Filter split / holdout if those columns exist
    if "split" in df.columns:
        df = df[df["split"].eq(SPLIT_FOR_META)].copy()
    if "holdout_mode" in df.columns:
        df = df[df["holdout_mode"].eq(HOLDOUT_MODE_FOR_META)].copy()

    df = df[df["method"].isin(INTERP_METHODS_TO_STACK)].copy()

    if df.empty:
        raise RuntimeError(
            f"No interpolation predictions found for split='{SPLIT_FOR_META}', "
            f"holdout_mode='{HOLDOUT_MODE_FOR_META}'. Check 10b output."
        )

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    wide = df.pivot_table(
        index=["date", "pass", KEY],
        columns="method",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={m: f"pred_{m}" for m in INTERP_METHODS_TO_STACK if m in wide.columns}
    )

    print(f"  Interpolation rows after pivot: {len(wide):,}")
    return wide


# ============================================================
# BUILD META TABLE
# ============================================================

def build_meta_table(ml: pd.DataFrame, interp: pd.DataFrame) -> pd.DataFrame:
    """
    Join ML and interpolation predictions on (date, pass, smap_pixel_key).
    Drop rows where the true target is missing.
    """
    meta = ml.merge(interp, on=["date", "pass", KEY], how="outer")

    # If target came only from ml, fill it back after outer join
    if TARGET not in meta.columns:
        meta[TARGET] = np.nan

    meta[TARGET] = pd.to_numeric(meta[TARGET], errors="coerce")
    meta = meta[meta[TARGET].notna()].copy()

    if meta.empty:
        raise RuntimeError(
            "Meta table is empty after joining ML and interpolation predictions. "
            "Check that both 10a and 10b ran on the same dates."
        )

    # Ensure all expected prediction columns exist (fill with NaN if absent)
    all_pred_cols = (
        [f"pred_{m}" for m in ML_MODELS_TO_STACK]
        + [f"pred_{m}" for m in INTERP_METHODS_TO_STACK]
    )
    for c in all_pred_cols:
        if c not in meta.columns:
            meta[c] = np.nan

    # Drop rows where ALL base predictions are NaN (unpredictable pixels)
    pred_cols_present = [c for c in all_pred_cols if c in meta.columns]
    has_any_pred = meta[pred_cols_present].notna().any(axis=1)
    n_before = len(meta)
    meta = meta[has_any_pred].copy()
    n_dropped = n_before - len(meta)

    print(f"  Rows with at least one base prediction: {len(meta):,}  (dropped {n_dropped:,} all-NaN)")

    # Column order
    id_cols = ["date", "pass", KEY]
    covariate_cols = [c for c in ["x", "y", "sin_doy", "cos_doy", "pass_pm"] if c in meta.columns]
    target_col = [TARGET]

    ordered = id_cols + target_col + covariate_cols + pred_cols_present
    extra = [c for c in meta.columns if c not in ordered]
    meta = meta[ordered + extra].reset_index(drop=True)

    return meta


# ============================================================
# WRITE SUMMARY
# ============================================================

def write_summary(meta: pd.DataFrame) -> None:
    pred_cols = [c for c in meta.columns if c.startswith("pred_")]

    lines = [
        "Meta-training table summary",
        "=" * 50,
        f"Rows:              {len(meta):,}",
        f"Unique dates:      {meta['date'].nunique():,}",
        f"Holdout mode:      {HOLDOUT_MODE_FOR_META}",
        f"Split:             {SPLIT_FOR_META}",
        f"ML feature group:  {BEST_ML_FEATURE_GROUP}",
        "",
        "Base prediction columns:",
    ]
    for c in pred_cols:
        n_valid = meta[c].notna().sum()
        pct = 100.0 * n_valid / len(meta)
        lines.append(f"  {c:<45} {n_valid:>8,}  ({pct:.1f}% non-null)")

    lines += [
        "",
        f"Target ({TARGET}) stats:",
        f"  mean:  {meta[TARGET].mean():.4f}",
        f"  std:   {meta[TARGET].std():.4f}",
        f"  min:   {meta[TARGET].min():.4f}",
        f"  max:   {meta[TARGET].max():.4f}",
    ]

    text = "\n".join(lines)
    print("\n" + text)
    SUMMARY_PATH.write_text(text)
    print(f"\nSummary written to: {SUMMARY_PATH}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("10f: Generate stacking meta-features")
    print("=" * 70)
    print(f"ML predictions:            {ML_PREDS_PATH}")
    print(f"Interpolation predictions: {INTERP_PREDS_PATH}")
    print(f"Output folder:             {OUT_DIR}")
    print(f"Holdout mode for meta:     {HOLDOUT_MODE_FOR_META}")
    print(f"Split:                     {SPLIT_FOR_META}")
    print(f"ML feature group:          {BEST_ML_FEATURE_GROUP}")
    print("=" * 70)

    ml = load_ml_predictions()
    interp = load_interp_predictions()

    print("\nBuilding meta table...")
    meta = build_meta_table(ml, interp)

    meta.to_csv(META_TABLE_PATH, index=False)
    print(f"Meta table saved: {META_TABLE_PATH}  ({len(meta):,} rows)")

    write_summary(meta)
    print("\nDone.")


if __name__ == "__main__":
    main()
