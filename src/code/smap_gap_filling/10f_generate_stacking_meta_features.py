#!/usr/bin/env python3
"""
10f_generate_stacking_meta_features.py
-- updated to include regression_kriging as a base model --
"""

from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


def load_config():
    config_path = Path(__file__).resolve().with_name("00_config.py")
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()

TARGET = "soil_moisture"
KEY    = "smap_pixel_key"

BEST_ML_FEATURE_GROUP = "all_iem_spatiotemporal"
ML_MODELS_TO_STACK    = ["xgboost", "hist_gbdt", "random_forest"]

# ← regression_kriging added here
INTERP_METHODS_TO_STACK = [
    "centroid_ordinary_kriging",
    "nearest_neighbor_same_day",
    "regression_kriging",
]

HOLDOUT_MODE_FOR_META = "spatial_block"
SPLIT_FOR_META        = "validation"
RANDOM_STATE          = 42

GAP_FILLING_DIR  = cfg.GAP_FILLING_DIR
VAL_DIR          = GAP_FILLING_DIR / "05_gapfill_model_validation"
ML_PREDS_PATH    = VAL_DIR / "ml" / "ml_validation_predictions_sample.csv"
INTERP_PREDS_PATH = VAL_DIR / "interpolation" / "interpolation_validation_predictions_sample.csv"

OUT_DIR          = VAL_DIR / "stacking"
OUT_DIR.mkdir(parents=True, exist_ok=True)
META_TABLE_PATH  = OUT_DIR / "meta_training_table.csv"
SUMMARY_PATH     = OUT_DIR / "meta_training_summary.txt"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}\nRun 10a and 10b first.")


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        dt  = pd.to_datetime(out["date"], errors="coerce")
        doy = dt.dt.dayofyear.fillna(1).astype(float)
        out["sin_doy"] = np.sin(2.0 * np.pi * doy / 366.0)
        out["cos_doy"] = np.cos(2.0 * np.pi * doy / 366.0)
    if "pass" in out.columns and "pass_pm" not in out.columns:
        out["pass_pm"] = (out["pass"].str.lower() == "pm").astype(int)
    return out


def load_ml_predictions() -> pd.DataFrame:
    require_file(ML_PREDS_PATH)
    print(f"Reading ML predictions: {ML_PREDS_PATH}")
    df = pd.read_csv(ML_PREDS_PATH, low_memory=False)

    required = {"date", "pass", KEY, TARGET, "model", "feature_group", "prediction"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"ML predictions missing columns: {sorted(missing)}")

    df = df[
        df["feature_group"].eq(BEST_ML_FEATURE_GROUP)
        & df.get("split",        pd.Series(SPLIT_FOR_META,        index=df.index)).eq(SPLIT_FOR_META)
        & df.get("holdout_mode", pd.Series(HOLDOUT_MODE_FOR_META, index=df.index)).eq(HOLDOUT_MODE_FOR_META)
    ].copy()
    df = df[df["model"].isin(ML_MODELS_TO_STACK)].copy()

    if df.empty:
        raise RuntimeError(
            f"No ML predictions for feature_group='{BEST_ML_FEATURE_GROUP}', "
            f"split='{SPLIT_FOR_META}', holdout_mode='{HOLDOUT_MODE_FOR_META}'."
        )

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df[TARGET]       = pd.to_numeric(df[TARGET],       errors="coerce")

    wide = df.pivot_table(
        index=["date", "pass", KEY], columns="model",
        values="prediction", aggfunc="first", dropna=False,
    ).reset_index()
    wide.columns.name = None
    # Guarantee every expected model column exists even if entirely empty,
    # so a missing base model never silently vanishes from the meta table.
    for m in ML_MODELS_TO_STACK:
        if m not in wide.columns:
            wide[m] = np.nan
    wide = wide.rename(columns={m: f"pred_{m}" for m in ML_MODELS_TO_STACK if m in wide.columns})

    first_model = ML_MODELS_TO_STACK[0]
    base_cols   = ["date", "pass", KEY, TARGET]
    for c in ["x", "y", "sin_doy", "cos_doy", "pass_pm"]:
        if c in df.columns:
            base_cols.append(c)

    anchor = (
        df[df["model"].eq(first_model)][base_cols]
        .drop_duplicates(["date", "pass", KEY]).copy()
    )
    out = anchor.merge(wide, on=["date", "pass", KEY], how="left")
    out = add_temporal_features(out)
    print(f"  ML rows after pivot: {len(out):,}")
    return out


def load_interp_predictions() -> pd.DataFrame:
    require_file(INTERP_PREDS_PATH)
    print(f"Reading interpolation predictions: {INTERP_PREDS_PATH}")
    df = pd.read_csv(INTERP_PREDS_PATH, low_memory=False)

    required = {"date", "pass", KEY, "method", "prediction"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Interpolation predictions missing columns: {sorted(missing)}")

    if "split"        in df.columns: df = df[df["split"].eq(SPLIT_FOR_META)].copy()
    if "holdout_mode" in df.columns: df = df[df["holdout_mode"].eq(HOLDOUT_MODE_FOR_META)].copy()

    df = df[df["method"].isin(INTERP_METHODS_TO_STACK)].copy()
    if df.empty:
        raise RuntimeError(
            f"No interpolation predictions for split='{SPLIT_FOR_META}', "
            f"holdout_mode='{HOLDOUT_MODE_FOR_META}'. "
            "If regression_kriging is new, re-run 10b first."
        )

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    wide = df.pivot_table(
        index=["date", "pass", KEY], columns="method",
        values="prediction", aggfunc="first", dropna=False,
    ).reset_index()
    wide.columns.name = None
    # Guarantee every expected method column exists even if entirely empty.
    # This is the fix that stops regression_kriging from disappearing here.
    for m in INTERP_METHODS_TO_STACK:
        if m not in wide.columns:
            wide[m] = np.nan
    wide = wide.rename(
        columns={m: f"pred_{m}" for m in INTERP_METHODS_TO_STACK if m in wide.columns}
    )
    print(f"  Interpolation rows after pivot: {len(wide):,}")
    return wide


def build_meta_table(ml: pd.DataFrame, interp: pd.DataFrame) -> pd.DataFrame:
    meta = ml.merge(interp, on=["date", "pass", KEY], how="outer")
    if TARGET not in meta.columns:
        meta[TARGET] = np.nan
    meta[TARGET] = pd.to_numeric(meta[TARGET], errors="coerce")
    meta = meta[meta[TARGET].notna()].copy()

    if meta.empty:
        raise RuntimeError("Meta table is empty. Check that 10a and 10b ran on the same dates.")

    all_pred_cols = (
        [f"pred_{m}" for m in ML_MODELS_TO_STACK]
        + [f"pred_{m}" for m in INTERP_METHODS_TO_STACK]
    )
    for c in all_pred_cols:
        if c not in meta.columns:
            meta[c] = np.nan

    pred_cols_present = [c for c in all_pred_cols if c in meta.columns]
    has_any = meta[pred_cols_present].notna().any(axis=1)
    n_before = len(meta)
    meta = meta[has_any].copy()
    print(f"  Rows with ≥1 base prediction: {len(meta):,}  (dropped {n_before - len(meta):,})")

    # Per-method coverage: a near-empty base method (e.g. regression_kriging)
    # is the symptom of an upstream silent failure. Make it visible.
    print("\n  Base method coverage in meta table:")
    for c in pred_cols_present:
        frac = meta[c].notna().mean() if len(meta) else 0.0
        flag = "   <-- LOW, check upstream stage" if frac < 0.05 else ""
        print(f"    {c:<40} {frac*100:5.1f}%{flag}")

    id_cols  = ["date", "pass", KEY]
    cov_cols = [c for c in ["x", "y", "sin_doy", "cos_doy", "pass_pm"] if c in meta.columns]
    ordered  = id_cols + [TARGET] + cov_cols + pred_cols_present
    extra    = [c for c in meta.columns if c not in ordered]
    return meta[ordered + extra].reset_index(drop=True)


def write_summary(meta: pd.DataFrame) -> None:
    pred_cols = [c for c in meta.columns if c.startswith("pred_")]
    lines = [
        "Meta-training table summary", "=" * 50,
        f"Rows:              {len(meta):,}",
        f"Unique dates:      {meta['date'].nunique():,}",
        f"Holdout mode:      {HOLDOUT_MODE_FOR_META}",
        f"Split:             {SPLIT_FOR_META}",
        f"ML feature group:  {BEST_ML_FEATURE_GROUP}",
        "", "Base prediction columns:",
    ]
    for c in pred_cols:
        n_valid = meta[c].notna().sum()
        lines.append(f"  {c:<48} {n_valid:>8,}  ({100*n_valid/len(meta):.1f}%)")
    lines += [
        "", f"Target ({TARGET}) stats:",
        f"  mean: {meta[TARGET].mean():.4f}",
        f"  std:  {meta[TARGET].std():.4f}",
        f"  min:  {meta[TARGET].min():.4f}",
        f"  max:  {meta[TARGET].max():.4f}",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    SUMMARY_PATH.write_text(text)
    print(f"\nSummary: {SUMMARY_PATH}")


def main() -> None:
    print("10f: Generate stacking meta-features (with regression_kriging)")
    print("=" * 70)
    print(f"ML predictions:            {ML_PREDS_PATH}")
    print(f"Interpolation predictions: {INTERP_PREDS_PATH}")
    print(f"Interp methods to stack:   {INTERP_METHODS_TO_STACK}")
    print("=" * 70)

    ml    = load_ml_predictions()
    interp = load_interp_predictions()

    print("\nBuilding meta table...")
    meta = build_meta_table(ml, interp)
    meta.to_csv(META_TABLE_PATH, index=False)
    print(f"Meta table saved: {META_TABLE_PATH}  ({len(meta):,} rows)")
    write_summary(meta)
    print("\nDone.")


if __name__ == "__main__":
    main()