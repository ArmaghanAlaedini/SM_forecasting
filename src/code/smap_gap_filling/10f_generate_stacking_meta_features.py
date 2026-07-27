#!/usr/bin/env python3
"""Build a complete, aligned 2024 stacking meta-training table.

Only spatial-block validation rows are used.  The table is constructed with an
inner join on ``date``, ``pass``, and ``smap_pixel_key`` and contains one real
prediction from every selected base learner.  Missing base predictions are not
imputed and incomplete rows are excluded with their coverage reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gapfill_workflow_common import cfg


ML_PATH = cfg.ML_VALIDATION_DIR / "ml_validation_predictions.csv"
GI_PATH = cfg.INTERP_VALIDATION_DIR / "interpolation_validation_predictions.csv"
OUT_DIR = cfg.STACKING_DIR
META_PATH = OUT_DIR / "meta_training_table.csv"
SUMMARY_PATH = OUT_DIR / "meta_training_summary.txt"
COVERAGE_PATH = OUT_DIR / "meta_training_coverage.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
KEY_COLUMNS = ["date", "pass", cfg.KEY]


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def add_meta_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    doy = out["date"].dt.dayofyear.astype(float)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 366.0)
    out["pass_pm"] = out["pass"].astype(str).str.lower().eq("pm").astype(int)
    return out


def load_ml() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(ML_PATH)
    df = pd.read_csv(ML_PATH, low_memory=False)
    df = df[
        df["split"].eq("validation")
        & df["holdout_mode"].eq(cfg.STACKING_HOLDOUT_MODE)
        & df["feature_group"].eq(cfg.FINAL_ML_FEATURE_GROUP)
        & df["model"].isin(cfg.SELECTED_ML_MODELS)
    ].copy()
    if df.empty:
        raise RuntimeError("No selected ML spatial-block validation predictions found.")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df[cfg.KEY] = df[cfg.KEY].astype(str)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["observed"] = pd.to_numeric(df["observed"], errors="coerce")

    context = (
        df[[*KEY_COLUMNS, "observed", "x", "y"]]
        .drop_duplicates(KEY_COLUMNS)
        .copy()
    )
    wide = df.pivot(index=KEY_COLUMNS, columns="model", values="prediction").reset_index()
    wide.columns.name = None
    rename = {model: f"pred_{model}" for model in cfg.SELECTED_ML_MODELS}
    wide = wide.rename(columns=rename)
    return wide, context


def load_gi() -> pd.DataFrame:
    require(GI_PATH)
    df = pd.read_csv(GI_PATH, low_memory=False)
    df = df[
        df["split"].eq("validation")
        & df["holdout_mode"].eq(cfg.STACKING_HOLDOUT_MODE)
        & df["method"].isin(cfg.SELECTED_INTERPOLATION_METHODS)
    ].copy()
    if df.empty:
        raise RuntimeError("No selected GI spatial-block validation predictions found.")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df[cfg.KEY] = df[cfg.KEY].astype(str)
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    wide = df.pivot(index=KEY_COLUMNS, columns="method", values="prediction").reset_index()
    wide.columns.name = None
    rename = {
        method: f"pred_{method}" for method in cfg.SELECTED_INTERPOLATION_METHODS
    }
    return wide.rename(columns=rename)


def main() -> None:
    print("10f: Build complete aligned stacking meta-features")
    print("=" * 78)
    print(f"Holdout mode: {cfg.STACKING_HOLDOUT_MODE}")
    print(f"Seed:         {cfg.RANDOM_SEED}")
    print("=" * 78)

    ml, context = load_ml()
    gi = load_gi()

    # Outer table is used only to document availability before filtering.
    availability = ml.merge(gi, on=KEY_COLUMNS, how="outer", validate="one_to_one")
    coverage_rows = []
    for column in cfg.BASE_PREDICTION_COLUMNS:
        if column not in availability.columns:
            availability[column] = np.nan
        coverage_rows.append(
            {
                "feature": column,
                "available_rows": int(availability[column].notna().sum()),
                "total_union_rows": len(availability),
                "coverage": float(availability[column].notna().mean()),
            }
        )

    meta = ml.merge(gi, on=KEY_COLUMNS, how="inner", validate="one_to_one")
    meta = meta.merge(context, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    missing_columns = [c for c in cfg.BASE_PREDICTION_COLUMNS if c not in meta.columns]
    if missing_columns:
        raise ValueError(f"Missing required base prediction columns: {missing_columns}")

    complete = np.isfinite(meta[cfg.BASE_PREDICTION_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    n_before = len(meta)
    meta = meta.loc[complete].copy()
    if meta.empty:
        raise RuntimeError(
            "No complete stacking rows remain. Every row must contain all six base predictions."
        )

    meta = add_meta_context(meta)
    meta["soil_moisture"] = pd.to_numeric(meta["observed"], errors="coerce")
    meta["year"] = meta["date"].dt.year
    meta["holdout_mode"] = cfg.STACKING_HOLDOUT_MODE
    meta["split"] = "validation"

    ordered = [
        "date",
        "year",
        "pass",
        cfg.KEY,
        "split",
        "holdout_mode",
        "soil_moisture",
        "x",
        "y",
        "sin_doy",
        "cos_doy",
        "pass_pm",
        *cfg.BASE_PREDICTION_COLUMNS,
    ]
    meta = meta[ordered].sort_values(["date", "pass", cfg.KEY]).reset_index(drop=True)

    pd.DataFrame(coverage_rows).to_csv(COVERAGE_PATH, index=False)
    meta.to_csv(META_PATH, index=False)

    lines = [
        "Aligned stacking meta-training summary",
        "=" * 52,
        f"Project seed: {cfg.RANDOM_SEED}",
        f"Split: validation",
        f"Holdout mode: {cfg.STACKING_HOLDOUT_MODE}",
        f"ML feature group: {cfg.FINAL_ML_FEATURE_GROUP}",
        f"Union rows before inner join: {len(availability):,}",
        f"Rows after ML/GI inner join: {n_before:,}",
        f"Complete six-prediction rows: {len(meta):,}",
        f"Rows removed for a missing/non-finite base prediction: {n_before - len(meta):,}",
        f"Unique date-pass groups: {meta[['date', 'pass']].drop_duplicates().shape[0]:,}",
        "",
        "Every saved meta row has a real finite prediction from all six base learners.",
    ]
    SUMMARY_PATH.write_text("\n".join(lines))

    print("\n".join(lines))
    print("\nSaved:")
    print(f"  {META_PATH}")
    print(f"  {COVERAGE_PATH}")
    print(f"  {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
