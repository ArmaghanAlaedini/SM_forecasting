#!/usr/bin/env python3
"""
11c_stack_and_finalize_gapfills.py

Merge original complete SMAP-IEM files with 11a ML predictions and 11b
interpolation predictions, then create final daily gap-filled files.

This script:
  - does NOT train models
  - does NOT perform true stacking yet

Current final rule:
  observed rows: use original soil_moisture
  missing rows: use centroid_ordinary_kriging first
                then nearest_neighbor_same_day
                then xgboost / hist_gbdt / random_forest if needed

Manual controls are in:
    11_gapfilling_setting.py

Outputs:
    src/data/processed/smap_gap_filling/08_gapfilled_final/
        am/
        pm/
        gapfill_summary_by_file.csv
        gapfill_overall_summary.csv
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# LOAD SETTINGS
# ============================================================

SETTINGS_PATH = Path(__file__).resolve().parent / "11_gapfilling_setting.py"
spec = importlib.util.spec_from_file_location("gapfill_settings", SETTINGS_PATH)
settings = importlib.util.module_from_spec(spec)

if spec.loader is None:
    raise ImportError(f"Could not load settings file: {SETTINGS_PATH}")

spec.loader.exec_module(settings)


# ============================================================
# PATHS
# ============================================================

ML_PRED_PATH = settings.PREDICTION_DIR / "ml/ml_gapfill_predictions.csv"
INTERP_PRED_PATH = settings.PREDICTION_DIR / "interpolation/interpolation_gapfill_predictions.csv"

SUMMARY_BY_FILE_PATH = settings.FINAL_DIR / "gapfill_summary_by_file.csv"
OVERALL_SUMMARY_PATH = settings.FINAL_DIR / "gapfill_overall_summary.csv"

for pass_name in settings.PASSES:
    (settings.FINAL_DIR / pass_name).mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def parse_date_from_filename(path: Path) -> pd.Timestamp:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        raise ValueError(f"Could not parse YYYYMMDD from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def file_id_from_path(pass_name: str, path: Path) -> str:
    return f"{pass_name}/{path.name}"


def list_complete_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []

    for pass_name in settings.PASSES:
        folder = settings.INPUT_DIR / pass_name / "complete"

        if not folder.exists():
            raise FileNotFoundError(f"Missing input folder: {folder}")

        for path in sorted(folder.glob("*.csv")):
            files.append((pass_name, path))

    if not files:
        raise FileNotFoundError(f"No complete CSV files found under {settings.INPUT_DIR}")

    return files


def add_basic_columns(df: pd.DataFrame, pass_name: str, path: Path) -> pd.DataFrame:
    out = df.loc[:, ~df.columns.duplicated()].copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        out["date"] = parse_date_from_filename(path)

    out["year"] = out["date"].dt.year
    out["pass"] = pass_name
    out["file_id"] = file_id_from_path(pass_name, path)

    if settings.KEY not in out.columns:
        if {"grid_row", "grid_col"}.issubset(out.columns):
            out[settings.KEY] = (
                out["grid_row"].astype(str)
                + "_"
                + out["grid_col"].astype(str)
            )
        else:
            out[settings.KEY] = np.arange(len(out)).astype(str)

    out[settings.KEY] = out[settings.KEY].astype(str)

    return out


def pred_col_name(method: str) -> str:
    return f"pred_{method}"


def read_ml_predictions() -> pd.DataFrame:
    if not ML_PRED_PATH.exists():
        print(f"Warning: ML prediction file not found: {ML_PRED_PATH}")
        return pd.DataFrame(columns=["file_id", settings.KEY])

    print(f"Reading ML predictions: {ML_PRED_PATH}")
    df = pd.read_csv(ML_PRED_PATH)

    needed = {"file_id", settings.KEY, "model", "prediction"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"ML prediction file missing columns: {missing}")

    df = df[list(needed)].copy()
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    wide = (
        df.pivot_table(
            index=["file_id", settings.KEY],
            columns="model",
            values="prediction",
            aggfunc="first",
        )
        .reset_index()
    )

    wide.columns.name = None

    rename = {
        c: pred_col_name(c)
        for c in wide.columns
        if c not in {"file_id", settings.KEY}
    }

    wide = wide.rename(columns=rename)
    return wide


def read_interpolation_predictions() -> pd.DataFrame:
    if not INTERP_PRED_PATH.exists():
        print(f"Warning: interpolation prediction file not found: {INTERP_PRED_PATH}")
        return pd.DataFrame(columns=["file_id", settings.KEY])

    print(f"Reading interpolation predictions: {INTERP_PRED_PATH}")
    df = pd.read_csv(INTERP_PRED_PATH)

    needed = {"file_id", settings.KEY, "method", "prediction"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"Interpolation prediction file missing columns: {missing}")

    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    pred_wide = (
        df.pivot_table(
            index=["file_id", settings.KEY],
            columns="method",
            values="prediction",
            aggfunc="first",
        )
        .reset_index()
    )

    pred_wide.columns.name = None

    pred_rename = {
        c: pred_col_name(c)
        for c in pred_wide.columns
        if c not in {"file_id", settings.KEY}
    }

    pred_wide = pred_wide.rename(columns=pred_rename)

    out = pred_wide

    if "kriging_variance" in df.columns:
        kv = df[df["method"].eq("centroid_ordinary_kriging")].copy()
        if not kv.empty:
            kv["kriging_variance"] = pd.to_numeric(kv["kriging_variance"], errors="coerce")
            kv = kv[["file_id", settings.KEY, "kriging_variance"]].rename(
                columns={"kriging_variance": "kriging_variance_centroid_ordinary_kriging"}
            )
            out = out.merge(kv, on=["file_id", settings.KEY], how="left")

    if "nearest_distance" in df.columns:
        nd = df[df["method"].eq("nearest_neighbor_same_day")].copy()
        if not nd.empty:
            nd["nearest_distance"] = pd.to_numeric(nd["nearest_distance"], errors="coerce")
            nd = nd[["file_id", settings.KEY, "nearest_distance"]].rename(
                columns={"nearest_distance": "nearest_distance_nearest_neighbor_same_day"}
            )
            out = out.merge(nd, on=["file_id", settings.KEY], how="left")

    return out


def choose_fill_for_row(row: pd.Series, candidate_methods: list[str]) -> tuple[float, str, str]:
    original = row.get(settings.TARGET)

    if pd.notna(original):
        return float(original), "observed", "observed"

    for method in candidate_methods:
        col = pred_col_name(method)
        if col in row.index:
            val = row[col]
            if pd.notna(val) and np.isfinite(val):
                return float(val), "filled", method

    return np.nan, "unfilled", "none"


def output_filename(input_path: Path) -> str:
    name = input_path.name
    name = name.replace("_complete_", "_gapfilled_")
    name = name.replace("complete", "gapfilled")
    return name


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("11c: Finalize SMAP gap-filled files")
    print("=" * 80)
    print(f"Project root: {settings.PROJECT_ROOT}")
    print(f"Input folder: {settings.INPUT_DIR}")
    print(f"Final output folder: {settings.FINAL_DIR}")
    print(f"Primary method: {settings.FINAL_PRIMARY_METHOD}")
    print(f"Fallback methods: {settings.FINAL_FALLBACK_METHODS}")
    print("=" * 80)

    candidate_methods = [settings.FINAL_PRIMARY_METHOD] + list(settings.FINAL_FALLBACK_METHODS)
    candidate_methods = list(dict.fromkeys(candidate_methods))

    print("\nFinal fill candidate order:")
    for method in candidate_methods:
        print(f"  - {method}")

    ml_preds = read_ml_predictions()
    interp_preds = read_interpolation_predictions()

    print(f"\nML prediction rows after pivot: {len(ml_preds):,}")
    print(f"Interpolation prediction rows after pivot: {len(interp_preds):,}")

    prediction_tables = []

    if len(interp_preds) > 0:
        prediction_tables.append(interp_preds)

    if len(ml_preds) > 0:
        prediction_tables.append(ml_preds)

    if not prediction_tables:
        raise RuntimeError("No prediction tables found. Run 11a and/or 11b first.")

    preds = prediction_tables[0]

    for table in prediction_tables[1:]:
        preds = preds.merge(table, on=["file_id", settings.KEY], how="outer")

    pred_cols = [c for c in preds.columns if c.startswith("pred_")]

    print("\nPrediction columns available:")
    for c in pred_cols:
        print(f"  - {c}")

    files = list_complete_files()
    summary_rows = []

    for i, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)

        if date.year not in settings.GAPFILL_YEARS:
            continue

        df = pd.read_csv(path, low_memory=False)
        df = add_basic_columns(df, pass_name, path)

        fid = file_id_from_path(pass_name, path)
        psub = preds[preds["file_id"].eq(fid)].copy()

        merged = df.merge(
            psub,
            on=["file_id", settings.KEY],
            how="left",
        )

        fill_values = []
        fill_statuses = []
        fill_methods = []

        for _, row in merged.iterrows():
            val, status, method = choose_fill_for_row(row, candidate_methods)
            fill_values.append(val)
            fill_statuses.append(status)
            fill_methods.append(method)

        merged["soil_moisture_filled"] = fill_values
        merged["fill_status"] = fill_statuses
        merged["fill_method"] = fill_methods

        if settings.CLIP_FILLED_VALUES:
            mask_filled = merged["fill_status"].eq("filled")
            merged.loc[mask_filled, "soil_moisture_filled"] = merged.loc[
                mask_filled, "soil_moisture_filled"
            ].clip(settings.CLIP_MIN, settings.CLIP_MAX)

        n_rows = len(merged)
        n_observed_original = int(merged[settings.TARGET].notna().sum())
        n_missing_original = int(merged[settings.TARGET].isna().sum())
        n_filled = int(merged["fill_status"].eq("filled").sum())
        n_unfilled = int(merged["fill_status"].eq("unfilled").sum())

        method_counts = merged["fill_method"].value_counts(dropna=False).to_dict()

        summary = {
            "file_id": fid,
            "date": date.date().isoformat(),
            "year": date.year,
            "pass": pass_name,
            "source_file": str(path),
            "output_file": "",
            "n_rows": n_rows,
            "n_observed_original": n_observed_original,
            "n_missing_original": n_missing_original,
            "n_filled": n_filled,
            "n_unfilled": n_unfilled,
            "min_soil_moisture_filled": pd.to_numeric(
                merged["soil_moisture_filled"], errors="coerce"
            ).min(),
            "max_soil_moisture_filled": pd.to_numeric(
                merged["soil_moisture_filled"], errors="coerce"
            ).max(),
        }

        for method, count in method_counts.items():
            summary[f"fill_method_count__{method}"] = int(count)

        out_name = output_filename(path)
        out_path = settings.FINAL_DIR / pass_name / out_name
        summary["output_file"] = str(out_path)

        merged.to_csv(out_path, index=False)
        summary_rows.append(summary)

        if i % 100 == 0:
            print(f"  finalized {i:,} files...")

    summary_by_file = pd.DataFrame(summary_rows)
    summary_by_file.to_csv(SUMMARY_BY_FILE_PATH, index=False)

    overall = {
        "n_files": len(summary_by_file),
        "n_rows": int(summary_by_file["n_rows"].sum()),
        "n_observed_original": int(summary_by_file["n_observed_original"].sum()),
        "n_missing_original": int(summary_by_file["n_missing_original"].sum()),
        "n_filled": int(summary_by_file["n_filled"].sum()),
        "n_unfilled": int(summary_by_file["n_unfilled"].sum()),
        "primary_method": settings.FINAL_PRIMARY_METHOD,
        "fallback_methods": ";".join(settings.FINAL_FALLBACK_METHODS),
        "clip_filled_values": settings.CLIP_FILLED_VALUES,
    }

    pd.DataFrame([overall]).to_csv(OVERALL_SUMMARY_PATH, index=False)

    print("\nSaved:")
    print(f"  {SUMMARY_BY_FILE_PATH}")
    print(f"  {OVERALL_SUMMARY_PATH}")
    print(f"  {settings.FINAL_DIR}")

    print("\nOverall summary:")
    for key, value in overall.items():
        print(f"  {key}: {value}")

    print("\nDone.")


if __name__ == "__main__":
    main()