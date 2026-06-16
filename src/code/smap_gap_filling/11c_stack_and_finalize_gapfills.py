#!/usr/bin/env python3
"""
11c_stack_and_finalize_gapfills.py
-- updated to include pred_regression_kriging in META_FEATURE_COLS --
"""

from __future__ import annotations
import importlib.util
import re
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

SETTINGS_PATH = Path(__file__).resolve().parent / "11_gapfilling_setting.py"
spec = importlib.util.spec_from_file_location("gapfill_settings", SETTINGS_PATH)
settings = importlib.util.module_from_spec(spec)
if spec.loader is None:
    raise ImportError(f"Could not load settings file: {SETTINGS_PATH}")
spec.loader.exec_module(settings)


# ============================================================
# PATHS
# ============================================================

ML_PRED_PATH     = settings.PREDICTION_DIR / "ml/ml_gapfill_predictions.csv"
INTERP_PRED_PATH = settings.PREDICTION_DIR / "interpolation/interpolation_gapfill_predictions.csv"
SUMMARY_BY_FILE_PATH = settings.FINAL_DIR / "gapfill_summary_by_file.csv"
OVERALL_SUMMARY_PATH = settings.FINAL_DIR / "gapfill_overall_summary.csv"

for pass_name in settings.PASSES:
    (settings.FINAL_DIR / pass_name).mkdir(parents=True, exist_ok=True)


# ============================================================
# STACKING META-MODEL
# ============================================================

def load_meta_model():
    path = getattr(settings, "META_MODEL_PATH", None)
    if path is None:
        print("META_MODEL_PATH not set — using waterfall fill only.")
        return None
    path = Path(path)
    if not path.exists():
        print(f"Warning: meta-model not found at {path}. Using waterfall.")
        return None
    try:
        import joblib
        model = joblib.load(path)
        print(f"Stacking meta-model loaded: {path}")
        return model
    except Exception as exc:
        print(f"Warning: could not load meta-model ({exc}). Using waterfall.")
        return None


META_MODEL = load_meta_model()

# ← regression_kriging added here
META_FEATURE_COLS = [
    "pred_centroid_ordinary_kriging",
    "pred_nearest_neighbor_same_day",
    "pred_regression_kriging",
    "pred_xgboost",
    "pred_hist_gbdt",
    "pred_random_forest",
    "x", "y", "sin_doy", "cos_doy", "pass_pm",
]


# ============================================================
# HELPERS
# ============================================================

def parse_date_from_filename(path: Path) -> pd.Timestamp:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        raise ValueError(f"Could not parse YYYYMMDD from: {path}")
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
        raise FileNotFoundError(f"No complete CSV files under {settings.INPUT_DIR}")
    return files


def add_basic_columns(df: pd.DataFrame, pass_name: str, path: Path) -> pd.DataFrame:
    out = df.loc[:, ~df.columns.duplicated()].copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    else:
        out["date"] = parse_date_from_filename(path)
    out["year"]    = out["date"].dt.year
    out["pass"]    = pass_name
    out["file_id"] = file_id_from_path(pass_name, path)

    doy = out["date"].dt.dayofyear.fillna(1).astype(float)
    out["sin_doy"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * doy / 366.0)
    out["pass_pm"] = (pass_name.lower() == "pm")

    if settings.KEY not in out.columns:
        if {"grid_row", "grid_col"}.issubset(out.columns):
            out[settings.KEY] = out["grid_row"].astype(str) + "_" + out["grid_col"].astype(str)
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
    if needed - set(df.columns):
        raise ValueError(f"ML prediction file missing: {needed - set(df.columns)}")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    wide = df.pivot_table(index=["file_id", settings.KEY], columns="model", values="prediction", aggfunc="first").reset_index()
    wide.columns.name = None
    return wide.rename(columns={c: pred_col_name(c) for c in wide.columns if c not in {"file_id", settings.KEY}})


def read_interpolation_predictions() -> pd.DataFrame:
    if not INTERP_PRED_PATH.exists():
        print(f"Warning: interpolation prediction file not found: {INTERP_PRED_PATH}")
        return pd.DataFrame(columns=["file_id", settings.KEY])
    print(f"Reading interpolation predictions: {INTERP_PRED_PATH}")
    df = pd.read_csv(INTERP_PRED_PATH)
    needed = {"file_id", settings.KEY, "method", "prediction"}
    if needed - set(df.columns):
        raise ValueError(f"Interpolation file missing: {needed - set(df.columns)}")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    wide = df.pivot_table(index=["file_id", settings.KEY], columns="method", values="prediction", aggfunc="first").reset_index()
    wide.columns.name = None
    out = wide.rename(columns={c: pred_col_name(c) for c in wide.columns if c not in {"file_id", settings.KEY}})

    for method, diag_col, rename_col in [
        ("centroid_ordinary_kriging", "kriging_variance", "kriging_variance_centroid_ok"),
        ("nearest_neighbor_same_day", "nearest_distance", "nearest_distance_nn"),
    ]:
        col = diag_col
        if col in df.columns:
            sub = df[df["method"].eq(method)][["file_id", settings.KEY, col]].rename(columns={col: rename_col})
            out = out.merge(sub, on=["file_id", settings.KEY], how="left")
    return out


def apply_waterfall(row: pd.Series, candidate_methods: list[str]) -> tuple[float, str]:
    for method in candidate_methods:
        col = pred_col_name(method)
        if col in row.index:
            val = row[col]
            if pd.notna(val) and np.isfinite(float(val)):
                return float(val), method
    return np.nan, "none"


def apply_stacking_to_missing(missing_df, meta_model, waterfall_methods):
    n = len(missing_df)
    fill_values   = np.full(n, np.nan)
    fill_methods  = ["none"] * n
    fill_statuses = ["unfilled"] * n

    if meta_model is not None:
        X_meta = missing_df.reindex(columns=META_FEATURE_COLS).copy()
        for c in META_FEATURE_COLS:
            if c not in X_meta.columns:
                X_meta[c] = np.nan
        X_meta = X_meta[META_FEATURE_COLS].to_numpy(dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meta_preds = np.asarray(meta_model.predict(X_meta), dtype=float)

        valid = np.isfinite(meta_preds)
        fill_values[valid]   = meta_preds[valid]
        for idx in np.where(valid)[0]:
            fill_methods[idx]  = "stacking"
            fill_statuses[idx] = "filled"

    needs_fallback  = ~np.isfinite(fill_values)
    if needs_fallback.any():
        fallback_rows = missing_df[needs_fallback].reset_index(drop=True)
        global_idxs   = np.where(needs_fallback)[0]
        for local_i, (_, row) in enumerate(fallback_rows.iterrows()):
            val, method = apply_waterfall(row, waterfall_methods)
            gi = global_idxs[local_i]
            if np.isfinite(val):
                fill_values[gi]   = val
                fill_methods[gi]  = method
                fill_statuses[gi] = "filled"

    return fill_values, fill_statuses, fill_methods


def output_filename(input_path: Path) -> str:
    name = input_path.name.replace("_complete_", "_gapfilled_").replace("complete", "gapfilled")
    return name


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("11c: Finalize SMAP gap-filled files (with stacking + regression_kriging)")
    print("=" * 80)
    print(f"Stacking model: {'enabled' if META_MODEL is not None else 'disabled'}")
    print(f"Waterfall order: {[settings.FINAL_PRIMARY_METHOD] + list(settings.FINAL_FALLBACK_METHODS)}")
    print("=" * 80)

    waterfall_methods = list(dict.fromkeys(
        [settings.FINAL_PRIMARY_METHOD] + list(settings.FINAL_FALLBACK_METHODS)
    ))
    print("\nWaterfall fallback order:")
    for m in waterfall_methods:
        print(f"  - {m}")

    ml_preds    = read_ml_predictions()
    interp_preds = read_interpolation_predictions()
    print(f"\nML rows:            {len(ml_preds):,}")
    print(f"Interpolation rows: {len(interp_preds):,}")

    preds = interp_preds if len(interp_preds) > 0 else pd.DataFrame(columns=["file_id", settings.KEY])
    if len(ml_preds) > 0:
        preds = preds.merge(ml_preds, on=["file_id", settings.KEY], how="outer")

    pred_cols = [c for c in preds.columns if c.startswith("pred_")]
    print(f"\nPrediction columns: {pred_cols}")

    files = list_complete_files()
    summary_rows = []

    for i, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)
        if date.year not in settings.GAPFILL_YEARS:
            continue

        df     = pd.read_csv(path, low_memory=False)
        df     = add_basic_columns(df, pass_name, path)
        fid    = file_id_from_path(pass_name, path)
        psub   = preds[preds["file_id"].eq(fid)].copy()
        merged = df.merge(psub, on=["file_id", settings.KEY], how="left")

        is_missing     = merged[settings.TARGET].isna()
        observed_mask  = ~is_missing

        fill_values    = np.where(observed_mask, pd.to_numeric(merged[settings.TARGET], errors="coerce"), np.nan)
        fill_statuses  = np.where(observed_mask, "observed", "unfilled").tolist()
        fill_methods   = np.where(observed_mask, "observed", "none").tolist()

        missing_rows = merged[is_missing].copy()
        if len(missing_rows) > 0:
            fv, fs, fm = apply_stacking_to_missing(missing_rows, META_MODEL, waterfall_methods)
            for local_i, global_i in enumerate(np.where(is_missing)[0]):
                fill_values[global_i]   = fv[local_i]
                fill_statuses[global_i] = fs[local_i]
                fill_methods[global_i]  = fm[local_i]

        merged["soil_moisture_filled"] = fill_values
        merged["fill_status"]          = fill_statuses
        merged["fill_method"]          = fill_methods

        if settings.CLIP_FILLED_VALUES:
            clip_mask = merged["fill_status"].eq("filled")
            merged.loc[clip_mask, "soil_moisture_filled"] = (
                merged.loc[clip_mask, "soil_moisture_filled"].clip(settings.CLIP_MIN, settings.CLIP_MAX)
            )

        n_rows    = len(merged)
        n_obs     = int(observed_mask.sum())
        n_miss    = int(is_missing.sum())
        n_filled  = int((merged["fill_status"] == "filled").sum())
        n_unfill  = int((merged["fill_status"] == "unfilled").sum())
        mc        = merged["fill_method"].value_counts(dropna=False).to_dict()

        out_name = output_filename(path)
        out_path = settings.FINAL_DIR / pass_name / out_name
        merged.to_csv(out_path, index=False)

        summary = {
            "file_id": fid, "date": date.date().isoformat(), "year": date.year,
            "pass": pass_name, "source_file": str(path), "output_file": str(out_path),
            "n_rows": n_rows, "n_observed_original": n_obs, "n_missing_original": n_miss,
            "n_filled": n_filled, "n_unfilled": n_unfill,
            "stacking_used": META_MODEL is not None,
            "min_soil_moisture_filled": pd.to_numeric(merged["soil_moisture_filled"], errors="coerce").min(),
            "max_soil_moisture_filled": pd.to_numeric(merged["soil_moisture_filled"], errors="coerce").max(),
        }
        for method, count in mc.items():
            summary[f"fill_method_count__{method}"] = int(count)
        summary_rows.append(summary)

        if i % 100 == 0 or i == 1:
            stacking_count = mc.get("stacking", 0)
            print(
                f"  [{i}] {date.date()} {pass_name.upper()} | "
                f"obs={n_obs} miss={n_miss} filled={n_filled} "
                f"(stacking={stacking_count}) unfilled={n_unfill}"
            )

    summary_by_file = pd.DataFrame(summary_rows)
    summary_by_file.to_csv(SUMMARY_BY_FILE_PATH, index=False)

    overall = {
        "n_files": len(summary_by_file),
        "n_rows": int(summary_by_file["n_rows"].sum()),
        "n_observed_original": int(summary_by_file["n_observed_original"].sum()),
        "n_missing_original": int(summary_by_file["n_missing_original"].sum()),
        "n_filled": int(summary_by_file["n_filled"].sum()),
        "n_unfilled": int(summary_by_file["n_unfilled"].sum()),
        "stacking_enabled": META_MODEL is not None,
        "primary_method": settings.FINAL_PRIMARY_METHOD,
        "fallback_methods": ";".join(settings.FINAL_FALLBACK_METHODS),
        "clip_filled_values": settings.CLIP_FILLED_VALUES,
    }
    pd.DataFrame([overall]).to_csv(OVERALL_SUMMARY_PATH, index=False)

    print("\nSaved:", SUMMARY_BY_FILE_PATH)
    print("Saved:", OVERALL_SUMMARY_PATH)
    print("\nOverall:")
    for k, v in overall.items():
        print(f"  {k}: {v}")

    if "fill_method_count__stacking" in summary_by_file.columns:
        print(f"\n  Total pixels by stacking: {int(summary_by_file['fill_method_count__stacking'].sum()):,}")

    print("\nDone.")


if __name__ == "__main__":
    main()