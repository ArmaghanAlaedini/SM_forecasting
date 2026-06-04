#!/usr/bin/env python3
"""
11a_generate_ml_gapfill_predictions.py

Train selected ML models and predict REAL original missing SMAP pixels.

This script:
  - does NOT validate models
  - does NOT finalize filled files
  - only creates ML prediction tables for rows where soil_moisture is NA

Manual controls are in:
    11_gapfilling_setting.py

Outputs:
    src/data/processed/smap_gap_filling/07_gapfill_predictions/ml/
        ml_gapfill_predictions.csv
        ml_gapfill_feature_manifest.csv
        ml_gapfill_run_manifest.csv
"""

from __future__ import annotations

import importlib.util
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# LOAD SETTINGS
# ============================================================

SETTINGS_PATH = Path(__file__).resolve().parent / "11_gapfilling_setting.py"
spec = importlib.util.spec_from_file_location("gapfill_settings", SETTINGS_PATH)
settings = importlib.util.module_from_spec(spec)

if spec.loader is None:
    raise ImportError(f"Could not load settings file: {SETTINGS_PATH}")

spec.loader.exec_module(settings)


try:
    from xgboost import XGBRegressor
    HAVE_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAVE_XGBOOST = False


# ============================================================
# PATHS
# ============================================================

OUT_DIR = settings.PREDICTION_DIR / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = OUT_DIR / "ml_gapfill_predictions.csv"
FEATURE_MANIFEST_PATH = OUT_DIR / "ml_gapfill_feature_manifest.csv"
RUN_MANIFEST_PATH = OUT_DIR / "ml_gapfill_run_manifest.csv"


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
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    out["sin_doy"] = np.sin(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * out["day_of_year"] / 366.0)

    out["pass"] = pass_name
    out["pass_pm"] = 1 if pass_name.lower() == "pm" else 0
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


def read_one_file(path: Path, pass_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return add_basic_columns(df, pass_name, path)


def inspect_available_features(files: list[tuple[str, Path]]) -> tuple[list[str], list[str]]:
    present_any = set()

    for pass_name, path in files:
        header = pd.read_csv(path, nrows=0)
        cols = set(header.columns)

        cols.update([
            "date",
            "year",
            "month",
            "day_of_year",
            "sin_doy",
            "cos_doy",
            "pass",
            "pass_pm",
            "file_id",
        ])

        present_any.update(cols)

    requested = list(settings.ML_FEATURES_TO_USE)
    used = [c for c in requested if c in present_any]
    missing = [c for c in requested if c not in present_any]

    if missing and settings.STRICT_ML_FEATURES:
        raise ValueError(f"Requested ML features missing from all files: {missing}")

    if not used:
        raise ValueError("No usable ML features found.")

    return used, missing


def ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def collect_training_data(
    files: list[tuple[str, Path]],
    feature_cols: list[str],
) -> pd.DataFrame:
    parts = []

    needed = list(dict.fromkeys([
        "file_id",
        settings.KEY,
        "date",
        "year",
        "pass",
        settings.TARGET,
    ] + feature_cols))

    print("\nCollecting ML training rows...")
    print(f"Training years: {settings.ML_TRAIN_YEARS}")

    for i, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)

        if date.year not in settings.ML_TRAIN_YEARS:
            continue

        df = read_one_file(path, pass_name)
        df = ensure_feature_columns(df, feature_cols)

        keep = [c for c in needed if c in df.columns]
        df = df[keep].copy()

        df[settings.TARGET] = pd.to_numeric(df[settings.TARGET], errors="coerce")
        df = df[df[settings.TARGET].notna()].copy()

        if df.empty:
            continue

        parts.append(df)

        if i % 250 == 0:
            print(f"  scanned {i:,} files...")

    if not parts:
        raise RuntimeError("No ML training rows were collected.")

    train = pd.concat(parts, ignore_index=True)
    train = train.loc[:, ~train.columns.duplicated()].copy()

    for c in feature_cols + [settings.TARGET]:
        train[c] = pd.to_numeric(train[c], errors="coerce")

    train = train[train[settings.TARGET].notna()].copy()

    if settings.MAX_ML_TRAIN_ROWS is not None and len(train) > settings.MAX_ML_TRAIN_ROWS:
        print(f"Sampling training rows: {settings.MAX_ML_TRAIN_ROWS:,} from {len(train):,}")
        train = train.sample(
            n=settings.MAX_ML_TRAIN_ROWS,
            random_state=settings.RANDOM_STATE,
        ).reset_index(drop=True)

    print(f"Final ML training rows: {len(train):,}")
    return train


def make_models() -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    requested = set(settings.ML_MODELS_TO_USE)

    if "xgboost" in requested:
        if HAVE_XGBOOST:
            models["xgboost"] = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=500,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            objective="reg:squarederror",
                            tree_method="hist",
                            random_state=settings.RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        else:
            print("Warning: xgboost requested but not installed. Skipping xgboost.")

    if "hist_gbdt" in requested:
        models["hist_gbdt"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=500,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        l2_regularization=0.01,
                        random_state=settings.RANDOM_STATE,
                    ),
                ),
            ]
        )

    if "random_forest" in requested:
        models["random_forest"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        max_features="sqrt",
                        min_samples_leaf=2,
                        random_state=settings.RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    unknown = requested - {"xgboost", "hist_gbdt", "random_forest"}

    if unknown:
        print(f"Warning: unknown ML models requested and skipped: {sorted(unknown)}")

    if not models:
        raise RuntimeError("No ML models available/requested.")

    return models


def initialize_output_file() -> None:
    if PRED_PATH.exists():
        PRED_PATH.unlink()

    header = pd.DataFrame(
        columns=[
            "file_id",
            "date",
            "year",
            "pass",
            settings.KEY,
            "model",
            "feature_group",
            "n_features",
            "features",
            "prediction",
            "source_file",
        ]
    )

    header.to_csv(PRED_PATH, index=False)


def append_predictions(df: pd.DataFrame) -> None:
    out_cols = [
        "file_id",
        "date",
        "year",
        "pass",
        settings.KEY,
        "model",
        "feature_group",
        "n_features",
        "features",
        "prediction",
        "source_file",
    ]

    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    df[out_cols].to_csv(PRED_PATH, mode="a", header=False, index=False)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    print("11a: Generate ML predictions for real SMAP gaps")
    print("=" * 80)
    print(f"Project root: {settings.PROJECT_ROOT}")
    print(f"Input folder: {settings.INPUT_DIR}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Models requested: {settings.ML_MODELS_TO_USE}")
    print(f"Gapfill years: {settings.GAPFILL_YEARS}")
    print("=" * 80)

    files = list_complete_files()
    feature_cols, missing_features = inspect_available_features(files)

    print("\nUsed ML features:")
    for c in feature_cols:
        print(f"  - {c}")

    if missing_features:
        print("\nMissing requested features skipped:")
        for c in missing_features:
            print(f"  - {c}")

    feature_manifest = pd.DataFrame(
        {
            "feature": settings.ML_FEATURES_TO_USE,
            "used": [c in feature_cols for c in settings.ML_FEATURES_TO_USE],
            "feature_group": settings.ML_FEATURE_GROUP_NAME,
        }
    )
    feature_manifest.to_csv(FEATURE_MANIFEST_PATH, index=False)

    train = collect_training_data(files, feature_cols)
    X_train = train[feature_cols]
    y_train = pd.to_numeric(train[settings.TARGET], errors="coerce").to_numpy()

    models = make_models()

    trained_models = {}

    for model_name, model in models.items():
        print("\n" + "=" * 80)
        print(f"Training ML model: {model_name}")
        print("=" * 80)

        model.fit(X_train, y_train)
        trained_models[model_name] = model

    initialize_output_file()

    manifest_rows = []
    total_missing_rows = 0
    total_prediction_rows = 0

    print("\nPredicting real missing rows...")

    for i, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)

        if date.year not in settings.GAPFILL_YEARS:
            continue

        df = read_one_file(path, pass_name)
        df = ensure_feature_columns(df, feature_cols)

        df[settings.TARGET] = pd.to_numeric(df[settings.TARGET], errors="coerce")

        missing = df[df[settings.TARGET].isna()].copy()
        n_missing = len(missing)

        if n_missing == 0:
            manifest_rows.append(
                {
                    "file_id": file_id_from_path(pass_name, path),
                    "date": date.date().isoformat(),
                    "year": date.year,
                    "pass": pass_name,
                    "source_file": str(path),
                    "n_rows": len(df),
                    "n_missing_target": 0,
                    "n_models": len(trained_models),
                    "n_prediction_rows": 0,
                }
            )
            continue

        base = missing[["file_id", "date", "year", "pass", settings.KEY]].copy()
        base["source_file"] = str(path)

        for model_name, model in trained_models.items():
            pred = np.asarray(model.predict(missing[feature_cols]), dtype=float)

            out = base.copy()
            out["model"] = model_name
            out["feature_group"] = settings.ML_FEATURE_GROUP_NAME
            out["n_features"] = len(feature_cols)
            out["features"] = ";".join(feature_cols)
            out["prediction"] = pred

            append_predictions(out)
            total_prediction_rows += len(out)

        total_missing_rows += n_missing

        manifest_rows.append(
            {
                "file_id": file_id_from_path(pass_name, path),
                "date": date.date().isoformat(),
                "year": date.year,
                "pass": pass_name,
                "source_file": str(path),
                "n_rows": len(df),
                "n_missing_target": n_missing,
                "n_models": len(trained_models),
                "n_prediction_rows": n_missing * len(trained_models),
            }
        )

        if i % 100 == 0:
            print(f"  scanned {i:,} files; missing rows so far: {total_missing_rows:,}")

    run_manifest = pd.DataFrame(manifest_rows)
    run_manifest.to_csv(RUN_MANIFEST_PATH, index=False)

    print("\nSaved:")
    print(f"  {PRED_PATH}")
    print(f"  {FEATURE_MANIFEST_PATH}")
    print(f"  {RUN_MANIFEST_PATH}")

    print("\nSummary:")
    print(f"  Files scanned: {len(files):,}")
    print(f"  Real missing rows found: {total_missing_rows:,}")
    print(f"  ML prediction rows written: {total_prediction_rows:,}")
    print(f"  Models used: {list(trained_models.keys())}")
    print(f"  Features used: {len(feature_cols)}")

    print("\nDone.")


if __name__ == "__main__":
    main()