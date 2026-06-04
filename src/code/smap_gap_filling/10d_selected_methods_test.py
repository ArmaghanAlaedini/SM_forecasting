#!/usr/bin/env python3
"""
10d_selected_methods_test.py

Selected-method ML test on 2025 artificial SMAP gaps.

This script does NOT select models. It only tests the ML models selected from
2024 validation / 10c:

    xgboost
    hist_gbdt
    random_forest

Feature group:

    all_iem_spatiotemporal

Design:
    Train: 2020-2023 observed SMAP
    Test:  2025 observed SMAP with artificial holdouts

Outputs:
    src/data/processed/smap_gap_filling/06_selected_methods_test/ml/
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling/03_full_smap_iem_data"
)

OUT_DIR = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling/06_selected_methods_test/ml"
)
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "soil_moisture"
KEY = "smap_pixel_key"

PASSES = ["am", "pm"]

TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS = [2025]

# Set to None for full training data.
# Keep this number if your laptop is slow.
MAX_TRAIN_ROWS = 250_000

# Set to None for full 2025 artificial test rows.
MAX_EVAL_ROWS_PER_HOLDOUT = None

RANDOM_STATE = 42

RANDOM_CELL_HOLDOUT_FRACTION = 0.25
SPATIAL_BLOCK_N_BINS = 3
MIN_HOLDOUT_ROWS = 10
MIN_OBSERVED_ROWS_PER_FILE = 30

FEATURE_GROUP_NAME = "all_iem_spatiotemporal"

FEATURES = [
    "precip_pta",
    "rh_pta",
    "speed_pta",
    "gust_pta",
    "et_pta",
    "soil04tn_pta",
    "soil04t_pta",
    "soil04tx_pta",
    "soil112tn_pta",
    "soil112t_pta",
    "soil112tx_pta",
    "soil112wc_pta",
    "soil24tn_pta",
    "soil24t_pta",
    "soil24tx_pta",
    "soil24wc_pta",
    "soil50tn_pta",
    "soil50t_pta",
    "soil50tx_pta",
    "soil50wc_pta",
    "x",
    "y",
    "sin_doy",
    "cos_doy",
    "pass_pm",
]


try:
    from xgboost import XGBRegressor

    HAVE_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAVE_XGBOOST = False


# ============================================================
# HELPERS
# ============================================================

def parse_date_from_filename(path: Path) -> pd.Timestamp:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        raise ValueError(f"Could not parse YYYYMMDD from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def list_complete_files() -> list[tuple[str, Path]]:
    files: list[tuple[str, Path]] = []

    for pass_name in PASSES:
        folder = INPUT_DIR / pass_name / "complete"
        if not folder.exists():
            raise FileNotFoundError(f"Missing input folder: {folder}")

        for path in sorted(folder.glob("*.csv")):
            files.append((pass_name, path))

    if not files:
        raise FileNotFoundError(f"No complete CSV files found under {INPUT_DIR}")

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
    out["pass_pm"] = 1 if pass_name == "pm" else 0

    if KEY not in out.columns:
        if {"grid_row", "grid_col"}.issubset(out.columns):
            out[KEY] = out["grid_row"].astype(str) + "_" + out["grid_col"].astype(str)
        else:
            out[KEY] = np.arange(len(out)).astype(str)

    out[KEY] = out[KEY].astype(str)

    return out


def read_one_file(path: Path, pass_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_basic_columns(df, pass_name, path)


def choose_feature_columns(files: list[tuple[str, Path]]) -> list[str]:
    available = set()

    # Inspect several files from both passes.
    for pass_name, path in files[: min(20, len(files))]:
        df = read_one_file(path, pass_name)
        available.update(df.columns)

    feature_cols = [c for c in FEATURES if c in available]
    missing = [c for c in FEATURES if c not in available]

    if missing:
        print("\nWarning: these requested features were not found and will be skipped:")
        print(missing)

    if len(feature_cols) < 5:
        raise ValueError(f"Too few feature columns found: {feature_cols}")

    return feature_cols


def collect_training_data(
    files: list[tuple[str, Path]],
    feature_cols: list[str],
) -> pd.DataFrame:
    parts = []

    needed = list(dict.fromkeys([KEY, "date", "year", "pass", TARGET] + feature_cols))

    print("\nCollecting training rows from 2020-2023 observed SMAP...")

    for i, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)

        if date.year not in TRAIN_YEARS:
            continue

        df = read_one_file(path, pass_name)
        df = df[[c for c in needed if c in df.columns]].copy()
        df = df[df[TARGET].notna()].copy()

        if df.empty:
            continue

        parts.append(df)

        if i % 250 == 0:
            print(f"  scanned {i:,} files...")

    if not parts:
        raise RuntimeError("No training rows found.")

    train = pd.concat(parts, ignore_index=True)
    train = train.loc[:, ~train.columns.duplicated()].copy()

    for c in feature_cols + [TARGET]:
        train[c] = pd.to_numeric(train[c], errors="coerce")

    train = train[train[TARGET].notna()].copy()

    if MAX_TRAIN_ROWS is not None and len(train) > MAX_TRAIN_ROWS:
        train = train.sample(n=MAX_TRAIN_ROWS, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"Training rows: {len(train):,}")
    return train


def make_random_cell_holdout(obs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    n = len(obs)

    if n < MIN_OBSERVED_ROWS_PER_FILE:
        return np.array([], dtype=int)

    k = int(round(RANDOM_CELL_HOLDOUT_FRACTION * n))
    k = max(MIN_HOLDOUT_ROWS, k)
    k = min(k, n - 1)

    return rng.choice(obs.index.to_numpy(), size=k, replace=False)


def make_spatial_block_holdout(obs: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    n = len(obs)

    if n < MIN_OBSERVED_ROWS_PER_FILE:
        return np.array([], dtype=int)

    work = obs.copy()

    if {"grid_row", "grid_col"}.issubset(work.columns):
        row_var = pd.to_numeric(work["grid_row"], errors="coerce")
        col_var = pd.to_numeric(work["grid_col"], errors="coerce")
    elif {"y", "x"}.issubset(work.columns):
        row_var = pd.to_numeric(work["y"], errors="coerce")
        col_var = pd.to_numeric(work["x"], errors="coerce")
    else:
        return make_random_cell_holdout(obs, rng)

    valid = row_var.notna() & col_var.notna()
    work = work.loc[valid].copy()
    row_var = row_var.loc[valid]
    col_var = col_var.loc[valid]

    if len(work) < MIN_OBSERVED_ROWS_PER_FILE:
        return make_random_cell_holdout(obs, rng)

    try:
        work["_row_bin"] = pd.qcut(
            row_var.rank(method="first"),
            q=SPATIAL_BLOCK_N_BINS,
            labels=False,
            duplicates="drop",
        )
        work["_col_bin"] = pd.qcut(
            col_var.rank(method="first"),
            q=SPATIAL_BLOCK_N_BINS,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return make_random_cell_holdout(obs, rng)

    candidates = []
    for _, sub in work.groupby(["_row_bin", "_col_bin"], dropna=True):
        if MIN_HOLDOUT_ROWS <= len(sub) < n:
            candidates.append(sub.index.to_numpy())

    if not candidates:
        return make_random_cell_holdout(obs, rng)

    chosen = candidates[int(rng.integers(0, len(candidates)))]
    return chosen.astype(int)


def collect_test_data(
    files: list[tuple[str, Path]],
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    manifest_rows = []

    needed = list(
        dict.fromkeys(
            [
                KEY,
                "date",
                "year",
                "month",
                "day_of_year",
                "pass",
                TARGET,
                "grid_row",
                "grid_col",
                "x",
                "y",
                "lon",
                "lat",
            ]
            + feature_cols
        )
    )

    print("\nCollecting 2025 artificial test holdouts...")

    for i, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)

        if date.year not in TEST_YEARS:
            continue

        df = read_one_file(path, pass_name)
        df = df[[c for c in needed if c in df.columns]].copy()

        obs = df[df[TARGET].notna()].copy()
        n_obs = len(obs)

        if n_obs < MIN_OBSERVED_ROWS_PER_FILE:
            continue

        seed_base = int(date.strftime("%Y%m%d")) + (1 if pass_name == "pm" else 0)

        for holdout_mode in ["random_cell", "spatial_block"]:
            rng = np.random.default_rng(
                seed_base + RANDOM_STATE + (10_000 if holdout_mode == "spatial_block" else 0)
            )

            if holdout_mode == "random_cell":
                hidden_idx = make_random_cell_holdout(obs, rng)
            else:
                hidden_idx = make_spatial_block_holdout(obs, rng)

            if len(hidden_idx) == 0:
                continue

            hidden = obs.loc[hidden_idx].copy()
            hidden["split"] = "test"
            hidden["holdout_mode"] = holdout_mode
            hidden["source_file"] = str(path)

            rows.append(hidden)

            manifest_rows.append(
                {
                    "date": date.date().isoformat(),
                    "pass": pass_name,
                    "source_file": str(path),
                    "split": "test",
                    "holdout_mode": holdout_mode,
                    "n_rows_file": len(df),
                    "n_observed_file": n_obs,
                    "n_hidden_test": len(hidden),
                }
            )

        if i % 250 == 0:
            print(f"  scanned {i:,} files...")

    if not rows:
        raise RuntimeError("No 2025 artificial test rows found.")

    test = pd.concat(rows, ignore_index=True)
    test = test.loc[:, ~test.columns.duplicated()].copy()

    for c in feature_cols + [TARGET]:
        test[c] = pd.to_numeric(test[c], errors="coerce")

    if MAX_EVAL_ROWS_PER_HOLDOUT is not None:
        sampled = []
        for holdout_mode, sub in test.groupby("holdout_mode"):
            if len(sub) > MAX_EVAL_ROWS_PER_HOLDOUT:
                sub = sub.sample(n=MAX_EVAL_ROWS_PER_HOLDOUT, random_state=RANDOM_STATE)
            sampled.append(sub)
        test = pd.concat(sampled, ignore_index=True)

    manifest = pd.DataFrame(manifest_rows)

    print(f"Test rows: {len(test):,}")
    print(test["holdout_mode"].value_counts().to_string())

    return test, manifest


def make_models() -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}

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
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        print("\nWarning: xgboost is not installed. Skipping xgboost.")

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
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    models["random_forest"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return models


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "n": 0}

    yt = y_true[mask]
    yp = y_pred[mask]

    return {
        "rmse": math.sqrt(mean_squared_error(yt, yp)),
        "mae": mean_absolute_error(yt, yp),
        "bias": float(np.mean(yp - yt)),
        "r2": r2_score(yt, yp) if len(yt) >= 2 else np.nan,
        "n": int(len(yt)),
    }


def plot_metric(metrics: pd.DataFrame, metric: str, out_path: Path) -> None:
    sub = metrics.sort_values(["holdout_mode", metric]).copy()

    if sub.empty:
        return

    sub["label"] = sub["model"] + " | " + sub["holdout_mode"]

    plt.figure(figsize=(10, max(4, 0.35 * len(sub))))
    plt.barh(sub["label"][::-1], sub[metric][::-1])
    plt.xlabel(metric.upper())
    plt.title(f"Selected ML 2025 test: {metric.upper()}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    print("Selected ML 2025 test")
    print("=" * 70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Output folder: {OUT_DIR}")
    print(f"Train years: {TRAIN_YEARS}")
    print(f"Test years: {TEST_YEARS}")
    print(f"MAX_TRAIN_ROWS: {MAX_TRAIN_ROWS}")
    print(f"MAX_EVAL_ROWS_PER_HOLDOUT: {MAX_EVAL_ROWS_PER_HOLDOUT}")
    print("=" * 70)

    files = list_complete_files()
    feature_cols = choose_feature_columns(files)

    print("\nFeatures used:")
    for c in feature_cols:
        print(f"  - {c}")

    train = collect_training_data(files, feature_cols)
    test, manifest = collect_test_data(files, feature_cols)

    X_train = train[feature_cols]
    y_train = pd.to_numeric(train[TARGET], errors="coerce").to_numpy()

    models = make_models()

    if not models:
        raise RuntimeError("No ML models available.")

    pred_parts = []
    metric_rows = []

    base_cols = [
        "split",
        "holdout_mode",
        "date",
        "pass",
        KEY,
        TARGET,
        "source_file",
    ]
    base_cols = [c for c in base_cols if c in test.columns]

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"Training: {model_name}")
        print("=" * 70)

        model.fit(X_train, y_train)

        print(f"Predicting 2025 artificial gaps: {model_name}")
        y_pred = np.asarray(model.predict(test[feature_cols]), dtype=float)

        out = test[base_cols].copy()
        out["model"] = model_name
        out["feature_group"] = FEATURE_GROUP_NAME
        out["n_features"] = len(feature_cols)
        out["features"] = ";".join(feature_cols)
        out["observed"] = pd.to_numeric(test[TARGET], errors="coerce").to_numpy()
        out["prediction"] = y_pred

        pred_parts.append(out)

        for holdout_mode, sub in out.groupby("holdout_mode"):
            m = metric_dict(
                sub["observed"].to_numpy(dtype=float),
                sub["prediction"].to_numpy(dtype=float),
            )

            metric_rows.append(
                {
                    "split": "test",
                    "holdout_mode": holdout_mode,
                    "feature_group": FEATURE_GROUP_NAME,
                    "model": model_name,
                    "n_features": len(feature_cols),
                    "features": ";".join(feature_cols),
                    **m,
                }
            )

    predictions = pd.concat(pred_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["holdout_mode", "rmse", "mae"])

    metrics_path = OUT_DIR / "ml_selected_test_metrics.csv"
    preds_path = OUT_DIR / "ml_selected_test_predictions.csv"
    manifest_path = OUT_DIR / "ml_selected_test_manifest.csv"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(preds_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    plot_metric(metrics, "rmse", FIG_DIR / "ml_selected_test_rmse.pdf")
    plot_metric(metrics, "bias", FIG_DIR / "ml_selected_test_bias.pdf")

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {preds_path}")
    print(f"  {manifest_path}")
    print(f"  {FIG_DIR}")

    print("\nSelected ML 2025 test results:")
    print(metrics.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()