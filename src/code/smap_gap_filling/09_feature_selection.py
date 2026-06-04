#!/usr/bin/env python3
"""
09_feature_selection.py

Feature screening for SMAP soil-moisture gap filling.

Purpose
-------
This script screens candidate predictors for filling NA values in the
completed daily SMAP + IEM files.

It uses only rows where soil_moisture is observed for supervised screening.

Core idea
---------
Observed rows:
    soil_moisture is known
    -> used for correlations, feature-group testing, and model-based importance

Missing rows:
    soil_moisture is NA
    -> not used for training/evaluation here, but predictor coverage is audited

Recommended split
-----------------
Train:      2020-2023
Validation: 2024
Test:       2025

Important:
The script uses train + validation for feature screening. It does not use
2025 test performance for feature selection by default.

Outputs
-------
src/data/processed/smap_gap_filling/04_feature_screening/

    dataset_audit_by_split.csv
    feature_missingness_by_split.csv
    feature_correlations_train.csv
    feature_group_model_scores_validation.csv
    model_builtin_importance.csv
    model_permutation_importance.csv
    feature_recommendation_table.csv
    top_cokriging_candidates.csv

    figures/
        top_feature_correlations.pdf
        feature_group_rmse.pdf
        top_permutation_importance.pdf
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import importlib.util
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. USER SETTINGS
# ============================================================

TARGET = "soil_moisture"

PASSES_TO_USE = ["am", "pm"]   # ["am"], ["pm"], or ["am", "pm"]

TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALIDATION_YEARS = [2024]
TEST_YEARS = [2025]

# Set to an integer for a quick test, e.g. 20.
# Set to None for full run.
MAX_FILES_PER_PASS = None
MAX_FILES_PER_SPLIT_PER_PASS = None

# For model screening, do not train on all millions of rows at first.
# These samples are enough for feature screening and much faster.
MAX_OBSERVED_ROWS_PER_FILE_FOR_STATS = 350
MAX_TRAIN_ROWS_FOR_MODELS = 250_000
MAX_VALID_ROWS_FOR_MODELS = 120_000

# Permutation importance can be expensive.
MAX_VALID_ROWS_FOR_PERMUTATION = 50_000
N_PERMUTATION_REPEATS = 5

RANDOM_STATE = 42

# Include only actual PTA values by default.
# Do not include *_pta_var or *_n_samples unless you intentionally want
# uncertainty/support variables as predictors.
INCLUDE_PTA_VALUE_COLUMNS = True
INCLUDE_PTA_VAR_COLUMNS = False
INCLUDE_N_SAMPLE_COLUMNS = False

# Include spatial and temporal predictors.
INCLUDE_SPATIAL_COLUMNS = True
INCLUDE_TEMPORAL_COLUMNS = True
INCLUDE_PASS_INDICATOR = True

# Optional XGBoost. If xgboost is installed, this script can include it.
INCLUDE_XGBOOST_IF_AVAILABLE = True

# By default, do not evaluate on 2025 in this screening script.
# Keep test year untouched for final model assessment.
EVALUATE_TEST_FOR_AUDIT_ONLY = False

# Robust plot limits / ranking behavior
TOP_N_TO_PLOT = 25

# Model sizes for screening. Increase later if needed.
N_TREES = 300


# ============================================================
# 1. LOAD CONFIG
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
# 2. PATHS
# ============================================================

def get_gap_filling_dir() -> Path:
    if hasattr(cfg, "GAP_FILLING_DIR"):
        return Path(cfg.GAP_FILLING_DIR)
    return Path(cfg.PROCESSED_DIR) / "smap_gap_filling"


def get_full_smap_iem_dir() -> Path:
    if hasattr(cfg, "FULL_SMAP_IEM_DIR"):
        return Path(cfg.FULL_SMAP_IEM_DIR)
    return get_gap_filling_dir() / "03_full_smap_iem_data"


FULL_DIR = get_full_smap_iem_dir()
OUT_DIR = get_gap_filling_dir() / "04_feature_screening"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. MANIFEST
# ============================================================

def parse_date_from_complete_file(path: Path) -> str:
    match = re.search(r"(20\d{6})", path.name)
    if match is None:
        raise ValueError(f"Could not parse date from file name: {path.name}")

    return pd.to_datetime(match.group(1), format="%Y%m%d").strftime("%Y-%m-%d")


def year_to_split(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VALIDATION_YEARS:
        return "validation"
    if year in TEST_YEARS:
        return "test"
    return "unused"


def build_manifest() -> pd.DataFrame:
    rows = []

    for pass_name in PASSES_TO_USE:
        pass_name = pass_name.lower()
        folder = FULL_DIR / pass_name / "complete"

        files = sorted(folder.glob(f"smap_iem_{pass_name}_complete_*.csv"))

        for path in files:
            date = parse_date_from_complete_file(path)
            year = pd.to_datetime(date).year

            rows.append({
                "path": str(path),
                "file_name": path.name,
                "pass": pass_name,
                "date": date,
                "year": year,
                "split": year_to_split(year),
            })

    manifest = pd.DataFrame(rows)

    manifest = manifest[manifest["split"] != "unused"].copy()
    manifest = manifest.sort_values(["split", "pass", "date"]).reset_index(drop=True)

    if manifest.empty:
        raise RuntimeError(f"No complete files found under {FULL_DIR}")

    # Quick-test mode: keep some files from each split/pass.
    # This is better than taking the first N files per pass, because the first N
    # files are only from 2020 and give no validation/test data.
    if "MAX_FILES_PER_SPLIT_PER_PASS" in globals() and MAX_FILES_PER_SPLIT_PER_PASS is not None:
        manifest = (
            manifest
            .groupby(["split", "pass"], group_keys=False)
            .head(MAX_FILES_PER_SPLIT_PER_PASS)
            .reset_index(drop=True)
        )

    return manifest

# ============================================================
# 4. FEATURE SELECTION RULES
# ============================================================

TARGET_DERIVED_COLUMNS_TO_EXCLUDE = {
    # These are derived from observed SMAP and should not be used
    # as ordinary ML predictors for filling true missing cells.
    "trend_hat",
    "resid",
    "sm_for_kriging",
    "trend_removed",
    "p_trend",
    "r2_trend",
    "trend_ind",
    "pixel_id",
}

METADATA_COLUMNS_TO_EXCLUDE = {
    "date",
    "pass",
    "smap_pixel_key",
    "smap_status",
    "source_file",
    "source_smap_file",
    "geometry_wkt",
    "geometry",
}

SPATIAL_CANDIDATES = [
    "x",
    "y",
    "lat",
    "lon",
    "grid_row",
    "grid_col",
]

TEMPORAL_COLUMNS = [
    "year",
    "month",
    "day_of_year",
    "sin_doy",
    "cos_doy",
]

PASS_COLUMNS = [
    "pass_pm",
]


def get_header_columns(first_file: Path) -> list[str]:
    return pd.read_csv(first_file, nrows=0).columns.tolist()


def get_pta_value_columns(columns: list[str]) -> list[str]:
    cols = []

    # Use config ordering if available.
    if hasattr(cfg, "IEM_VARIABLES"):
        for base in cfg.IEM_VARIABLES:
            c = f"{base}_pta"
            if c in columns:
                cols.append(c)

    remaining = sorted([
        c for c in columns
        if c.endswith("_pta")
        and not c.endswith("_pta_var")
        and c not in cols
    ])

    cols.extend(remaining)
    return cols


def get_pta_var_columns(columns: list[str]) -> list[str]:
    return sorted([c for c in columns if c.endswith("_pta_var")])


def get_n_sample_columns(columns: list[str]) -> list[str]:
    return sorted([c for c in columns if c.endswith("_n_samples")])


def select_candidate_features(columns: list[str]) -> dict[str, list[str]]:
    spatial = [
        c for c in SPATIAL_CANDIDATES
        if c in columns
    ] if INCLUDE_SPATIAL_COLUMNS else []

    pta_value = get_pta_value_columns(columns) if INCLUDE_PTA_VALUE_COLUMNS else []
    pta_var = get_pta_var_columns(columns) if INCLUDE_PTA_VAR_COLUMNS else []
    n_samples = get_n_sample_columns(columns) if INCLUDE_N_SAMPLE_COLUMNS else []

    temporal = TEMPORAL_COLUMNS if INCLUDE_TEMPORAL_COLUMNS else []
    pass_features = PASS_COLUMNS if INCLUDE_PASS_INDICATOR and len(PASSES_TO_USE) > 1 else []

    excluded = TARGET_DERIVED_COLUMNS_TO_EXCLUDE | METADATA_COLUMNS_TO_EXCLUDE | {TARGET}

    # Safety: remove excluded columns if they slipped in.
    spatial = [c for c in spatial if c not in excluded]
    pta_value = [c for c in pta_value if c not in excluded]
    pta_var = [c for c in pta_var if c not in excluded]
    n_samples = [c for c in n_samples if c not in excluded]

    all_features = []
    for group in [spatial, temporal, pass_features, pta_value, pta_var, n_samples]:
        for c in group:
            if c not in all_features:
                all_features.append(c)

    return {
        "spatial": spatial,
        "temporal": temporal,
        "pass": pass_features,
        "pta_value": pta_value,
        "pta_var": pta_var,
        "n_samples": n_samples,
        "all_features": all_features,
    }


def build_feature_groups(feature_sets: dict[str, list[str]]) -> dict[str, list[str]]:
    spatial = feature_sets["spatial"]
    temporal = feature_sets["temporal"]
    pass_cols = feature_sets["pass"]
    pta = feature_sets["pta_value"]
    pta_var = feature_sets["pta_var"]
    n_samples = feature_sets["n_samples"]

    weather_bases = ["precip", "rh", "speed", "gust", "et"]
    weather_pta = [f"{b}_pta" for b in weather_bases if f"{b}_pta" in pta]

    soil_vwc = [c for c in pta if "vwc" in c.lower()]
    soil_temp = [
        c for c in pta
        if c.startswith("soil")
        and "vwc" not in c.lower()
    ]

    spatial_temporal = spatial + temporal + pass_cols
    soil_all = soil_temp + soil_vwc

    groups = {
        "spatial_only": spatial,
        "temporal_only": temporal + pass_cols,
        "spatial_temporal": spatial_temporal,
        "weather_pta_only": weather_pta,
        "soil_temperature_pta_only": soil_temp,
        "soil_vwc_pta_only": soil_vwc,
        "all_iem_pta": pta,
        "spatial_temporal_weather": spatial_temporal + weather_pta,
        "spatial_temporal_soil": spatial_temporal + soil_all,
        "spatial_temporal_all_iem": spatial_temporal + pta,
    }

    if INCLUDE_PTA_VAR_COLUMNS or INCLUDE_N_SAMPLE_COLUMNS:
        groups["spatial_temporal_iem_quality"] = (
            spatial_temporal + pta + pta_var + n_samples
        )

    # Remove duplicates and empty groups.
    clean = {}
    for name, cols in groups.items():
        seen = set()
        deduped = []
        for c in cols:
            if c not in seen:
                deduped.append(c)
                seen.add(c)

        if deduped:
            clean[name] = deduped

    return clean


# ============================================================
# 5. DATE FEATURES
# ============================================================

def add_date_features(df: pd.DataFrame, date: str, pass_name: str) -> pd.DataFrame:
    dt = pd.to_datetime(date)

    df = df.copy()

    # Keep the actual pass label for grouping and reporting.
    df["pass"] = pass_name.lower()

    df["date"] = dt.strftime("%Y-%m-%d")
    df["year"] = dt.year
    df["month"] = dt.month
    df["day_of_year"] = dt.dayofyear
    df["sin_doy"] = np.sin(2.0 * np.pi * dt.dayofyear / 366.0)
    df["cos_doy"] = np.cos(2.0 * np.pi * dt.dayofyear / 366.0)

    if len(PASSES_TO_USE) > 1:
        df["pass_pm"] = 1 if pass_name.lower() == "pm" else 0

    return df


# ============================================================
# 6. LOADING + AUDIT
# ============================================================

def read_selected_columns(path: Path, available_cols: list[str], feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    needed = set()

    needed.add(TARGET)
    needed.add("smap_status")
    needed.add("smap_pixel_key")

    for c in feature_sets["spatial"]:
        needed.add(c)
    for c in feature_sets["pta_value"]:
        needed.add(c)
    for c in feature_sets["pta_var"]:
        needed.add(c)
    for c in feature_sets["n_samples"]:
        needed.add(c)

    # Date/pass features are added later, not read from file.
    usecols = [c for c in available_cols if c in needed]

    return pd.read_csv(path, usecols=usecols, low_memory=False)


def initialize_feature_count_table(feature_cols: list[str]) -> dict:
    return {
        c: {
            "n_all": 0,
            "n_nonmissing_all": 0,
            "n_observed_rows": 0,
            "n_nonmissing_observed": 0,
            "n_gap_rows": 0,
            "n_nonmissing_gap": 0,
        }
        for c in feature_cols
    }


def collect_data_and_audit(
    manifest: pd.DataFrame,
    available_cols: list[str],
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = feature_sets["all_features"]

    audit_rows = []
    missingness_records = {}
    sample_parts = []

    for _, row in manifest.iterrows():
        path = Path(row["path"])
        date = row["date"]
        pass_name = row["pass"]
        split = row["split"]

        if split not in missingness_records:
            missingness_records[split] = initialize_feature_count_table(feature_cols)

        df = read_selected_columns(path, available_cols, feature_sets)
        df = add_date_features(df, date=date, pass_name=pass_name)

        # Ensure every candidate feature exists in the per-file dataframe.
        for c in feature_cols:
            if c not in df.columns:
                df[c] = np.nan

        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

        observed_mask = df[TARGET].notna()
        gap_mask = df[TARGET].isna()

        n_total = len(df)
        n_observed = int(observed_mask.sum())
        n_gap = int(gap_mask.sum())

        audit_rows.append({
            "file": path.name,
            "date": date,
            "pass": pass_name,
            "split": split,
            "n_rows": n_total,
            "n_observed_target": n_observed,
            "n_gap_target": n_gap,
        })

        # Accumulate feature missingness.
        for c in feature_cols:
            s = pd.to_numeric(df[c], errors="coerce")

            missingness_records[split][c]["n_all"] += n_total
            missingness_records[split][c]["n_nonmissing_all"] += int(s.notna().sum())

            missingness_records[split][c]["n_observed_rows"] += n_observed
            missingness_records[split][c]["n_nonmissing_observed"] += int(s[observed_mask].notna().sum())

            missingness_records[split][c]["n_gap_rows"] += n_gap
            missingness_records[split][c]["n_nonmissing_gap"] += int(s[gap_mask].notna().sum())

        # Collect observed sample for correlations/models.
        obs_cols = feature_cols + [TARGET, "date", "year", "month", "pass"]

        # Remove duplicate column names while preserving order.
        obs_cols = list(dict.fromkeys(obs_cols))

        obs = df.loc[observed_mask, obs_cols].copy()

        if MAX_OBSERVED_ROWS_PER_FILE_FOR_STATS is not None and len(obs) > MAX_OBSERVED_ROWS_PER_FILE_FOR_STATS:
            seed = RANDOM_STATE + int(pd.to_datetime(date).strftime("%Y%m%d")) + (1 if pass_name == "pm" else 0)
            obs = obs.sample(n=MAX_OBSERVED_ROWS_PER_FILE_FOR_STATS, random_state=seed)

        sample_parts.append(obs)

        if len(audit_rows) % 200 == 0:
            print(f"Processed {len(audit_rows):,} files...")

    audit = pd.DataFrame(audit_rows)

    missing_rows = []
    for split, feature_dict in missingness_records.items():
        for feature, counts in feature_dict.items():
            n_all = counts["n_all"]
            n_obs = counts["n_observed_rows"]
            n_gap = counts["n_gap_rows"]

            missing_rows.append({
                "split": split,
                "feature": feature,
                **counts,
                "missing_rate_all": 1.0 - counts["n_nonmissing_all"] / n_all if n_all else np.nan,
                "missing_rate_observed": 1.0 - counts["n_nonmissing_observed"] / n_obs if n_obs else np.nan,
                "missing_rate_gap": 1.0 - counts["n_nonmissing_gap"] / n_gap if n_gap else np.nan,
            })

    missingness = pd.DataFrame(missing_rows)

    observed_sample = pd.concat(sample_parts, ignore_index=True)

    # Remove any accidental duplicate columns after concatenation.
    observed_sample = observed_sample.loc[:, ~observed_sample.columns.duplicated()].copy()

    # Convert feature columns to numeric.
    for c in feature_cols:
        if c in observed_sample.columns:
            observed_sample[c] = pd.to_numeric(observed_sample[c], errors="coerce")

    observed_sample[TARGET] = pd.to_numeric(observed_sample[TARGET], errors="coerce")

    return audit, missingness, observed_sample

# ============================================================
# 7. CORRELATIONS
# ============================================================

def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(tmp) < 20:
        return np.nan

    if tmp["x"].nunique(dropna=True) <= 1 or tmp["y"].nunique(dropna=True) <= 1:
        return np.nan

    return float(tmp["x"].corr(tmp["y"], method=method))


def compute_correlations(train_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []

    y = train_df[TARGET]

    # Month anomaly target
    y_month_anom = y - train_df.groupby("month")[TARGET].transform("mean")

    for c in feature_cols:
        x = train_df[c]

        x_month_anom = x - train_df.groupby("month")[c].transform("mean")

        rows.append({
            "feature": c,
            "n_nonmissing_pair": int(pd.DataFrame({"x": x, "y": y}).dropna().shape[0]),
            "feature_missing_rate_train_observed_sample": float(x.isna().mean()),
            "feature_std_train_observed_sample": float(x.std(skipna=True)) if x.notna().sum() else np.nan,
            "feature_nunique_train_observed_sample": int(x.nunique(dropna=True)),
            "pearson": safe_corr(x, y, "pearson"),
            "spearman": safe_corr(x, y, "spearman"),
            "month_anomaly_pearson": safe_corr(x_month_anom, y_month_anom, "pearson"),
            "month_anomaly_spearman": safe_corr(x_month_anom, y_month_anom, "spearman"),
        })

    out = pd.DataFrame(rows)

    for col in ["pearson", "spearman", "month_anomaly_pearson", "month_anomaly_spearman"]:
        out[f"abs_{col}"] = out[col].abs()

    out = out.sort_values(
        ["abs_month_anomaly_spearman", "abs_spearman", "abs_pearson"],
        ascending=False,
    ).reset_index(drop=True)

    return out


# ============================================================
# 8. MODEL DATA
# ============================================================

def sample_rows(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=random_state).copy()


def get_split_data(observed_sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = observed_sample[observed_sample["year"].isin(TRAIN_YEARS)].copy()
    val = observed_sample[observed_sample["year"].isin(VALIDATION_YEARS)].copy()
    test = observed_sample[observed_sample["year"].isin(TEST_YEARS)].copy()

    train_model = sample_rows(train, MAX_TRAIN_ROWS_FOR_MODELS, RANDOM_STATE)
    val_model = sample_rows(val, MAX_VALID_ROWS_FOR_MODELS, RANDOM_STATE + 1)
    test_model = sample_rows(test, MAX_VALID_ROWS_FOR_MODELS, RANDOM_STATE + 2)

    return train_model, val_model, test_model


def clean_feature_list(df: pd.DataFrame, cols: list[str]) -> list[str]:
    usable = []

    for c in cols:
        if c not in df.columns:
            continue

        s = pd.to_numeric(df[c], errors="coerce")

        # Drop completely missing or constant features in training.
        if s.notna().sum() < 20:
            continue

        if s.nunique(dropna=True) <= 1:
            continue

        usable.append(c)

    return usable


# ============================================================
# 9. MODELS
# ============================================================

def get_models() -> dict:
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=N_TREES,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=N_TREES,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            max_iter=350,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
    }

    if INCLUDE_XGBOOST_IF_AVAILABLE:
        try:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor(
                n_estimators=500,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        except Exception as exc:
            print(f"[info] XGBoost not available or failed to import: {exc}")

    return models


def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def compute_metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r2": r2,
    }


def baseline_scores(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()

    # Global train mean
    pred_mean = np.full_like(y_val, fill_value=np.mean(y_train), dtype=float)
    rows.append({
        "feature_group": "baseline_global_train_mean",
        "model": "baseline",
        "n_features": 0,
        **compute_metrics(y_val, pred_mean),
    })

    # Month-specific train mean
    month_mean = train_df.groupby("month")[TARGET].mean()
    global_mean = float(np.mean(y_train))

    pred_month = val_df["month"].map(month_mean).fillna(global_mean).to_numpy(dtype=float)

    rows.append({
        "feature_group": "baseline_train_month_mean",
        "model": "baseline",
        "n_features": 1,
        **compute_metrics(y_val, pred_month),
    })

    return pd.DataFrame(rows)


def run_feature_group_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict]:
    models = get_models()

    score_rows = []
    fitted = {}

    # Baselines first
    score_rows.extend(baseline_scores(train_df, val_df).to_dict("records"))

    for group_name, cols in feature_groups.items():
        usable_cols = clean_feature_list(train_df, cols)

        if not usable_cols:
            print(f"[skip] {group_name}: no usable features")
            continue

        X_train = train_df[usable_cols]
        y_train = train_df[TARGET].to_numpy()

        X_val = val_df[usable_cols]
        y_val = val_df[TARGET].to_numpy()

        print(f"\nFeature group: {group_name}")
        print(f"  features: {len(usable_cols)}")
        print(f"  train rows: {len(X_train):,}")
        print(f"  validation rows: {len(X_val):,}")

        for model_name, model in models.items():
            print(f"  fitting {model_name}...")

            # Important: clone the estimator so each feature group gets
            # its own independent model object. Otherwise the same model
            # gets refit repeatedly and earlier stored pipelines point to
            # the last fitted feature group.
            pipe = make_pipeline(clone(model))
            pipe.fit(X_train, y_train)

            pred = pipe.predict(X_val)
            metrics = compute_metrics(y_val, pred)

            score_rows.append({
                "feature_group": group_name,
                "model": model_name,
                "n_features": len(usable_cols),
                "features": ";".join(usable_cols),
                **metrics,
            })

            fitted[(group_name, model_name)] = {
                "pipeline": pipe,
                "features": usable_cols,
            }

    scores = pd.DataFrame(score_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)

    return scores, fitted


# ============================================================
# 10. IMPORTANCE
# ============================================================

def extract_builtin_importance(fitted: dict) -> pd.DataFrame:
    rows = []

    for (group_name, model_name), obj in fitted.items():
        pipe = obj["pipeline"]
        features = obj["features"]
        model = pipe.named_steps["model"]

        if not hasattr(model, "feature_importances_"):
            continue

        imp = model.feature_importances_

        if len(imp) != len(features):
            continue

        for feature, value in zip(features, imp):
            rows.append({
                "feature_group": group_name,
                "model": model_name,
                "feature": feature,
                "builtin_importance": float(value),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["model", "feature_group", "builtin_importance"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return out


def run_permutation_importance_on_best_models(
    fitted: dict,
    scores: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    # Use best model per model type among non-baselines.
    non_base = scores[scores["model"] != "baseline"].copy()

    if non_base.empty:
        return pd.DataFrame()

    best_rows = (
        non_base
        .sort_values(["model", "rmse"])
        .groupby("model", as_index=False)
        .head(1)
    )

    val_perm = sample_rows(val_df, MAX_VALID_ROWS_FOR_PERMUTATION, RANDOM_STATE + 10)

    for _, row in best_rows.iterrows():
        group_name = row["feature_group"]
        model_name = row["model"]

        key = (group_name, model_name)

        if key not in fitted:
            continue

        obj = fitted[key]
        pipe = obj["pipeline"]
        features = obj["features"]

        X_val = val_perm[features]
        y_val = val_perm[TARGET].to_numpy()

        print(f"\nPermutation importance: {model_name} / {group_name}")
        print(f"  validation rows used: {len(X_val):,}")

        result = permutation_importance(
            pipe,
            X_val,
            y_val,
            scoring="neg_root_mean_squared_error",
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        for feature, mean_imp, std_imp in zip(
            features,
            result.importances_mean,
            result.importances_std,
        ):
            rows.append({
                "model": model_name,
                "feature_group": group_name,
                "feature": feature,
                "permutation_importance_rmse_increase": float(mean_imp),
                "permutation_importance_std": float(std_imp),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(
        "permutation_importance_rmse_increase",
        ascending=False,
    ).reset_index(drop=True)

    return out


# ============================================================
# 11. RECOMMENDATION TABLE
# ============================================================

def make_recommendation_table(
    feature_cols: list[str],
    missingness: pd.DataFrame,
    corr: pd.DataFrame,
    builtin: pd.DataFrame,
    perm: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame({"feature": feature_cols})

    # Train missingness on observed rows and all rows
    miss_train = missingness[missingness["split"] == "train"].copy()

    keep_miss_cols = [
        "feature",
        "missing_rate_all",
        "missing_rate_observed",
        "missing_rate_gap",
    ]

    miss_train = miss_train[keep_miss_cols].rename(columns={
        "missing_rate_all": "train_missing_rate_all",
        "missing_rate_observed": "train_missing_rate_observed",
        "missing_rate_gap": "train_missing_rate_gap",
    })

    base = base.merge(miss_train, on="feature", how="left")

    corr_keep = [
        "feature",
        "pearson",
        "spearman",
        "month_anomaly_pearson",
        "month_anomaly_spearman",
        "abs_spearman",
        "abs_month_anomaly_spearman",
    ]
    base = base.merge(corr[corr_keep], on="feature", how="left")

    if not builtin.empty:
        builtin_agg = (
            builtin
            .groupby("feature", as_index=False)["builtin_importance"]
            .mean()
            .rename(columns={"builtin_importance": "mean_builtin_importance"})
        )
        base = base.merge(builtin_agg, on="feature", how="left")
    else:
        base["mean_builtin_importance"] = np.nan

    if not perm.empty:
        perm_agg = (
            perm
            .groupby("feature", as_index=False)["permutation_importance_rmse_increase"]
            .mean()
            .rename(columns={
                "permutation_importance_rmse_increase": "mean_permutation_rmse_increase"
            })
        )
        base = base.merge(perm_agg, on="feature", how="left")
    else:
        base["mean_permutation_rmse_increase"] = np.nan

    # Ranks: smaller is better.
    base["rank_abs_month_anom_spearman"] = (
        base["abs_month_anomaly_spearman"]
        .rank(ascending=False, method="min")
    )
    base["rank_abs_spearman"] = (
        base["abs_spearman"]
        .rank(ascending=False, method="min")
    )
    base["rank_builtin_importance"] = (
        base["mean_builtin_importance"]
        .rank(ascending=False, method="min")
    )
    base["rank_permutation_importance"] = (
        base["mean_permutation_rmse_increase"]
        .rank(ascending=False, method="min")
    )

    rank_cols = [
        "rank_abs_month_anom_spearman",
        "rank_abs_spearman",
        "rank_builtin_importance",
        "rank_permutation_importance",
    ]

    base["mean_rank"] = base[rank_cols].mean(axis=1, skipna=True)

    # Basic recommendation flag.
    base["recommended_initial_keep"] = (
        (base["train_missing_rate_gap"].fillna(1.0) < 0.25)
        & (
            (base["abs_month_anomaly_spearman"].fillna(0.0) >= 0.03)
            | (base["mean_permutation_rmse_increase"].fillna(0.0) > 0.0)
            | (base["mean_builtin_importance"].fillna(0.0) > 0.0)
        )
    )

    base = base.sort_values(["recommended_initial_keep", "mean_rank"], ascending=[False, True])

    return base.reset_index(drop=True)


def make_cokriging_candidate_table(recommendation: pd.DataFrame) -> pd.DataFrame:
    # Cokriging should usually use a small number of strong, continuous auxiliary variables.
    pta = recommendation[
        recommendation["feature"].str.endswith("_pta", na=False)
        & ~recommendation["feature"].str.endswith("_pta_var", na=False)
    ].copy()

    pta = pta.sort_values(
        [
            "abs_month_anomaly_spearman",
            "abs_spearman",
            "mean_permutation_rmse_increase",
        ],
        ascending=False,
    ).reset_index(drop=True)

    pta["cokriging_note"] = (
        "Candidate auxiliary variable. Prefer high correlation, low missingness, "
        "and physically interpretable relation to SMAP soil moisture."
    )

    return pta


# ============================================================
# 12. PLOTS
# ============================================================

def plot_top_correlations(corr: pd.DataFrame) -> None:
    if corr.empty:
        return

    top = corr.head(TOP_N_TO_PLOT).copy()
    top = top.sort_values("abs_month_anomaly_spearman", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["feature"], top["abs_month_anomaly_spearman"])
    ax.set_xlabel("|Month-anomaly Spearman correlation|")
    ax.set_title("Top feature correlations with observed SMAP soil moisture")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = FIG_DIR / "top_feature_correlations.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_feature_group_rmse(scores: pd.DataFrame) -> None:
    if scores.empty:
        return

    non_base = scores.copy()
    non_base["label"] = non_base["model"] + " | " + non_base["feature_group"]

    top = non_base.sort_values("rmse").head(TOP_N_TO_PLOT).copy()
    top = top.sort_values("rmse", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["label"], top["rmse"])
    ax.set_xlabel("Validation RMSE")
    ax.set_title("Best feature-group / model combinations")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = FIG_DIR / "feature_group_rmse.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_top_permutation_importance(perm: pd.DataFrame) -> None:
    if perm.empty:
        return

    agg = (
        perm
        .groupby("feature", as_index=False)["permutation_importance_rmse_increase"]
        .mean()
        .sort_values("permutation_importance_rmse_increase", ascending=False)
        .head(TOP_N_TO_PLOT)
    )

    agg = agg.sort_values("permutation_importance_rmse_increase", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(agg["feature"], agg["permutation_importance_rmse_increase"])
    ax.set_xlabel("Mean validation RMSE increase after permutation")
    ax.set_title("Top permutation-importance features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = FIG_DIR / "top_permutation_importance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 13. MAIN
# ============================================================

def main() -> None:
    print("\nFeature screening for SMAP gap filling")
    print("=" * 80)
    print(f"Full input folder: {FULL_DIR}")
    print(f"Output folder:     {OUT_DIR}")
    print(f"Passes:            {PASSES_TO_USE}")
    print(f"Train years:       {TRAIN_YEARS}")
    print(f"Validation years:  {VALIDATION_YEARS}")
    print(f"Test years:        {TEST_YEARS}")
    print("=" * 80)

    manifest = build_manifest()
    manifest_path = OUT_DIR / "file_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nFiles by split/pass:")
    print(manifest.groupby(["split", "pass"]).size())

    first_file = Path(manifest.iloc[0]["path"])
    available_cols = get_header_columns(first_file)

    feature_sets = select_candidate_features(available_cols)
    feature_cols = feature_sets["all_features"]
    feature_groups = build_feature_groups(feature_sets)

    print("\nCandidate feature counts:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {len(cols)}")

    print("\nFeature groups:")
    for name, cols in feature_groups.items():
        print(f"  {name}: {len(cols)}")

    print("\nCollecting audits and observed samples...")
    audit, missingness, observed_sample = collect_data_and_audit(
        manifest=manifest,
        available_cols=available_cols,
        feature_sets=feature_sets,
    )

    audit_path = OUT_DIR / "dataset_audit_by_split.csv"
    missing_path = OUT_DIR / "feature_missingness_by_split.csv"
    sample_path = OUT_DIR / "observed_sample_for_screening_preview.csv"

    audit.to_csv(audit_path, index=False)
    missingness.to_csv(missing_path, index=False)
    observed_sample.head(5000).to_csv(sample_path, index=False)

    print(f"\nSaved audit:       {audit_path}")
    print(f"Saved missingness: {missing_path}")
    print(f"Saved preview:     {sample_path}")

    print("\nObserved sample rows by split:")
    print(observed_sample.groupby(
        observed_sample["year"].map(year_to_split)
    ).size())

    train_df, val_df, test_df = get_split_data(observed_sample)

    print("\nModel-screening sample sizes:")
    print(f"  train:      {len(train_df):,}")
    print(f"  validation: {len(val_df):,}")
    print(f"  test:       {len(test_df):,}  [not used for feature selection]")

    if train_df.empty:
        raise RuntimeError("No training rows found.")
    if val_df.empty:
        raise RuntimeError("No validation rows found.")

    print("\nComputing train correlations...")
    corr = compute_correlations(train_df, feature_cols)
    corr_path = OUT_DIR / "feature_correlations_train.csv"
    corr.to_csv(corr_path, index=False)
    print(f"Saved correlations: {corr_path}")

    print("\nRunning feature-group model screening...")
    scores, fitted = run_feature_group_models(
        train_df=train_df,
        val_df=val_df,
        feature_groups=feature_groups,
    )

    scores_path = OUT_DIR / "feature_group_model_scores_validation.csv"
    scores.to_csv(scores_path, index=False)
    print(f"\nSaved validation model scores: {scores_path}")

    print("\nExtracting built-in tree importances...")
    builtin = extract_builtin_importance(fitted)
    builtin_path = OUT_DIR / "model_builtin_importance.csv"
    builtin.to_csv(builtin_path, index=False)
    print(f"Saved built-in importances: {builtin_path}")

    print("\nRunning permutation importance on best models...")
    perm = run_permutation_importance_on_best_models(
        fitted=fitted,
        scores=scores,
        val_df=val_df,
    )

    perm_path = OUT_DIR / "model_permutation_importance.csv"
    perm.to_csv(perm_path, index=False)
    print(f"Saved permutation importances: {perm_path}")

    print("\nBuilding recommendation tables...")
    recommendation = make_recommendation_table(
        feature_cols=feature_cols,
        missingness=missingness,
        corr=corr,
        builtin=builtin,
        perm=perm,
    )

    recommendation_path = OUT_DIR / "feature_recommendation_table.csv"
    recommendation.to_csv(recommendation_path, index=False)

    cokriging_candidates = make_cokriging_candidate_table(recommendation)
    cokriging_path = OUT_DIR / "top_cokriging_candidates.csv"
    cokriging_candidates.to_csv(cokriging_path, index=False)

    print(f"Saved recommendation table:       {recommendation_path}")
    print(f"Saved cokriging candidate table:  {cokriging_path}")

    print("\nMaking figures...")
    plot_top_correlations(corr)
    plot_feature_group_rmse(scores)
    plot_top_permutation_importance(perm)

    print("\nTop 20 recommended features:")
    display_cols = [
        "feature",
        "recommended_initial_keep",
        "train_missing_rate_gap",
        "spearman",
        "month_anomaly_spearman",
        "mean_builtin_importance",
        "mean_permutation_rmse_increase",
        "mean_rank",
    ]
    display_cols = [c for c in display_cols if c in recommendation.columns]
    print(recommendation[display_cols].head(20).to_string(index=False))

    print("\nBest validation model/group scores:")
    print(scores.head(20).to_string(index=False))

    print("\nDone.")
    print(f"Outputs saved in:\n{OUT_DIR}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()