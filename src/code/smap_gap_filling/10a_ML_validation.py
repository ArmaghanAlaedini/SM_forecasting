#!/usr/bin/env python3
"""
10a_ML_validation.py

ML-only validation for SMAP gap filling.

Models:
    random_forest
    extra_trees
    hist_gbdt
    xgboost, if installed
    ffnn_mlp

Validation modes:
    random_cell
    spatial_block

Input:
    03_full_smap_iem_data/{am,pm}/complete/*.csv

Output:
    05_gapfill_model_validation/ml/
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

TARGET = "soil_moisture"

PASSES_TO_USE = ["am", "pm"]

TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALIDATION_YEARS = [2024]
TEST_YEARS = [2025]

RUN_TEST = False

# Laptop quick test: 20
# Full HPC run: None
MAX_FILES_PER_SPLIT_PER_PASS = None

HOLDOUT_MODES = ["random_cell", "spatial_block"]

MAX_OBSERVED_ROWS_PER_FILE = 800
MAX_TRAIN_ROWS = 250_000
MAX_EVAL_TARGET_ROWS_PER_MODE = 120_000

EVAL_HOLDOUT_FRACTION = 0.25
MIN_DONOR_ROWS = 5
BLOCK_ATTEMPTS_PER_GROUP = 60

RANDOM_STATE = 42
N_TREES = 300

INCLUDE_XGBOOST_IF_AVAILABLE = True
INCLUDE_FFNN = True

SELECTED_IEM_PTA = [
    "soil12vwc_pta",
    "soil24vwc_pta",
    "soil50vwc_pta",
    "soil04tx_pta",
    "soil04t_pta",
    "soil04tn_pta",
    "rh_pta",
    "precip_pta",
    "et_pta",
]

SPATIOTEMPORAL_MINIMAL = [
    "x",
    "y",
    "sin_doy",
    "cos_doy",
    "pass_pm",
]

SAVE_PREDICTION_SAMPLE_ROWS = 250_000


# ============================================================
# CONFIG / PATHS
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


def get_gap_filling_dir() -> Path:
    if hasattr(cfg, "GAP_FILLING_DIR"):
        return Path(cfg.GAP_FILLING_DIR)
    return Path(cfg.PROCESSED_DIR) / "smap_gap_filling"


def get_full_smap_iem_dir() -> Path:
    if hasattr(cfg, "FULL_SMAP_IEM_DIR"):
        return Path(cfg.FULL_SMAP_IEM_DIR)
    return get_gap_filling_dir() / "03_full_smap_iem_data"


FULL_DIR = get_full_smap_iem_dir()
OUT_DIR = get_gap_filling_dir() / "05_gapfill_model_validation" / "ml"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MANIFEST
# ============================================================

def parse_date_from_file(path: Path) -> str:
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
        folder = FULL_DIR / pass_name / "complete"
        files = sorted(folder.glob(f"smap_iem_{pass_name}_complete_*.csv"))

        for path in files:
            date = parse_date_from_file(path)
            year = pd.to_datetime(date).year
            split = year_to_split(year)

            if split == "unused":
                continue

            rows.append({
                "path": str(path),
                "file_name": path.name,
                "date": date,
                "year": year,
                "pass": pass_name,
                "split": split,
            })

    manifest = pd.DataFrame(rows)

    if manifest.empty:
        raise RuntimeError(f"No complete files found under {FULL_DIR}")

    manifest = manifest.sort_values(["split", "pass", "date"]).reset_index(drop=True)

    if MAX_FILES_PER_SPLIT_PER_PASS is not None:
        manifest = (
            manifest
            .groupby(["split", "pass"], group_keys=False)
            .head(MAX_FILES_PER_SPLIT_PER_PASS)
            .reset_index(drop=True)
        )

    return manifest


# ============================================================
# FEATURES
# ============================================================

def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def get_all_iem_pta_columns(columns: list[str]) -> list[str]:
    ordered = []

    if hasattr(cfg, "IEM_PTA_VARIABLES"):
        for base in cfg.IEM_PTA_VARIABLES:
            col = f"{base}_pta"
            if col in columns:
                ordered.append(col)

    remaining = sorted([
        c for c in columns
        if c.endswith("_pta")
        and not c.endswith("_pta_var")
        and c not in ordered
    ])

    return ordered + remaining


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    all_iem = get_all_iem_pta_columns(columns)

    selected_iem = [c for c in SELECTED_IEM_PTA if c in columns]

    st_min = [
        c for c in SPATIOTEMPORAL_MINIMAL
        if c in columns or c in {"sin_doy", "cos_doy", "pass_pm"}
    ]

    feature_sets = {
        "selected_iem_spatiotemporal": selected_iem + st_min,
        "selected_iem_pta": selected_iem,
        "all_iem_spatiotemporal": all_iem + st_min,
        "all_iem_pta": all_iem,
    }

    clean = {}

    excluded = {
        TARGET,
        "smap_status",
        "geometry_wkt",
        "date",
        "pass",
        "smap_pixel_key",
        "source_file",
    }

    for name, cols in feature_sets.items():
        cols = list(dict.fromkeys(cols))
        cols = [c for c in cols if c not in excluded]
        if cols:
            clean[name] = cols

    return clean


def all_needed_columns(feature_sets: dict[str, list[str]], available_cols: list[str]) -> list[str]:
    needed = {
        TARGET,
        "smap_status",
        "smap_pixel_key",
        "x",
        "y",
        "grid_row",
        "grid_col",
    }

    for cols in feature_sets.values():
        needed.update(cols)

    generated_later = {
        "year",
        "month",
        "day_of_year",
        "sin_doy",
        "cos_doy",
        "pass_pm",
    }

    needed = needed - generated_later

    return [c for c in available_cols if c in needed]


# ============================================================
# DATA LOADING
# ============================================================

def add_date_features(df: pd.DataFrame, date: str, pass_name: str) -> pd.DataFrame:
    dt = pd.to_datetime(date)

    df = df.copy()
    df["date"] = dt.strftime("%Y-%m-%d")
    df["year"] = int(dt.year)
    df["month"] = int(dt.month)
    df["day_of_year"] = int(dt.dayofyear)
    df["sin_doy"] = float(np.sin(2.0 * np.pi * dt.dayofyear / 366.0))
    df["cos_doy"] = float(np.cos(2.0 * np.pi * dt.dayofyear / 366.0))
    df["pass"] = pass_name.lower()
    df["pass_pm"] = 1 if pass_name.lower() == "pm" else 0

    return df


def sample_if_needed(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=seed).copy()


def collect_observed_data(
    manifest: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    available_cols: list[str],
) -> pd.DataFrame:
    usecols = all_needed_columns(feature_sets, available_cols)
    feature_cols = sorted(set(sum(feature_sets.values(), [])))

    parts = []

    for i, row in manifest.iterrows():
        path = Path(row["path"])
        date = row["date"]
        pass_name = row["pass"]

        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df = add_date_features(df, date=date, pass_name=pass_name)

        for c in feature_cols:
            if c not in df.columns:
                df[c] = np.nan

        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        obs = df.loc[df[TARGET].notna()].copy()

        for c in feature_cols + ["x", "y", "grid_row", "grid_col"]:
            if c in obs.columns:
                obs[c] = pd.to_numeric(obs[c], errors="coerce")

        if MAX_OBSERVED_ROWS_PER_FILE is not None and len(obs) > MAX_OBSERVED_ROWS_PER_FILE:
            seed = (
                RANDOM_STATE
                + int(pd.to_datetime(date).strftime("%Y%m%d"))
                + (1 if pass_name == "pm" else 0)
            )
            obs = obs.sample(n=MAX_OBSERVED_ROWS_PER_FILE, random_state=seed)

        parts.append(obs)

        if (i + 1) % 200 == 0:
            print(f"Loaded observed rows from {i + 1:,} files...")

    if not parts:
        raise RuntimeError("No observed rows collected.")

    out = pd.concat(parts, ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


# ============================================================
# HOLDOUTS
# ============================================================

def mark_random_cell_holdouts(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["eval_role"] = "donor"

    rng = np.random.default_rng(
        RANDOM_STATE + (100 if split_name == "validation" else 200)
    )

    for (_, _), idx in df.groupby(["date", "pass"]).groups.items():
        idx = np.asarray(list(idx))

        if len(idx) <= MIN_DONOR_ROWS + 1:
            continue

        n_holdout = max(1, int(round(len(idx) * EVAL_HOLDOUT_FRACTION)))
        n_holdout = min(n_holdout, len(idx) - MIN_DONOR_ROWS)

        chosen = rng.choice(idx, size=n_holdout, replace=False)
        df.loc[chosen, "eval_role"] = "target"

    return df


def choose_spatial_block_indices(group: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    if len(group) <= MIN_DONOR_ROWS + 1:
        return np.asarray([], dtype=object)

    desired = max(1, int(round(len(group) * EVAL_HOLDOUT_FRACTION)))
    max_allowed = len(group) - MIN_DONOR_ROWS

    best_idx = None
    best_score = np.inf

    has_grid = (
        "grid_row" in group.columns
        and "grid_col" in group.columns
        and group["grid_row"].notna().any()
        and group["grid_col"].notna().any()
    )

    if has_grid:
        g = group.dropna(subset=["grid_row", "grid_col"]).copy()

        rows = np.sort(g["grid_row"].unique())
        cols = np.sort(g["grid_col"].unique())

        if len(rows) == 0 or len(cols) == 0:
            return np.asarray([], dtype=object)

        n_row_block = max(1, int(np.ceil(np.sqrt(EVAL_HOLDOUT_FRACTION) * len(rows))))
        n_col_block = max(1, int(np.ceil(np.sqrt(EVAL_HOLDOUT_FRACTION) * len(cols))))

        n_row_block = min(n_row_block, len(rows))
        n_col_block = min(n_col_block, len(cols))

        for _ in range(BLOCK_ATTEMPTS_PER_GROUP):
            row_start = int(rng.integers(0, max(1, len(rows) - n_row_block + 1)))
            col_start = int(rng.integers(0, max(1, len(cols) - n_col_block + 1)))

            row_vals = set(rows[row_start: row_start + n_row_block])
            col_vals = set(cols[col_start: col_start + n_col_block])

            mask = g["grid_row"].isin(row_vals) & g["grid_col"].isin(col_vals)
            candidate_idx = g.index[mask].to_numpy()
            n_candidate = len(candidate_idx)

            if n_candidate < 1 or n_candidate > max_allowed:
                continue

            score = abs(n_candidate - desired)

            if score < best_score:
                best_score = score
                best_idx = candidate_idx

        if best_idx is not None and len(best_idx) > 0:
            return best_idx

    g = group.dropna(subset=["x", "y"]).copy()

    if len(g) <= MIN_DONOR_ROWS + 1:
        return np.asarray([], dtype=object)

    x_min, x_max = float(g["x"].min()), float(g["x"].max())
    y_min, y_max = float(g["y"].min()), float(g["y"].max())

    x_width = (x_max - x_min) * np.sqrt(EVAL_HOLDOUT_FRACTION)
    y_width = (y_max - y_min) * np.sqrt(EVAL_HOLDOUT_FRACTION)

    if x_width <= 0 or y_width <= 0:
        return np.asarray([], dtype=object)

    for _ in range(BLOCK_ATTEMPTS_PER_GROUP):
        cx = float(rng.uniform(x_min, x_max))
        cy = float(rng.uniform(y_min, y_max))

        mask = (
            (g["x"] >= cx - x_width / 2)
            & (g["x"] <= cx + x_width / 2)
            & (g["y"] >= cy - y_width / 2)
            & (g["y"] <= cy + y_width / 2)
        )

        candidate_idx = g.index[mask].to_numpy()
        n_candidate = len(candidate_idx)

        if n_candidate < 1 or n_candidate > max_allowed:
            continue

        score = abs(n_candidate - desired)

        if score < best_score:
            best_score = score
            best_idx = candidate_idx

    if best_idx is not None and len(best_idx) > 0:
        return best_idx

    return np.asarray([], dtype=object)


def mark_spatial_block_holdouts(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    df = df.copy()
    df["eval_role"] = "donor"

    rng = np.random.default_rng(
        RANDOM_STATE + (300 if split_name == "validation" else 400)
    )

    for (_, _), idx in df.groupby(["date", "pass"]).groups.items():
        group = df.loc[list(idx)]
        chosen_idx = choose_spatial_block_indices(group, rng)

        if len(chosen_idx) > 0:
            df.loc[chosen_idx, "eval_role"] = "target"

    return df


def mark_eval_holdouts(df: pd.DataFrame, split_name: str, holdout_mode: str) -> pd.DataFrame:
    if holdout_mode == "random_cell":
        return mark_random_cell_holdouts(df, split_name)
    if holdout_mode == "spatial_block":
        return mark_spatial_block_holdouts(df, split_name)
    raise ValueError(f"Unknown holdout mode: {holdout_mode}")


def prepare_base_splits(observed: pd.DataFrame):
    train = observed[observed["year"].isin(TRAIN_YEARS)].copy()
    val = observed[observed["year"].isin(VALIDATION_YEARS)].copy()
    test = observed[observed["year"].isin(TEST_YEARS)].copy()

    train = sample_if_needed(train, MAX_TRAIN_ROWS, RANDOM_STATE)

    return train, val, test


def make_eval_sets(base_df: pd.DataFrame, split_name: str) -> dict[str, pd.DataFrame]:
    eval_sets = {}

    for mode in HOLDOUT_MODES:
        df = mark_eval_holdouts(base_df, split_name=split_name, holdout_mode=mode)
        df["holdout_mode"] = mode

        targets = df[df["eval_role"] == "target"].copy()
        donors = df[df["eval_role"] == "donor"].copy()

        targets = sample_if_needed(
            targets,
            MAX_EVAL_TARGET_ROWS_PER_MODE,
            RANDOM_STATE + len(mode),
        )

        eval_sets[mode] = pd.concat([targets, donors], ignore_index=True)

    return eval_sets


# ============================================================
# MODELS
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
            print(f"[info] XGBoost skipped: {exc}")

    if INCLUDE_FFNN:
        models["ffnn_mlp"] = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )

    return models


def make_pipeline(model_name: str, model):
    if model_name == "ffnn_mlp":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", clone(model)),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clone(model)),
    ])


def clean_features(train_df: pd.DataFrame, cols: list[str]) -> list[str]:
    usable = []

    for c in cols:
        if c not in train_df.columns:
            continue

        s = pd.to_numeric(train_df[c], errors="coerce")

        if s.notna().sum() < 20:
            continue

        if s.nunique(dropna=True) <= 1:
            continue

        usable.append(c)

    return usable


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def evaluate_ml_models(train_df, eval_sets, feature_sets, split_name):
    models = get_models()
    y_train = train_df[TARGET].to_numpy(dtype=float)

    score_rows = []
    pred_parts = []

    for holdout_mode, eval_df in eval_sets.items():
        target_eval = eval_df[eval_df["eval_role"] == "target"].copy()
        if target_eval.empty:
            continue

        y_eval = target_eval[TARGET].to_numpy(dtype=float)
        mean_pred = np.full(len(target_eval), np.nanmean(y_train), dtype=float)

        score_rows.append({
            "split": split_name,
            "holdout_mode": holdout_mode,
            "feature_group": "baseline_train_mean",
            "model": "baseline",
            "n_features": 0,
            "features": "",
            **compute_metrics(y_eval, mean_pred),
        })

        out = target_eval[["date", "pass", "smap_pixel_key", TARGET, "x", "y"]].copy()
        out["split"] = split_name
        out["holdout_mode"] = holdout_mode
        out["feature_group"] = "baseline_train_mean"
        out["model"] = "baseline"
        out["prediction"] = mean_pred
        pred_parts.append(out)

    for feature_group, cols in feature_sets.items():
        usable = clean_features(train_df, cols)

        if not usable:
            print(f"[skip] {feature_group}: no usable features")
            continue

        X_train = train_df[usable]

        print(f"\nTraining feature group: {feature_group}")
        print(f"  features: {len(usable)}")
        print(f"  train rows: {len(X_train):,}")

        for model_name, model in models.items():
            print(f"  fitting {model_name}...")

            pipe = make_pipeline(model_name, model)
            pipe.fit(X_train, y_train)

            for holdout_mode, eval_df in eval_sets.items():
                target_eval = eval_df[eval_df["eval_role"] == "target"].copy()

                if target_eval.empty:
                    continue

                X_eval = target_eval[usable]
                y_eval = target_eval[TARGET].to_numpy(dtype=float)

                pred = pipe.predict(X_eval)

                score_rows.append({
                    "split": split_name,
                    "holdout_mode": holdout_mode,
                    "feature_group": feature_group,
                    "model": model_name,
                    "n_features": len(usable),
                    "features": ";".join(usable),
                    **compute_metrics(y_eval, pred),
                })

                out = target_eval[["date", "pass", "smap_pixel_key", TARGET, "x", "y"]].copy()
                out["split"] = split_name
                out["holdout_mode"] = holdout_mode
                out["feature_group"] = feature_group
                out["model"] = model_name
                out["prediction"] = pred
                pred_parts.append(out)

    return pd.DataFrame(score_rows), pd.concat(pred_parts, ignore_index=True)


# ============================================================
# PLOTS
# ============================================================

def plot_rmse(scores: pd.DataFrame):
    for mode in sorted(scores["holdout_mode"].dropna().unique()):
        sub = scores[
            (scores["split"] == "validation")
            & (scores["holdout_mode"] == mode)
        ].copy()

        if sub.empty:
            continue

        sub["label"] = sub["model"] + " | " + sub["feature_group"]
        sub = sub.sort_values("rmse", ascending=True).head(30)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(sub["label"], sub["rmse"])
        ax.invert_yaxis()
        ax.set_xlabel("Validation RMSE, lower is better")
        ax.set_title(f"ML validation RMSE: {mode}")
        ax.grid(axis="x", alpha=0.25)

        for i, value in enumerate(sub["rmse"]):
            ax.text(value + 0.0003, i, f"{value:.4f}", va="center", fontsize=8)

        fig.tight_layout()
        out = FIG_DIR / f"ml_validation_rmse_{mode}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


def plot_bias(scores: pd.DataFrame):
    for mode in sorted(scores["holdout_mode"].dropna().unique()):
        sub = scores[
            (scores["split"] == "validation")
            & (scores["holdout_mode"] == mode)
        ].copy()

        if sub.empty:
            continue

        sub["label"] = sub["model"] + " | " + sub["feature_group"]
        sub = sub.sort_values("rmse", ascending=True).head(30)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(sub["label"], sub["bias"])
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Bias = mean(prediction - observed)")
        ax.set_title(f"ML validation bias: {mode}")
        ax.grid(axis="x", alpha=0.25)

        fig.tight_layout()
        out = FIG_DIR / f"ml_validation_bias_{mode}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nML-only SMAP gap-filling validation")
    print("=" * 80)
    print(f"Input folder:     {FULL_DIR}")
    print(f"Output folder:    {OUT_DIR}")
    print(f"Max files/split:  {MAX_FILES_PER_SPLIT_PER_PASS}")
    print(f"RUN_TEST:         {RUN_TEST}")
    print("=" * 80)

    manifest = build_manifest()
    manifest.to_csv(OUT_DIR / "ml_validation_manifest.csv", index=False)

    print("\nFiles by split/pass:")
    print(manifest.groupby(["split", "pass"]).size())

    first_file = Path(manifest.iloc[0]["path"])
    available_cols = read_header(first_file)

    feature_sets = build_feature_sets(available_cols)

    pd.DataFrame(
        [{"feature_group": k, "feature": f} for k, cols in feature_sets.items() for f in cols]
    ).to_csv(OUT_DIR / "ml_feature_sets_used.csv", index=False)

    print("\nFeature sets:")
    for name, cols in feature_sets.items():
        print(f"  {name}: {len(cols)}")

    print("\nCollecting observed rows...")
    observed = collect_observed_data(manifest, feature_sets, available_cols)

    train_df, val_base, test_base = prepare_base_splits(observed)

    print("\nRows:")
    print(f"  train: {len(train_df):,}")
    print(f"  validation base: {len(val_base):,}")
    print(f"  test base: {len(test_base):,}")

    if train_df.empty:
        raise RuntimeError("No training rows found.")

    val_eval_sets = make_eval_sets(val_base, "validation")

    for mode, df in val_eval_sets.items():
        print(
            f"  validation {mode}: "
            f"donors={(df['eval_role'] == 'donor').sum():,}, "
            f"targets={(df['eval_role'] == 'target').sum():,}"
        )

    all_scores = []
    all_preds = []

    scores, preds = evaluate_ml_models(
        train_df=train_df,
        eval_sets=val_eval_sets,
        feature_sets=feature_sets,
        split_name="validation",
    )

    all_scores.append(scores)
    all_preds.append(preds)

    if RUN_TEST:
        test_eval_sets = make_eval_sets(test_base, "test")
        test_scores, test_preds = evaluate_ml_models(
            train_df=train_df,
            eval_sets=test_eval_sets,
            feature_sets=feature_sets,
            split_name="test",
        )
        all_scores.append(test_scores)
        all_preds.append(test_preds)

    scores = pd.concat(all_scores, ignore_index=True)
    scores = scores.sort_values(["split", "holdout_mode", "rmse", "mae"]).reset_index(drop=True)

    preds = pd.concat(all_preds, ignore_index=True)

    scores_path = OUT_DIR / "ml_validation_metrics.csv"
    preds_path = OUT_DIR / "ml_validation_predictions_sample.csv"

    scores.to_csv(scores_path, index=False)

    if SAVE_PREDICTION_SAMPLE_ROWS is not None and len(preds) > SAVE_PREDICTION_SAMPLE_ROWS:
        preds = preds.sample(n=SAVE_PREDICTION_SAMPLE_ROWS, random_state=RANDOM_STATE)

    preds.to_csv(preds_path, index=False)

    print(f"\nSaved metrics:     {scores_path}")
    print(f"Saved predictions: {preds_path}")

    print("\nBest validation results:")
    for mode in HOLDOUT_MODES:
        sub = scores[(scores["split"] == "validation") & (scores["holdout_mode"] == mode)]
        if not sub.empty:
            print("\n" + "=" * 80)
            print(mode)
            print("=" * 80)
            print(sub.head(25).to_string(index=False))

    plot_rmse(scores)
    plot_bias(scores)

    print("\nDone.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()