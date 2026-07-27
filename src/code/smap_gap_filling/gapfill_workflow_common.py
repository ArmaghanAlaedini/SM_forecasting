#!/usr/bin/env python3
"""Shared Python utilities for the SMAP gap-filling validation/test workflow."""

from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_config():
    """Load ``00_config.py`` from the same folder as this module."""
    config_path = Path(__file__).resolve().with_name("00_config.py")
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load configuration from {config_path}")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


def stable_seed(*parts: object, base_seed: int | None = None) -> int:
    """Create a deterministic 32-bit seed from labels and the project seed."""
    seed = cfg.RANDOM_SEED if base_seed is None else int(base_seed)
    text = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def parse_date_from_filename(path: Path) -> pd.Timestamp:
    match = re.search(r"(20\d{6})", path.name)
    if match is None:
        raise ValueError(f"Could not parse YYYYMMDD from filename: {path}")
    return pd.to_datetime(match.group(1), format="%Y%m%d")


def file_id_from_path(pass_name: str, path: Path) -> str:
    return f"{pass_name.lower()}/{path.name}"


def list_complete_files(
    years: Iterable[int] | None = None,
    passes: Iterable[str] | None = None,
) -> list[tuple[str, Path]]:
    """List complete daily SMAP+IEM CSVs, optionally restricted by year/pass."""
    years_set = None if years is None else {int(y) for y in years}
    passes = list(cfg.PASSES if passes is None else passes)
    files: list[tuple[str, Path]] = []
    for pass_name in passes:
        folder = cfg.FULL_SMAP_IEM_DIR / pass_name / "complete"
        if not folder.exists():
            raise FileNotFoundError(f"Missing complete-file directory: {folder}")
        for path in sorted(folder.glob("*.csv")):
            year = int(parse_date_from_filename(path).year)
            if years_set is None or year in years_set:
                files.append((pass_name.lower(), path))
    if not files:
        raise FileNotFoundError(
            f"No complete SMAP+IEM files found under {cfg.FULL_SMAP_IEM_DIR}"
        )
    return files


def add_basic_columns(
    df: pd.DataFrame,
    pass_name: str,
    path: Path,
) -> pd.DataFrame:
    """Normalize identifiers and derive date/pass features."""
    out = df.loc[:, ~df.columns.duplicated()].copy()
    if "date" in out.columns:
        date = pd.to_datetime(out["date"], errors="coerce")
        fallback = parse_date_from_filename(path)
        date = date.fillna(fallback)
    else:
        date = pd.Series(parse_date_from_filename(path), index=out.index)

    out["date"] = pd.to_datetime(date).dt.normalize()
    out["year"] = out["date"].dt.year.astype(int)
    out["month"] = out["date"].dt.month.astype(int)
    out["day_of_year"] = out["date"].dt.dayofyear.astype(int)
    out["sin_doy"] = np.sin(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["cos_doy"] = np.cos(2.0 * np.pi * out["day_of_year"] / 366.0)
    out["pass"] = pass_name.lower()
    out["pass_pm"] = (out["pass"] == "pm").astype(int)
    out["file_id"] = file_id_from_path(pass_name, path)

    if cfg.KEY not in out.columns:
        if {"grid_row", "grid_col"}.issubset(out.columns):
            out[cfg.KEY] = (
                pd.to_numeric(out["grid_row"], errors="coerce")
                .round()
                .astype("Int64")
                .astype(str)
                + "_"
                + pd.to_numeric(out["grid_col"], errors="coerce")
                .round()
                .astype("Int64")
                .astype(str)
            )
        elif {"x", "y"}.issubset(out.columns):
            out[cfg.KEY] = (
                pd.to_numeric(out["x"], errors="coerce")
                .round()
                .astype("Int64")
                .astype(str)
                + "_"
                + pd.to_numeric(out["y"], errors="coerce")
                .round()
                .astype("Int64")
                .astype(str)
            )
        else:
            raise ValueError(
                f"{path} has no {cfg.KEY}, grid_row/grid_col, or x/y columns."
            )
    out[cfg.KEY] = out[cfg.KEY].astype(str)
    return out


def read_complete_file(
    pass_name: str,
    path: Path,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """Read one complete file and add normalized identifiers/features."""
    if usecols is None:
        raw = pd.read_csv(path, low_memory=False)
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        existing = [c for c in usecols if c in header]
        raw = pd.read_csv(path, usecols=existing, low_memory=False)
    return add_basic_columns(raw, pass_name, path)


def feature_groups_from_available(available_columns: Iterable[str]) -> dict[str, list[str]]:
    """Return candidate feature groups restricted to available/generated columns."""
    available = set(available_columns) | {
        "date",
        "year",
        "month",
        "day_of_year",
        "sin_doy",
        "cos_doy",
        "pass",
        "pass_pm",
        "file_id",
    }
    groups: dict[str, list[str]] = {}
    for name, requested in cfg.ML_FEATURE_GROUPS.items():
        cols = list(dict.fromkeys([c for c in requested if c in available]))
        if cols:
            groups[name] = cols
    return groups


def inspect_available_columns(files: list[tuple[str, Path]]) -> set[str]:
    available: set[str] = set()
    for _, path in files:
        available.update(pd.read_csv(path, nrows=0).columns)
    available.update(
        {
            "date",
            "year",
            "month",
            "day_of_year",
            "sin_doy",
            "cos_doy",
            "pass",
            "pass_pm",
            "file_id",
        }
    )
    return available


def resolve_final_feature_columns(
    files: list[tuple[str, Path]],
    strict: bool | None = None,
) -> list[str]:
    """Resolve and optionally require the complete final predictor list."""
    available = inspect_available_columns(files)
    requested = list(cfg.FINAL_ML_FEATURES)
    missing = [c for c in requested if c not in available]
    strict = cfg.STRICT_FINAL_ML_FEATURES if strict is None else strict
    if missing and strict:
        raise ValueError(
            "Final ML predictors are missing from the complete files:\n  - "
            + "\n  - ".join(missing)
            + "\nDo not silently use a different feature set for validation/test/filling."
        )
    used = [c for c in requested if c in available]
    if not used:
        raise ValueError("No final ML features are available.")
    return used


def collect_training_rows(
    files: list[tuple[str, Path]],
    feature_columns: list[str],
    years: Iterable[int] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Collect observed supervised rows from the specified training years."""
    years_set = set(cfg.TRAIN_YEARS if years is None else years)
    needed = list(
        dict.fromkeys(
            ["file_id", "date", "year", "pass", cfg.KEY, cfg.TARGET]
            + feature_columns
        )
    )
    parts: list[pd.DataFrame] = []
    for i, (pass_name, path) in enumerate(files, start=1):
        if int(parse_date_from_filename(path).year) not in years_set:
            continue
        df = read_complete_file(pass_name, path, usecols=needed)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[cfg.TARGET] = pd.to_numeric(df[cfg.TARGET], errors="coerce")
        df = df[df[cfg.TARGET].notna()].copy()
        if not df.empty:
            parts.append(df[needed])
        if i % 250 == 0:
            print(f"  scanned {i:,} files for ML training...")
    if not parts:
        raise RuntimeError(f"No observed training rows found for years {sorted(years_set)}")
    train = pd.concat(parts, ignore_index=True)
    limit = cfg.MAX_ML_TRAIN_ROWS if max_rows is None else max_rows
    if limit is not None and len(train) > limit:
        train = train.sample(n=limit, random_state=cfg.RANDOM_SEED).reset_index(drop=True)
    return train


def load_holdout_manifest(path: Path, holdout_modes: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a shared holdout manifest and validate its unique target keys."""
    if not path.exists():
        raise FileNotFoundError(
            f"Holdout manifest not found: {path}\n"
            "Run 10_generate_holdout_manifests.py first."
        )
    manifest = pd.read_csv(path, low_memory=False)
    required = {
        "split",
        "holdout_mode",
        "date",
        "year",
        "pass",
        "file_id",
        "source_file",
        cfg.KEY,
        "observed",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Holdout manifest missing columns: {sorted(missing)}")
    manifest["date"] = pd.to_datetime(manifest["date"], errors="raise").dt.normalize()
    manifest["pass"] = manifest["pass"].astype(str).str.lower()
    manifest[cfg.KEY] = manifest[cfg.KEY].astype(str)
    manifest["observed"] = pd.to_numeric(manifest["observed"], errors="coerce")
    if holdout_modes is not None:
        manifest = manifest[manifest["holdout_mode"].isin(list(holdout_modes))].copy()
    key_cols = ["split", "holdout_mode", "date", "pass", cfg.KEY]
    duplicated = manifest.duplicated(key_cols, keep=False)
    if duplicated.any():
        example = manifest.loc[duplicated, key_cols].head().to_dict("records")
        raise ValueError(f"Duplicate holdout keys found. Examples: {example}")
    return manifest.sort_values(key_cols).reset_index(drop=True)


def load_manifest_target_rows(
    manifest: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Load exact target rows identified by a shared holdout manifest."""
    groups: list[pd.DataFrame] = []
    needed = list(
        dict.fromkeys(
            [cfg.KEY, cfg.TARGET, "x", "y", "grid_row", "grid_col"]
            + feature_columns
        )
    )
    for (file_id, holdout_mode), sub in manifest.groupby(
        ["file_id", "holdout_mode"], sort=True
    ):
        source = Path(sub["source_file"].iloc[0])
        pass_name = str(sub["pass"].iloc[0])
        df = read_complete_file(pass_name, source, usecols=needed)
        keys = set(sub[cfg.KEY].astype(str))
        target = df[df[cfg.KEY].isin(keys)].copy()
        if len(target) != len(sub):
            missing_keys = sorted(keys - set(target[cfg.KEY]))[:10]
            raise ValueError(
                f"Could not find all holdout keys in {source}. Missing examples: {missing_keys}"
            )
        target = target.merge(
            sub[["split", "holdout_mode", cfg.KEY, "observed"]],
            on=cfg.KEY,
            how="inner",
            validate="one_to_one",
        )
        for col in feature_columns:
            if col not in target.columns:
                target[col] = np.nan
            target[col] = pd.to_numeric(target[col], errors="coerce")
        target[cfg.TARGET] = pd.to_numeric(target[cfg.TARGET], errors="coerce")
        mismatch = ~np.isclose(
            target[cfg.TARGET].to_numpy(float),
            target["observed"].to_numpy(float),
            equal_nan=True,
        )
        if mismatch.any():
            raise ValueError(
                f"Observed values in the manifest do not match {source} for "
                f"{int(mismatch.sum())} rows."
            )
        groups.append(target)
    if not groups:
        raise RuntimeError("No holdout target rows were loaded.")
    return pd.concat(groups, ignore_index=True)


def make_ml_models(model_names: Iterable[str]) -> dict[str, Pipeline]:
    """Construct ML pipelines using the single project-wide settings."""
    model_names = list(model_names)
    models: dict[str, Pipeline] = {}
    settings = cfg.ML_MODEL_SETTINGS

    for name in model_names:
        if name == "random_forest":
            estimator = RandomForestRegressor(
                **settings[name], random_state=cfg.RANDOM_SEED
            )
            models[name] = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
            )
        elif name == "extra_trees":
            estimator = ExtraTreesRegressor(
                **settings[name], random_state=cfg.RANDOM_SEED
            )
            models[name] = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
            )
        elif name == "hist_gbdt":
            estimator = HistGradientBoostingRegressor(
                **settings[name], random_state=cfg.RANDOM_SEED
            )
            models[name] = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
            )
        elif name == "xgboost":
            try:
                from xgboost import XGBRegressor
            except ImportError as exc:
                raise ImportError(
                    "XGBoost is selected but xgboost is not installed in this environment."
                ) from exc
            estimator = XGBRegressor(
                **settings[name], random_state=cfg.RANDOM_SEED
            )
            models[name] = Pipeline(
                [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
            )
        elif name == "ffnn_mlp":
            estimator = MLPRegressor(
                **settings[name], random_state=cfg.RANDOM_SEED
            )
            models[name] = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )
        else:
            raise ValueError(f"Unknown ML model: {name}")
    return models


def clone_models(models: dict[str, Pipeline]) -> dict[str, Pipeline]:
    return {name: clone(model) for name, model in models.items()}


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    """Compute pooled metrics on finite matched observations."""
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    valid = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[valid]
    yp = yp[valid]
    if len(yt) == 0:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "n": 0}
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "bias": float(np.mean(yp - yt)),
        "r2": float(r2_score(yt, yp)) if len(yt) >= 2 else np.nan,
        "n": int(len(yt)),
    }


def add_prediction_metrics(
    predictions: pd.DataFrame,
    method_column: str,
    group_columns: list[str],
    observed_column: str = "observed",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    rows: list[dict] = []
    for keys, sub in predictions.groupby(group_columns + [method_column], dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = group_columns + [method_column]
        row = dict(zip(labels, keys))
        row.update(compute_metrics(sub[observed_column], sub[prediction_column]))
        row["coverage"] = row["n"] / len(sub) if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
