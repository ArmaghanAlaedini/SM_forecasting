#!/usr/bin/env python3
"""Validate candidate ML models on the shared 2024 artificial gaps.

The ML models are trained only on observed 2020--2023 SMAP rows.  Target pixels
are read from ``validation_holdouts_2024.csv``; this script never creates its
own gaps.  Every candidate model/feature set therefore predicts the same 2024
questions used by the GI validation script.

Outputs
-------
* ml_validation_metrics.csv
* ml_validation_predictions.csv (all rows; no row-level sampling)
* ml_feature_sets_used.csv
* figures/*.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapfill_workflow_common import (
    cfg,
    collect_training_rows,
    compute_metrics,
    feature_groups_from_available,
    inspect_available_columns,
    list_complete_files,
    load_holdout_manifest,
    load_manifest_target_rows,
    make_ml_models,
)


OUT_DIR = cfg.ML_VALIDATION_DIR
FIG_DIR = OUT_DIR / "figures"
PREDICTION_PATH = OUT_DIR / "ml_validation_predictions.csv"
METRICS_PATH = OUT_DIR / "ml_validation_metrics.csv"
FEATURE_SET_PATH = OUT_DIR / "ml_feature_sets_used.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def usable_features(train: pd.DataFrame, requested: list[str]) -> list[str]:
    """Keep predictors with enough finite training values and nonzero variation."""
    usable: list[str] = []
    for col in requested:
        if col not in train.columns:
            continue
        values = pd.to_numeric(train[col], errors="coerce")
        if values.notna().sum() < 20:
            continue
        # Keep constant predictors so the validated feature contract is identical
        # in 2024, 2025, and production. Tree/linear pipelines can safely ignore
        # a zero-variance column.
        usable.append(col)
    return usable


def initialize_prediction_file() -> None:
    if PREDICTION_PATH.exists():
        PREDICTION_PATH.unlink()
    columns = [
        "split",
        "holdout_mode",
        "date",
        "year",
        "pass",
        "file_id",
        cfg.KEY,
        "observed",
        "x",
        "y",
        "model",
        "feature_group",
        "n_features",
        "features",
        "prediction",
    ]
    pd.DataFrame(columns=columns).to_csv(PREDICTION_PATH, index=False)


def append_predictions(frame: pd.DataFrame) -> None:
    frame.to_csv(PREDICTION_PATH, mode="a", header=False, index=False)


def predict_one_configuration(
    train: pd.DataFrame,
    targets: pd.DataFrame,
    model_name: str,
    feature_group: str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    model = make_ml_models([model_name])[model_name]
    model.fit(train[feature_columns], train[cfg.TARGET].to_numpy(dtype=float))
    prediction = np.asarray(model.predict(targets[feature_columns]), dtype=float)

    out = targets[
        [
            "split",
            "holdout_mode",
            "date",
            "year",
            "pass",
            "file_id",
            cfg.KEY,
            "observed",
            "x",
            "y",
        ]
    ].copy()
    out["model"] = model_name
    out["feature_group"] = feature_group
    out["n_features"] = len(feature_columns)
    out["features"] = ";".join(feature_columns)
    out["prediction"] = prediction

    metric_rows: list[dict] = []
    for holdout_mode, sub in out.groupby("holdout_mode", sort=True):
        metric_rows.append(
            {
                "split": "validation",
                "holdout_mode": holdout_mode,
                "feature_group": feature_group,
                "model": model_name,
                "n_features": len(feature_columns),
                "features": ";".join(feature_columns),
                **compute_metrics(sub["observed"], sub["prediction"]),
                "coverage": float(np.isfinite(sub["prediction"]).mean()),
            }
        )
    return out, metric_rows


def plot_metric(metrics: pd.DataFrame, metric: str) -> None:
    for holdout_mode in cfg.HOLDOUT_MODES:
        sub = metrics[metrics["holdout_mode"].eq(holdout_mode)].copy()
        if sub.empty:
            continue
        sub["label"] = sub["model"] + " | " + sub["feature_group"]
        sub = sub.sort_values([metric, "mae"], ascending=True).head(30)
        fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(sub))))
        ax.barh(sub["label"], sub[metric])
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_title(f"ML validation {metric.upper()}: {holdout_mode}")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = FIG_DIR / f"ml_validation_{metric}_{holdout_mode}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("10a: ML validation on shared 2024 holdouts")
    print("=" * 78)
    print(f"Training years:   {cfg.TRAIN_YEARS}")
    print(f"Validation years: {cfg.VALIDATION_YEARS}")
    print(f"Seed:             {cfg.RANDOM_SEED}")
    print(f"Holdout manifest: {cfg.VALIDATION_HOLDOUT_PATH}")
    print("=" * 78)

    files = list_complete_files(years=cfg.TRAIN_YEARS + cfg.VALIDATION_YEARS)
    available = inspect_available_columns(files)
    feature_groups = feature_groups_from_available(available)

    feature_rows = [
        {"feature_group": group, "feature": feature, "position": position}
        for group, features in feature_groups.items()
        for position, feature in enumerate(features, start=1)
    ]
    pd.DataFrame(feature_rows).to_csv(FEATURE_SET_PATH, index=False)

    union_features = list(
        dict.fromkeys(feature for features in feature_groups.values() for feature in features)
    )
    train = collect_training_rows(
        files,
        union_features,
        years=cfg.TRAIN_YEARS,
        max_rows=cfg.MAX_ML_TRAIN_ROWS,
    )

    manifest = load_holdout_manifest(cfg.VALIDATION_HOLDOUT_PATH)
    targets = load_manifest_target_rows(manifest, union_features)

    # Guarantee numeric model inputs.
    for feature in union_features:
        train[feature] = pd.to_numeric(train[feature], errors="coerce")
        targets[feature] = pd.to_numeric(targets[feature], errors="coerce")

    print(f"ML training rows:       {len(train):,}")
    print(f"2024 validation targets:{len(targets):,}")
    print("Targets by holdout mode:")
    print(targets.groupby("holdout_mode").size())

    initialize_prediction_file()
    all_metric_rows: list[dict] = []

    # Baseline is useful for interpretation but is not a stacking base learner.
    train_mean = float(train[cfg.TARGET].mean())
    baseline = targets[
        [
            "split",
            "holdout_mode",
            "date",
            "year",
            "pass",
            "file_id",
            cfg.KEY,
            "observed",
            "x",
            "y",
        ]
    ].copy()
    baseline["model"] = "baseline"
    baseline["feature_group"] = "baseline_train_mean"
    baseline["n_features"] = 0
    baseline["features"] = ""
    baseline["prediction"] = train_mean
    append_predictions(baseline)
    for mode, sub in baseline.groupby("holdout_mode"):
        all_metric_rows.append(
            {
                "split": "validation",
                "holdout_mode": mode,
                "feature_group": "baseline_train_mean",
                "model": "baseline",
                "n_features": 0,
                "features": "",
                **compute_metrics(sub["observed"], sub["prediction"]),
                "coverage": 1.0,
            }
        )

    for feature_group, requested in feature_groups.items():
        features = usable_features(train, requested)
        if not features:
            print(f"[skip] {feature_group}: no usable features")
            continue
        if feature_group == cfg.FINAL_ML_FEATURE_GROUP and features != requested:
            missing = [c for c in requested if c not in features]
            raise ValueError(
                "The selected final feature group changed during validation. "
                f"Unusable predictors: {missing}"
            )
        print("\n" + "-" * 78)
        print(f"Feature group: {feature_group} ({len(features)} predictors)")
        for model_name in cfg.CANDIDATE_ML_MODELS:
            print(f"  fitting {model_name}...")
            predictions, metric_rows = predict_one_configuration(
                train,
                targets,
                model_name,
                feature_group,
                features,
            )
            append_predictions(predictions)
            all_metric_rows.extend(metric_rows)

    metrics = pd.DataFrame(all_metric_rows).sort_values(
        ["holdout_mode", "rmse", "mae", "model"]
    )
    metrics.to_csv(METRICS_PATH, index=False)

    plot_metric(metrics, "rmse")
    plot_metric(metrics, "bias")

    print("\nSaved:")
    print(f"  {METRICS_PATH}")
    print(f"  {PREDICTION_PATH}")
    print(f"  {FEATURE_SET_PATH}")
    print("\nBest spatial-block results:")
    print(
        metrics[metrics["holdout_mode"].eq("spatial_block")]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
