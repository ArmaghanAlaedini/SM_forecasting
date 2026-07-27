#!/usr/bin/env python3
"""Independent 2025 test of the selected ML base models.

The selected ML models are trained on observed 2020--2023 SMAP rows using the
same feature list and hyperparameters used in validation.  They predict the
exact target keys in ``test_holdouts_2025.csv``.  This script never selects
models and never creates its own gaps.
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapfill_workflow_common import (
    cfg,
    collect_training_rows,
    compute_metrics,
    list_complete_files,
    load_holdout_manifest,
    load_manifest_target_rows,
    make_ml_models,
    resolve_final_feature_columns,
)


OUT_DIR = cfg.ML_TEST_DIR
FIG_DIR = OUT_DIR / "figures"
PREDICTION_PATH = OUT_DIR / "ml_selected_test_predictions.csv"
METRICS_PATH = OUT_DIR / "ml_selected_test_metrics.csv"
FEATURE_MANIFEST_PATH = OUT_DIR / "ml_selected_test_feature_manifest.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_metric(metrics: pd.DataFrame, metric: str) -> None:
    sub = metrics.sort_values(["holdout_mode", metric]).copy()
    if sub.empty:
        return
    sub["label"] = sub["model"] + " | " + sub["holdout_mode"]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(sub))))
    ax.barh(sub["label"], sub[metric])
    ax.invert_yaxis()
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Selected ML 2025 independent test: {metric.upper()}")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"ml_selected_test_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("10d: Independent 2025 test of selected ML models")
    print("=" * 78)
    print(f"Training years: {cfg.TRAIN_YEARS}")
    print(f"Test year:      {cfg.TEST_YEAR}")
    print(f"Seed:           {cfg.RANDOM_SEED}")
    print(f"Holdouts:       {cfg.TEST_HOLDOUT_PATH}")
    print("=" * 78)

    files = list_complete_files(years=cfg.TRAIN_YEARS + cfg.TEST_YEARS)
    feature_columns = resolve_final_feature_columns(files, strict=True)
    if feature_columns != list(cfg.FINAL_ML_FEATURES):
        raise ValueError(
            "The 2025 ML test feature list does not exactly match the validated "
            "final feature list."
        )

    train = collect_training_rows(
        files,
        feature_columns,
        years=cfg.TRAIN_YEARS,
        max_rows=cfg.MAX_ML_TRAIN_ROWS,
    )
    manifest = load_holdout_manifest(cfg.TEST_HOLDOUT_PATH)
    targets = load_manifest_target_rows(manifest, feature_columns)

    for feature in feature_columns:
        train[feature] = pd.to_numeric(train[feature], errors="coerce")
        targets[feature] = pd.to_numeric(targets[feature], errors="coerce")

    pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_group": cfg.FINAL_ML_FEATURE_GROUP,
            "position": range(1, len(feature_columns) + 1),
        }
    ).to_csv(FEATURE_MANIFEST_PATH, index=False)

    models = make_ml_models(cfg.SELECTED_ML_MODELS)
    predictions: list[pd.DataFrame] = []
    metric_rows: list[dict] = []

    base_columns = [
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

    for model_name, model in models.items():
        print(f"Training {model_name} on {len(train):,} rows...")
        model.fit(train[feature_columns], train[cfg.TARGET].to_numpy(dtype=float))
        prediction = np.asarray(model.predict(targets[feature_columns]), dtype=float)

        out = targets[base_columns].copy()
        out["model"] = model_name
        out["feature_group"] = cfg.FINAL_ML_FEATURE_GROUP
        out["n_features"] = len(feature_columns)
        out["features"] = ";".join(feature_columns)
        out["prediction"] = prediction
        predictions.append(out)

        for holdout_mode, sub in out.groupby("holdout_mode", sort=True):
            metric_rows.append(
                {
                    "split": "test",
                    "holdout_mode": holdout_mode,
                    "feature_group": cfg.FINAL_ML_FEATURE_GROUP,
                    "model": model_name,
                    "n_features": len(feature_columns),
                    "features": ";".join(feature_columns),
                    **compute_metrics(sub["observed"], sub["prediction"]),
                    "coverage": float(np.isfinite(sub["prediction"]).mean()),
                }
            )

    prediction_table = pd.concat(predictions, ignore_index=True).sort_values(
        ["holdout_mode", "date", "pass", cfg.KEY, "model"]
    )
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["holdout_mode", "rmse", "mae", "model"]
    )
    prediction_table.to_csv(PREDICTION_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    plot_metric(metrics, "rmse")
    plot_metric(metrics, "bias")

    print("\nSaved:")
    print(f"  {PREDICTION_PATH}")
    print(f"  {METRICS_PATH}")
    print(f"  {FEATURE_MANIFEST_PATH}")
    print("\n2025 ML test metrics:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
