#!/usr/bin/env python3
"""Train selected ML models and predict original missing SMAP pixels.

The production ML models use exactly the same predictors, hyperparameters, and
2020--2023 training period used in validation and independent testing.  The
script writes one long prediction table; it does not alter observed SMAP values
or finalize gap-filled files.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from gapfill_workflow_common import (
    cfg,
    collect_training_rows,
    file_id_from_path,
    list_complete_files,
    make_ml_models,
    parse_date_from_filename,
    read_complete_file,
    resolve_final_feature_columns,
)


OUT_DIR = cfg.PREDICTION_DIR / "ml"
PREDICTION_PATH = OUT_DIR / "ml_gapfill_predictions.csv"
FEATURE_MANIFEST_PATH = OUT_DIR / "ml_gapfill_feature_manifest.csv"
RUN_MANIFEST_PATH = OUT_DIR / "ml_gapfill_run_manifest.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COLUMNS = [
    "file_id",
    "date",
    "year",
    "pass",
    cfg.KEY,
    "model",
    "feature_group",
    "n_features",
    "features",
    "prediction",
    "source_file",
]


def initialize_output() -> None:
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(PREDICTION_PATH, index=False)


def append_output(frame: pd.DataFrame) -> None:
    frame[OUTPUT_COLUMNS].to_csv(
        PREDICTION_PATH,
        mode="a",
        header=False,
        index=False,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    print("11a: Generate ML predictions for original SMAP gaps")
    print("=" * 78)
    print(f"Training years: {cfg.TRAIN_YEARS}")
    print(f"Gap-fill years:{cfg.GAPFILL_YEARS}")
    print(f"Seed:          {cfg.RANDOM_SEED}")
    print(f"Models:        {cfg.SELECTED_ML_MODELS}")
    print("=" * 78)

    files = list_complete_files(years=cfg.GAPFILL_YEARS)
    feature_columns = resolve_final_feature_columns(files, strict=True)
    if feature_columns != list(cfg.FINAL_ML_FEATURES):
        raise ValueError(
            "Production ML feature columns differ from the validated final list."
        )

    pd.DataFrame(
        {
            "feature": feature_columns,
            "position": range(1, len(feature_columns) + 1),
            "feature_group": cfg.FINAL_ML_FEATURE_GROUP,
            "project_seed": cfg.RANDOM_SEED,
        }
    ).to_csv(FEATURE_MANIFEST_PATH, index=False)

    train = collect_training_rows(
        files,
        feature_columns,
        years=cfg.TRAIN_YEARS,
        max_rows=cfg.MAX_ML_TRAIN_ROWS,
    )
    for feature in feature_columns:
        train[feature] = pd.to_numeric(train[feature], errors="coerce")

    models = make_ml_models(cfg.SELECTED_ML_MODELS)
    for model_name, model in models.items():
        print(f"Training {model_name} on {len(train):,} observed rows...")
        model.fit(train[feature_columns], train[cfg.TARGET].to_numpy(dtype=float))

    initialize_output()
    manifest_rows: list[dict] = []
    total_missing = 0
    total_predictions = 0

    needed = list(dict.fromkeys([cfg.KEY, cfg.TARGET, *feature_columns]))
    for index, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)
        if int(date.year) not in cfg.GAPFILL_YEARS:
            continue

        df = read_complete_file(pass_name, path, usecols=needed)
        for feature in feature_columns:
            if feature not in df.columns:
                raise ValueError(f"Required production feature {feature} missing in {path}")
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[cfg.TARGET] = pd.to_numeric(df[cfg.TARGET], errors="coerce")
        missing = df[df[cfg.TARGET].isna()].copy()

        n_missing = len(missing)
        n_written = 0
        if n_missing:
            base = missing[["file_id", "date", "year", "pass", cfg.KEY]].copy()
            base["source_file"] = str(path.resolve())
            for model_name, model in models.items():
                out = base.copy()
                out["model"] = model_name
                out["feature_group"] = cfg.FINAL_ML_FEATURE_GROUP
                out["n_features"] = len(feature_columns)
                out["features"] = ";".join(feature_columns)
                out["prediction"] = np.asarray(
                    model.predict(missing[feature_columns]), dtype=float
                )
                append_output(out)
                n_written += len(out)

        manifest_rows.append(
            {
                "file_id": file_id_from_path(pass_name, path),
                "date": date.date().isoformat(),
                "year": int(date.year),
                "pass": pass_name,
                "source_file": str(path.resolve()),
                "n_rows": len(df),
                "n_original_missing": n_missing,
                "n_models": len(models),
                "n_prediction_rows": n_written,
                "project_seed": cfg.RANDOM_SEED,
            }
        )
        total_missing += n_missing
        total_predictions += n_written
        if index % 100 == 0:
            print(
                f"  processed {index:,} files; original gaps={total_missing:,}; "
                f"prediction rows={total_predictions:,}"
            )

    pd.DataFrame(manifest_rows).to_csv(RUN_MANIFEST_PATH, index=False)
    print("\nSaved:")
    print(f"  {PREDICTION_PATH}")
    print(f"  {FEATURE_MANIFEST_PATH}")
    print(f"  {RUN_MANIFEST_PATH}")
    print(f"\nOriginal missing pixels: {total_missing:,}")
    print(f"ML prediction rows:       {total_predictions:,}")


if __name__ == "__main__":
    main()
