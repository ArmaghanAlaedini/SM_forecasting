#!/usr/bin/env python3
"""Create final gap-filled daily SMAP files from aligned base predictions.

A ridge-stacking prediction is used only when all six required base predictions
and all context features are finite.  Incomplete rows are sent to the explicit
fallback waterfall; missing base predictions are never median-imputed.

The large ML/GI prediction CSVs are streamed once into temporary per-file
pieces, then only one retrieval is held in memory at a time.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gapfill_workflow_common import (
    add_basic_columns,
    cfg,
    file_id_from_path,
    list_complete_files,
    parse_date_from_filename,
)


ML_PREDICTION_PATH = cfg.PREDICTION_DIR / "ml" / "ml_gapfill_predictions.csv"
GI_PREDICTION_PATH = (
    cfg.PREDICTION_DIR / "interpolation" / "interpolation_gapfill_predictions.csv"
)
SUMMARY_BY_FILE_PATH = cfg.FINAL_DIR / "gapfill_summary_by_file.csv"
OVERALL_SUMMARY_PATH = cfg.FINAL_DIR / "gapfill_overall_summary.csv"

SPLIT_ROOT = cfg.PREDICTION_DIR / "_split_by_file_11c"
ML_SPLIT_DIR = SPLIT_ROOT / "ml"
GI_SPLIT_DIR = SPLIT_ROOT / "interpolation"
SPLIT_CHUNK_ROWS = 2_000_000

for pass_name in cfg.PASSES:
    (cfg.FINAL_DIR / pass_name).mkdir(parents=True, exist_ok=True)


def safe_file_id(file_id: str) -> str:
    return str(file_id).replace("/", "__").replace("\\", "__")


def prediction_column(method: str) -> str:
    return f"pred_{method}"


def load_meta_bundle() -> dict | None:
    if not cfg.META_MODEL_PATH.exists():
        print(f"Meta-model not found: {cfg.META_MODEL_PATH}; using fallbacks only.")
        return None
    bundle = joblib.load(cfg.META_MODEL_PATH)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError(
            "Saved meta-model has the old format. Re-run 10f and 10g with the fixed scripts."
        )
    if bundle.get("feature_columns") != list(cfg.META_FEATURE_COLUMNS):
        raise ValueError(
            "Saved meta-model feature order differs from 00_config.py. "
            "Re-run 10f and 10g."
        )
    return bundle


def split_prediction_file(
    source: Path,
    output_dir: Path,
    required_columns: list[str],
    optional_columns: list[str],
    label: str,
) -> int:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        print(f"Warning: {label} prediction file not found: {source}")
        return 0

    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [c for c in required_columns if c not in header]
    if missing:
        raise ValueError(f"{label} prediction file missing columns: {missing}")
    usecols = required_columns + [c for c in optional_columns if c in header]

    seen: set[str] = set()
    total = 0
    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        chunksize=SPLIT_CHUNK_ROWS,
        low_memory=False,
    ):
        total += len(chunk)
        for file_id, group in chunk.groupby("file_id", sort=False):
            destination = output_dir / f"{safe_file_id(file_id)}.csv"
            write_header = destination.name not in seen
            group.to_csv(destination, mode="a", header=write_header, index=False)
            seen.add(destination.name)
        print(f"  {label}: {total:,} rows; {len(seen):,} retrieval files")
    return len(seen)


def prepare_prediction_index() -> tuple[int, int]:
    ml_count = split_prediction_file(
        ML_PREDICTION_PATH,
        ML_SPLIT_DIR,
        required_columns=["file_id", cfg.KEY, "model", "prediction"],
        optional_columns=[],
        label="ML",
    )
    gi_count = split_prediction_file(
        GI_PREDICTION_PATH,
        GI_SPLIT_DIR,
        required_columns=["file_id", cfg.KEY, "method", "prediction"],
        optional_columns=[
            "kriging_variance",
            "nearest_distance_m",
            "prediction_status",
        ],
        label="GI",
    )
    return ml_count, gi_count


def read_ml_predictions(file_id: str) -> pd.DataFrame:
    path = ML_SPLIT_DIR / f"{safe_file_id(file_id)}.csv"
    columns = ["file_id", cfg.KEY] + [
        prediction_column(model) for model in cfg.SELECTED_ML_MODELS
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, low_memory=False)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    wide = frame.pivot_table(
        index=["file_id", cfg.KEY],
        columns="model",
        values="prediction",
        aggfunc="first",
        dropna=False,
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={model: prediction_column(model) for model in cfg.SELECTED_ML_MODELS}
    )
    for column in columns:
        if column not in wide.columns:
            wide[column] = np.nan
    return wide[columns]


def read_gi_predictions(file_id: str) -> pd.DataFrame:
    path = GI_SPLIT_DIR / f"{safe_file_id(file_id)}.csv"
    base_columns = ["file_id", cfg.KEY] + [
        prediction_column(method) for method in cfg.SELECTED_INTERPOLATION_METHODS
    ]
    if not path.exists():
        return pd.DataFrame(columns=base_columns)
    frame = pd.read_csv(path, low_memory=False)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    wide = frame.pivot_table(
        index=["file_id", cfg.KEY],
        columns="method",
        values="prediction",
        aggfunc="first",
        dropna=False,
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={
            method: prediction_column(method)
            for method in cfg.SELECTED_INTERPOLATION_METHODS
        }
    )

    diagnostics = [
        (
            "centroid_ordinary_kriging",
            "kriging_variance",
            "kriging_variance_centroid_ok",
        ),
        ("regression_kriging", "kriging_variance", "kriging_variance_rk"),
        (
            "nearest_neighbor_same_day",
            "nearest_distance_m",
            "nearest_distance_m_nn",
        ),
    ]
    for method, source_column, output_column in diagnostics:
        if source_column in frame.columns:
            sub = (
                frame[frame["method"].eq(method)][
                    ["file_id", cfg.KEY, source_column]
                ]
                .drop_duplicates(["file_id", cfg.KEY])
                .rename(columns={source_column: output_column})
            )
            wide = wide.merge(
                sub,
                on=["file_id", cfg.KEY],
                how="left",
                validate="one_to_one",
            )

    for column in base_columns:
        if column not in wide.columns:
            wide[column] = np.nan
    return wide


def predictions_for_file(file_id: str) -> pd.DataFrame:
    ml = read_ml_predictions(file_id)
    gi = read_gi_predictions(file_id)
    return gi.merge(ml, on=["file_id", cfg.KEY], how="outer", validate="one_to_one")


def waterfall_value(row: pd.Series, methods: list[str]) -> tuple[float, str]:
    for method in methods:
        column = prediction_column(method)
        if column in row.index:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.notna(value) and np.isfinite(float(value)):
                return float(value), method
    return np.nan, "none"


def fill_missing_rows(
    missing: pd.DataFrame,
    model_bundle: dict | None,
    fallback_methods: list[str],
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    n = len(missing)
    values = np.full(n, np.nan)
    statuses = ["unfilled"] * n
    methods = ["none"] * n
    stacking_eligible = np.zeros(n, dtype=bool)

    if model_bundle is not None and n:
        meta = missing.reindex(columns=cfg.META_FEATURE_COLUMNS).apply(
            pd.to_numeric, errors="coerce"
        )
        stacking_eligible = np.isfinite(meta.to_numpy(dtype=float)).all(axis=1)
        if stacking_eligible.any():
            prediction = np.asarray(
                model_bundle["pipeline"].predict(meta.loc[stacking_eligible]),
                dtype=float,
            )
            eligible_indices = np.where(stacking_eligible)[0]
            finite = np.isfinite(prediction)
            for local_position, global_position in enumerate(eligible_indices):
                if finite[local_position]:
                    values[global_position] = prediction[local_position]
                    statuses[global_position] = "filled"
                    methods[global_position] = "stacking"

    for index in np.where(~np.isfinite(values))[0]:
        value, method = waterfall_value(missing.iloc[index], fallback_methods)
        if np.isfinite(value):
            values[index] = value
            statuses[index] = "filled"
            methods[index] = method

    return values, statuses, methods, stacking_eligible


def output_filename(input_path: Path) -> str:
    return input_path.name.replace("_complete_", "_gapfilled_").replace(
        "complete", "gapfilled"
    )


def main() -> None:
    bundle = load_meta_bundle()
    fallback_methods = list(
        dict.fromkeys([cfg.FINAL_PRIMARY_METHOD, *cfg.FINAL_FALLBACK_METHODS])
    )

    print("11c: Finalize SMAP gap-filled files")
    print("=" * 78)
    print(f"Project seed: {cfg.RANDOM_SEED}")
    print(f"Stacking:     {'enabled' if bundle is not None else 'disabled'}")
    print(f"Fallbacks:    {fallback_methods}")
    print("=" * 78)

    ml_count, gi_count = prepare_prediction_index()
    print(f"Indexed ML retrievals: {ml_count:,}")
    print(f"Indexed GI retrievals: {gi_count:,}")

    summaries: list[dict] = []
    files = list_complete_files(years=cfg.GAPFILL_YEARS)
    for index, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)
        frame = pd.read_csv(path, low_memory=False)
        frame = add_basic_columns(frame, pass_name, path)
        file_id = file_id_from_path(pass_name, path)
        predictions = predictions_for_file(file_id)
        merged = frame.merge(
            predictions,
            on=["file_id", cfg.KEY],
            how="left",
            validate="one_to_one",
        )

        target = pd.to_numeric(merged[cfg.TARGET], errors="coerce")
        observed_mask = target.notna().to_numpy()
        missing_mask = ~observed_mask
        filled = target.to_numpy(dtype=float)
        fill_status = np.where(observed_mask, "observed", "unfilled").astype(object)
        fill_method = np.where(observed_mask, "observed", "none").astype(object)
        stack_eligible_global = np.zeros(len(merged), dtype=bool)

        if missing_mask.any():
            missing_rows = merged.loc[missing_mask].copy()
            values, statuses, methods, eligible = fill_missing_rows(
                missing_rows,
                bundle,
                fallback_methods,
            )
            global_indices = np.where(missing_mask)[0]
            filled = np.array(filled, dtype=float, copy=True)
            filled[global_indices] = values
            fill_status[global_indices] = statuses
            fill_method[global_indices] = methods
            stack_eligible_global[global_indices] = eligible

        merged["soil_moisture_filled"] = filled
        merged["fill_status"] = fill_status
        merged["fill_method"] = fill_method
        merged["stacking_eligible"] = stack_eligible_global

        if cfg.CLIP_FILLED_VALUES:
            mask = merged["fill_status"].eq("filled")
            merged.loc[mask, "soil_moisture_filled"] = merged.loc[
                mask, "soil_moisture_filled"
            ].clip(cfg.CLIP_MIN, cfg.CLIP_MAX)

        output_path = cfg.FINAL_DIR / pass_name / output_filename(path)
        merged.to_csv(output_path, index=False)

        counts = merged["fill_method"].value_counts(dropna=False).to_dict()
        summary = {
            "file_id": file_id,
            "date": date.date().isoformat(),
            "year": int(date.year),
            "pass": pass_name,
            "source_file": str(path.resolve()),
            "output_file": str(output_path.resolve()),
            "n_rows": len(merged),
            "n_observed_original": int(observed_mask.sum()),
            "n_missing_original": int(missing_mask.sum()),
            "n_stacking_eligible": int(stack_eligible_global.sum()),
            "n_filled": int(merged["fill_status"].eq("filled").sum()),
            "n_unfilled": int(merged["fill_status"].eq("unfilled").sum()),
            "stacking_enabled": bundle is not None,
            "min_soil_moisture_filled": pd.to_numeric(
                merged["soil_moisture_filled"], errors="coerce"
            ).min(),
            "max_soil_moisture_filled": pd.to_numeric(
                merged["soil_moisture_filled"], errors="coerce"
            ).max(),
        }
        for method, count in counts.items():
            summary[f"fill_method_count__{method}"] = int(count)
        summaries.append(summary)

        if index == 1 or index % 100 == 0:
            print(
                f"  [{index}] {date.date()} {pass_name.upper()} | "
                f"missing={summary['n_missing_original']} "
                f"stack-eligible={summary['n_stacking_eligible']} "
                f"stacked={counts.get('stacking', 0)} "
                f"unfilled={summary['n_unfilled']}"
            )

    summary_by_file = pd.DataFrame(summaries)
    summary_by_file.to_csv(SUMMARY_BY_FILE_PATH, index=False)
    method_columns = [
        column for column in summary_by_file if column.startswith("fill_method_count__")
    ]
    overall = {
        "n_files": len(summary_by_file),
        "n_rows": int(summary_by_file["n_rows"].sum()),
        "n_observed_original": int(summary_by_file["n_observed_original"].sum()),
        "n_missing_original": int(summary_by_file["n_missing_original"].sum()),
        "n_stacking_eligible": int(summary_by_file["n_stacking_eligible"].sum()),
        "n_filled": int(summary_by_file["n_filled"].sum()),
        "n_unfilled": int(summary_by_file["n_unfilled"].sum()),
        "stacking_enabled": bundle is not None,
        "project_seed": cfg.RANDOM_SEED,
        "fallback_methods": ";".join(fallback_methods),
        "clip_filled_values": cfg.CLIP_FILLED_VALUES,
    }
    for column in method_columns:
        overall[column] = int(summary_by_file[column].fillna(0).sum())
    pd.DataFrame([overall]).to_csv(OVERALL_SUMMARY_PATH, index=False)

    if SPLIT_ROOT.exists():
        shutil.rmtree(SPLIT_ROOT)

    print("\nSaved:")
    print(f"  {SUMMARY_BY_FILE_PATH}")
    print(f"  {OVERALL_SUMMARY_PATH}")
    print("\nOverall summary:")
    for key, value in overall.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
