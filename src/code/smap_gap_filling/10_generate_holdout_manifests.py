#!/usr/bin/env python3
"""Create the single shared artificial-gap manifests for 2024 and 2025.

Why this script exists
----------------------
Every ML and GI method must predict exactly the same hidden SMAP pixels.  This
script chooses those pixels once, using seed 1234, and saves them to CSV.  The
later Python and R scripts read these files; they do not generate new gaps.

Outputs
-------
* 05_gapfill_model_validation/holdouts/validation_holdouts_2024.csv
* 06_selected_methods_test/holdouts/test_holdouts_2025.csv
* support/holdout_manifest_summary.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gapfill_workflow_common import (
    add_basic_columns,
    cfg,
    file_id_from_path,
    list_complete_files,
    parse_date_from_filename,
    stable_seed,
)


BASE_COLUMNS = [
    cfg.KEY,
    cfg.TARGET,
    "x",
    "y",
    "grid_row",
    "grid_col",
]


def read_observed(pass_name: str, path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in BASE_COLUMNS if c in header]
    raw = pd.read_csv(path, usecols=usecols, low_memory=False)
    df = add_basic_columns(raw, pass_name, path)
    df[cfg.TARGET] = pd.to_numeric(df[cfg.TARGET], errors="coerce")
    for col in ["x", "y", "grid_row", "grid_col"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df[cfg.TARGET].notna()].copy()


def random_cell_indices(obs: pd.DataFrame, split: str, pass_name: str, date: pd.Timestamp) -> np.ndarray:
    n = len(obs)
    desired = max(cfg.MIN_HOLDOUT_ROWS, int(round(cfg.RANDOM_CELL_HOLDOUT_FRACTION * n)))
    desired = min(desired, n - cfg.MIN_DONOR_ROWS)
    if desired <= 0:
        return np.array([], dtype=int)
    rng = np.random.default_rng(stable_seed(split, pass_name, date.date(), "random_cell"))
    return np.sort(rng.choice(obs.index.to_numpy(), size=desired, replace=False))


def make_rank_bins(values: pd.Series, n_bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    valid = numeric.notna()
    if valid.sum() < n_bins:
        return result
    ranks = numeric.loc[valid].rank(method="first")
    try:
        bins = pd.qcut(ranks, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return result
    result.loc[valid] = pd.Series(bins, index=ranks.index).astype("Int64")
    return result


def spatial_block_indices(obs: pd.DataFrame, split: str, pass_name: str, date: pd.Timestamp) -> tuple[np.ndarray, str]:
    """Choose one contiguous 2x2 block from a 4x4 quantile grid."""
    work = obs.copy()
    if {"grid_row", "grid_col"}.issubset(work.columns):
        row_source = work["grid_row"]
        col_source = work["grid_col"]
    elif {"y", "x"}.issubset(work.columns):
        row_source = work["y"]
        col_source = work["x"]
    else:
        return random_cell_indices(obs, split, pass_name, date), "random_fallback_no_coordinates"

    work["_row_bin"] = make_rank_bins(row_source, cfg.SPATIAL_BLOCK_N_BINS)
    work["_col_bin"] = make_rank_bins(col_source, cfg.SPATIAL_BLOCK_N_BINS)
    valid = work["_row_bin"].notna() & work["_col_bin"].notna()
    work = work.loc[valid].copy()
    if len(work) < cfg.MIN_OBSERVED_ROWS_PER_RETRIEVAL:
        return random_cell_indices(obs, split, pass_name, date), "random_fallback_sparse_coordinates"

    desired = max(cfg.MIN_HOLDOUT_ROWS, int(round(cfg.RANDOM_CELL_HOLDOUT_FRACTION * len(obs))))
    max_holdout = len(obs) - cfg.MIN_DONOR_ROWS
    width = cfg.SPATIAL_BLOCK_WIDTH_BINS
    n_bins = cfg.SPATIAL_BLOCK_N_BINS

    candidates: list[tuple[int, int, np.ndarray]] = []
    for row_start in range(0, n_bins - width + 1):
        for col_start in range(0, n_bins - width + 1):
            row_bins = set(range(row_start, row_start + width))
            col_bins = set(range(col_start, col_start + width))
            mask = work["_row_bin"].isin(row_bins) & work["_col_bin"].isin(col_bins)
            idx = work.index[mask].to_numpy()
            if cfg.MIN_HOLDOUT_ROWS <= len(idx) <= max_holdout:
                candidates.append((abs(len(idx) - desired), len(idx), idx))

    if not candidates:
        return random_cell_indices(obs, split, pass_name, date), "random_fallback_no_eligible_block"

    best_score = min(score for score, _, _ in candidates)
    best = [item for item in candidates if item[0] == best_score]
    rng = np.random.default_rng(stable_seed(split, pass_name, date.date(), "spatial_block"))
    chosen = best[int(rng.integers(0, len(best)))]
    return np.sort(chosen[2]), "contiguous_quantile_block"


def build_manifest(year: int, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for pass_name, path in list_complete_files(years=[year]):
        date = parse_date_from_filename(path)
        obs = read_observed(pass_name, path)
        n_observed = len(obs)
        if n_observed < cfg.MIN_OBSERVED_ROWS_PER_RETRIEVAL:
            summaries.append(
                {
                    "split": split,
                    "year": year,
                    "date": date.date().isoformat(),
                    "pass": pass_name,
                    "file_id": file_id_from_path(pass_name, path),
                    "holdout_mode": "none",
                    "n_observed": n_observed,
                    "n_holdout": 0,
                    "n_donors": n_observed,
                    "status": "skipped_too_few_observed",
                }
            )
            continue

        mode_indices: dict[str, tuple[np.ndarray, str]] = {
            "random_cell": (
                random_cell_indices(obs, split, pass_name, date),
                "random_cell",
            ),
            "spatial_block": spatial_block_indices(obs, split, pass_name, date),
        }

        for holdout_mode, (indices, generation) in mode_indices.items():
            if len(indices) == 0:
                summaries.append(
                    {
                        "split": split,
                        "year": year,
                        "date": date.date().isoformat(),
                        "pass": pass_name,
                        "file_id": file_id_from_path(pass_name, path),
                        "holdout_mode": holdout_mode,
                        "n_observed": n_observed,
                        "n_holdout": 0,
                        "n_donors": n_observed,
                        "status": "skipped_no_holdout",
                    }
                )
                continue

            target = obs.loc[indices].copy()
            def numeric_or_nan(column: str) -> np.ndarray:
                if column not in target.columns:
                    return np.full(len(target), np.nan)
                return pd.to_numeric(target[column], errors="coerce").to_numpy(dtype=float)

            target_out = pd.DataFrame(
                {
                    "split": split,
                    "holdout_mode": holdout_mode,
                    "date": date.date().isoformat(),
                    "year": year,
                    "pass": pass_name,
                    "file_id": file_id_from_path(pass_name, path),
                    "source_file": str(path.resolve()),
                    cfg.KEY: target[cfg.KEY].astype(str).to_numpy(),
                    "observed": target[cfg.TARGET].to_numpy(dtype=float),
                    "x": numeric_or_nan("x"),
                    "y": numeric_or_nan("y"),
                    "grid_row": numeric_or_nan("grid_row"),
                    "grid_col": numeric_or_nan("grid_col"),
                    "holdout_generation": generation,
                    "random_seed": cfg.RANDOM_SEED,
                }
            )
            rows.append(target_out)
            summaries.append(
                {
                    "split": split,
                    "year": year,
                    "date": date.date().isoformat(),
                    "pass": pass_name,
                    "file_id": file_id_from_path(pass_name, path),
                    "holdout_mode": holdout_mode,
                    "n_observed": n_observed,
                    "n_holdout": len(target_out),
                    "n_donors": n_observed - len(target_out),
                    "status": generation,
                }
            )

    if not rows:
        raise RuntimeError(f"No holdouts were generated for {year} ({split}).")

    manifest = pd.concat(rows, ignore_index=True)
    key_cols = ["split", "holdout_mode", "date", "pass", cfg.KEY]
    if manifest.duplicated(key_cols).any():
        raise RuntimeError("Generated holdout manifest contains duplicate target keys.")
    manifest = manifest.sort_values(key_cols).reset_index(drop=True)
    return manifest, pd.DataFrame(summaries)


def main() -> None:
    print("Creating shared SMAP artificial-gap manifests")
    print("=" * 78)
    print(f"Project-wide seed: {cfg.RANDOM_SEED}")
    print(f"Validation year:   {cfg.VALIDATION_YEARS}")
    print(f"Test year:         {cfg.TEST_YEAR}")
    print("=" * 78)

    validation, validation_summary = build_manifest(
        cfg.VALIDATION_YEARS[0], "validation"
    )
    test, test_summary = build_manifest(cfg.TEST_YEAR, "test")

    cfg.VALIDATION_HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TEST_HOLDOUT_DIR.mkdir(parents=True, exist_ok=True)
    validation.to_csv(cfg.VALIDATION_HOLDOUT_PATH, index=False)
    test.to_csv(cfg.TEST_HOLDOUT_PATH, index=False)
    pd.concat([validation_summary, test_summary], ignore_index=True).to_csv(
        cfg.HOLDOUT_SUMMARY_PATH, index=False
    )

    print(f"\nValidation holdouts: {len(validation):,}")
    print(validation.groupby(["holdout_mode", "pass"]).size())
    print(f"\nTest holdouts: {len(test):,}")
    print(test.groupby(["holdout_mode", "pass"]).size())
    print("\nSaved:")
    print(f"  {cfg.VALIDATION_HOLDOUT_PATH}")
    print(f"  {cfg.TEST_HOLDOUT_PATH}")
    print(f"  {cfg.HOLDOUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
