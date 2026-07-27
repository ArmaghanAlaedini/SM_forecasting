#!/usr/bin/env python3
"""Screen candidate predictors before formal 2024 model validation.

This script performs descriptive screening only:

* audits predictor availability in train/validation/test files;
* calculates raw and month-anomaly Spearman correlations using observed
  2020--2023 training rows; and
* records the four candidate feature sets defined in ``00_config.py``.

It does not select ML algorithms and does not use 2025 prediction performance.
Formal feature-set/model comparison is performed by ``10a_ML_validation.py`` on
the shared 2024 artificial gaps.
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
    list_complete_files,
    parse_date_from_filename,
    read_complete_file,
    stable_seed,
)


OUT_DIR = cfg.FEATURE_SCREENING_DIR
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_ROWS_PER_FILE_FOR_SCREENING = 350

DATASET_AUDIT_PATH = OUT_DIR / "dataset_audit_by_split.csv"
MISSINGNESS_PATH = OUT_DIR / "feature_missingness_by_split.csv"
CORRELATION_PATH = OUT_DIR / "feature_correlations_train.csv"
FEATURE_SET_PATH = OUT_DIR / "feature_sets_screened.csv"
SELECTED_SCREENING_PATH = OUT_DIR / "selected_feature_screening.csv"


def split_for_year(year: int) -> str:
    if year in cfg.TRAIN_YEARS:
        return "train"
    if year in cfg.VALIDATION_YEARS:
        return "validation"
    if year in cfg.TEST_YEARS:
        return "test"
    return "unused"


def collect_screening_rows() -> pd.DataFrame:
    needed = list(
        dict.fromkeys(
            [cfg.KEY, cfg.TARGET, *cfg.ALL_IEM_PTA_FEATURES]
        )
    )
    parts: list[pd.DataFrame] = []
    files = list_complete_files(years=cfg.ALL_YEARS)

    for index, (pass_name, path) in enumerate(files, start=1):
        date = parse_date_from_filename(path)
        split = split_for_year(int(date.year))
        if split == "unused":
            continue
        frame = read_complete_file(pass_name, path, usecols=needed)
        for feature in cfg.ALL_IEM_PTA_FEATURES:
            if feature not in frame.columns:
                frame[feature] = np.nan
            frame[feature] = pd.to_numeric(frame[feature], errors="coerce")
        frame[cfg.TARGET] = pd.to_numeric(frame[cfg.TARGET], errors="coerce")
        frame["split"] = split
        frame["target_status"] = np.where(
            frame[cfg.TARGET].notna(), "observed", "original_gap"
        )

        if (
            MAX_ROWS_PER_FILE_FOR_SCREENING is not None
            and len(frame) > MAX_ROWS_PER_FILE_FOR_SCREENING
        ):
            frame = frame.sample(
                n=MAX_ROWS_PER_FILE_FOR_SCREENING,
                random_state=stable_seed("feature_screening", pass_name, date.date()),
            )
        parts.append(
            frame[
                [
                    "date",
                    "year",
                    "month",
                    "pass",
                    "split",
                    "target_status",
                    cfg.KEY,
                    cfg.TARGET,
                    *cfg.ALL_IEM_PTA_FEATURES,
                ]
            ]
        )
        if index % 250 == 0:
            print(f"  screened {index:,} complete files...")

    if not parts:
        raise RuntimeError("No rows were collected for feature screening.")
    return pd.concat(parts, ignore_index=True)


def make_dataset_audit(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["split", "pass"], as_index=False)
        .agg(
            n_rows=(cfg.KEY, "size"),
            n_observed=(cfg.TARGET, lambda s: int(s.notna().sum())),
            n_original_gaps=(cfg.TARGET, lambda s: int(s.isna().sum())),
            n_unique_dates=("date", "nunique"),
            n_unique_pixels=(cfg.KEY, "nunique"),
        )
        .sort_values(["split", "pass"])
    )


def make_missingness(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (split, target_status), sub in data.groupby(
        ["split", "target_status"], sort=True
    ):
        for feature in cfg.ALL_IEM_PTA_FEATURES:
            values = pd.to_numeric(sub[feature], errors="coerce")
            rows.append(
                {
                    "split": split,
                    "target_status": target_status,
                    "feature": feature,
                    "n_rows": len(sub),
                    "n_available": int(values.notna().sum()),
                    "missing_rate": float(values.isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def month_anomaly(series: pd.Series, month: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric - numeric.groupby(month).transform("mean")


def make_correlations(data: pd.DataFrame) -> pd.DataFrame:
    train = data[
        data["split"].eq("train") & data[cfg.TARGET].notna()
    ].copy()
    target = pd.to_numeric(train[cfg.TARGET], errors="coerce")
    target_anomaly = month_anomaly(target, train["month"])

    rows: list[dict] = []
    for feature in cfg.ALL_IEM_PTA_FEATURES:
        values = pd.to_numeric(train[feature], errors="coerce")
        values_anomaly = month_anomaly(values, train["month"])

        raw_valid = target.notna() & values.notna()
        anomaly_valid = target_anomaly.notna() & values_anomaly.notna()
        raw = (
            target.loc[raw_valid].corr(values.loc[raw_valid], method="spearman")
            if raw_valid.sum() >= 3
            else np.nan
        )
        anomaly = (
            target_anomaly.loc[anomaly_valid].corr(
                values_anomaly.loc[anomaly_valid], method="spearman"
            )
            if anomaly_valid.sum() >= 3
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "n_raw_pairs": int(raw_valid.sum()),
                "raw_spearman": raw,
                "abs_raw_spearman": abs(raw) if pd.notna(raw) else np.nan,
                "n_month_anomaly_pairs": int(anomaly_valid.sum()),
                "month_anomaly_spearman": anomaly,
                "abs_month_anomaly_spearman": (
                    abs(anomaly) if pd.notna(anomaly) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "abs_month_anomaly_spearman", ascending=False, na_position="last"
    )


def make_feature_set_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_group": group,
                "position": position,
                "feature": feature,
                "n_features": len(features),
                "selected_for_final_validation": group == cfg.FINAL_ML_FEATURE_GROUP,
            }
            for group, features in cfg.ML_FEATURE_GROUPS.items()
            for position, feature in enumerate(features, start=1)
        ]
    )


def plot_correlations(correlations: pd.DataFrame) -> None:
    sub = correlations.dropna(subset=["abs_month_anomaly_spearman"]).head(20)
    if sub.empty:
        return
    sub = sub.sort_values("abs_month_anomaly_spearman")
    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(sub))))
    ax.barh(sub["feature"], sub["abs_month_anomaly_spearman"])
    ax.set_xlabel("Absolute month-anomaly Spearman correlation with SM")
    ax.set_title("Training-period predictor screening")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_feature_correlations.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    print("09: Feature availability and correlation screening")
    print("=" * 78)
    print(f"Training years:   {cfg.TRAIN_YEARS}")
    print(f"Validation years: {cfg.VALIDATION_YEARS}")
    print(f"Test year:        {cfg.TEST_YEAR} (coverage audit only)")
    print(f"Seed:             {cfg.RANDOM_SEED}")
    print("=" * 78)

    data = collect_screening_rows()
    audit = make_dataset_audit(data)
    missingness = make_missingness(data)
    correlations = make_correlations(data)
    feature_sets = make_feature_set_table()

    train_gap_missingness = missingness[
        missingness["split"].eq("train")
        & missingness["target_status"].eq("original_gap")
    ][["feature", "missing_rate"]].rename(
        columns={"missing_rate": "missing_rate_among_train_gaps"}
    )
    selected = (
        correlations[
            correlations["feature"].isin(cfg.SELECTED_IEM_PTA_FEATURES)
        ]
        .merge(train_gap_missingness, on="feature", how="left")
        .copy()
    )
    selected["selected_position"] = selected["feature"].map(
        {feature: i for i, feature in enumerate(cfg.SELECTED_IEM_PTA_FEATURES, start=1)}
    )
    selected = selected.sort_values("selected_position")

    audit.to_csv(DATASET_AUDIT_PATH, index=False)
    missingness.to_csv(MISSINGNESS_PATH, index=False)
    correlations.to_csv(CORRELATION_PATH, index=False)
    feature_sets.to_csv(FEATURE_SET_PATH, index=False)
    selected.to_csv(SELECTED_SCREENING_PATH, index=False)
    plot_correlations(correlations)

    print("\nSaved:")
    for path in [
        DATASET_AUDIT_PATH,
        MISSINGNESS_PATH,
        CORRELATION_PATH,
        FEATURE_SET_PATH,
        SELECTED_SCREENING_PATH,
    ]:
        print(f"  {path}")
    print("\nReduced nine-variable screening table:")
    print(selected.to_string(index=False))
    print(
        "\nFormal feature-set and ML-model selection is performed by "
        "10a_ML_validation.py on shared 2024 artificial gaps."
    )


if __name__ == "__main__":
    main()
