#!/usr/bin/env python3
"""Central configuration for the SMAP soil-moisture gap-filling workflow.

All Python scripts load this file.  The R scripts read the CSV holdout manifests
created by ``10_generate_holdout_manifests.py`` and mirror the geostatistical
constants defined here.

Study design
------------
* 2020--2023: train ML base models.
* 2024: model/feature validation and ridge meta-model development.
* 2025: independent test of the frozen base models and ridge stack.

The project-wide random seed is 1234.
"""

from __future__ import annotations

import os
from pathlib import Path


# ============================================================================
# Project and data roots
# ============================================================================

def find_project_root() -> Path:
    """Locate the repository root or use ``SMAP_PROJECT_ROOT`` when supplied."""
    env_root = os.environ.get("SMAP_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"SMAP_PROJECT_ROOT is set but does not exist: {root}"
            )
        return root

    start = Path(__file__).resolve().parent
    for folder in [start, *start.parents]:
        has_src = (folder / "src").is_dir()
        marker = (
            (folder / ".git").exists()
            or (folder / "renv").is_dir()
            or (folder / "environment.yml").exists()
        )
        if has_src and marker:
            return folder

    raise FileNotFoundError(
        "Could not find the project root. Run inside the repository or set "
        "SMAP_PROJECT_ROOT."
    )


def get_data_root() -> Path:
    """Return the local/HPC data root and create it when needed."""
    env_data = os.environ.get("SMAP_DATA_ROOT")
    root = (
        Path(env_data).expanduser().resolve()
        if env_data
        else PROJECT_ROOT / "src" / "data"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATA_ROOT = get_data_root()
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"


# ============================================================================
# Inputs
# ============================================================================

RAW_SMAP_NC_DIR = Path(
    os.environ.get("SMAP_RAW_NC_DIR", RAW_DIR / "smap_observations")
).expanduser().resolve()

TOWNSHIP_SHP_PATH = RAW_DIR / "townships" / "civil_townships_a_ia.shp"

SMAP_DETRENDED_DIR = PROCESSED_DIR / "smap_detrended"
SMAP_DETRENDED_AM_RDS_DIR = SMAP_DETRENDED_DIR / "am" / "rds"
SMAP_DETRENDED_PM_RDS_DIR = SMAP_DETRENDED_DIR / "pm" / "rds"
SMAP_DETRENDED_AM_CSV_DIR = SMAP_DETRENDED_DIR / "am" / "csv"
SMAP_DETRENDED_PM_CSV_DIR = SMAP_DETRENDED_DIR / "pm" / "csv"

IEM_STATION_DIR = PROCESSED_DIR / "isu_stations"
IEM_STATIONS_FULL_PATH = IEM_STATION_DIR / "stations_full.csv"
IEM_STATIONS_FULL_FALLBACK_PATH = IEM_STATION_DIR / "full_stations.csv"


# ============================================================================
# Study design and reproducibility
# ============================================================================

PASSES = ["am", "pm"]
TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALIDATION_YEARS = [2024]
TEST_YEAR = 2025
TEST_YEARS = [TEST_YEAR]
ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS

RANDOM_SEED = 1234
# Backward-compatible alias used by a few scripts.
RANDOM_STATE = RANDOM_SEED

TARGET = "soil_moisture"
KEY = "smap_pixel_key"

CRS_WGS84 = 4326
CRS_EASE = 6933
SMAP_CELLSIZE_M = 9024.31
# Backward-compatible spelling used in one older script.
SMAP_CELL_SIZE_M = SMAP_CELLSIZE_M
IOWA_BBOX = (-97.0, 40.0, -89.0, 44.0)


# ============================================================================
# Output structure
# ============================================================================

GAP_FILLING_DIR = PROCESSED_DIR / "smap_gap_filling"
SUPPORT_DIR = GAP_FILLING_DIR / "support"
SMAP_LATTICE_DIR = SUPPORT_DIR / "smap_lattice"
IEM_PTA_DIR = GAP_FILLING_DIR / "iem_point_to_area"

FULL_SMAP_IEM_DIR = GAP_FILLING_DIR / "03_full_smap_iem_data"
FEATURE_SCREENING_DIR = GAP_FILLING_DIR / "04_feature_screening"
VALIDATION_DIR = GAP_FILLING_DIR / "05_gapfill_model_validation"
TEST_DIR = GAP_FILLING_DIR / "06_selected_methods_test"
PREDICTION_DIR = GAP_FILLING_DIR / "07_gapfill_predictions"
FINAL_DIR = GAP_FILLING_DIR / "08_gapfilled_final"

VALIDATION_HOLDOUT_DIR = VALIDATION_DIR / "holdouts"
TEST_HOLDOUT_DIR = TEST_DIR / "holdouts"
VALIDATION_HOLDOUT_PATH = VALIDATION_HOLDOUT_DIR / "validation_holdouts_2024.csv"
TEST_HOLDOUT_PATH = TEST_HOLDOUT_DIR / "test_holdouts_2025.csv"
HOLDOUT_SUMMARY_PATH = SUPPORT_DIR / "holdout_manifest_summary.csv"

ML_VALIDATION_DIR = VALIDATION_DIR / "ml"
INTERP_VALIDATION_DIR = VALIDATION_DIR / "interpolation"
COMPARISON_DIR = VALIDATION_DIR / "comparison"
STACKING_DIR = VALIDATION_DIR / "stacking"

ML_TEST_DIR = TEST_DIR / "ml"
INTERP_TEST_DIR = TEST_DIR / "interpolation"
STACKING_TEST_DIR = TEST_DIR / "stacking"

META_MODEL_PATH = STACKING_DIR / "meta_model.joblib"

FULL_SMAP_IEM_AM_DIR = FULL_SMAP_IEM_DIR / "am"
FULL_SMAP_IEM_PM_DIR = FULL_SMAP_IEM_DIR / "pm"
FULL_SMAP_IEM_AM_COMPLETE_DIR = FULL_SMAP_IEM_AM_DIR / "complete"
FULL_SMAP_IEM_AM_OBSERVED_DIR = FULL_SMAP_IEM_AM_DIR / "observed"
FULL_SMAP_IEM_AM_MISSING_DIR = FULL_SMAP_IEM_AM_DIR / "missing"
FULL_SMAP_IEM_PM_COMPLETE_DIR = FULL_SMAP_IEM_PM_DIR / "complete"
FULL_SMAP_IEM_PM_OBSERVED_DIR = FULL_SMAP_IEM_PM_DIR / "observed"
FULL_SMAP_IEM_PM_MISSING_DIR = FULL_SMAP_IEM_PM_DIR / "missing"
FULL_SMAP_IEM_SUMMARY_PATH = FULL_SMAP_IEM_DIR / "full_smap_iem_build_summary.csv"


# ============================================================================
# Runtime limits
# ============================================================================

def get_file_limit(env_name: str, default: int | None = None) -> int | None:
    """Read a positive integer limit, or ``all``/``none`` for no limit."""
    value = os.environ.get(env_name)
    if value is None or value.strip() == "":
        return default
    value = value.strip().lower()
    if value in {"all", "none"}:
        return None
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be a positive integer, 'all', 'none', or unset."
        ) from exc
    if n <= 0:
        raise ValueError(f"{env_name} must be positive.")
    return n


MAX_FILES = get_file_limit("SMAP_MAX_FILES", None)
MAX_DAYS = get_file_limit("SMAP_MAX_DAYS", None)
LATTICE_SCAN_FILES = get_file_limit("SMAP_LATTICE_SCAN_FILES", 100)
MAX_ML_TRAIN_ROWS = get_file_limit("SMAP_MAX_ML_TRAIN_ROWS", 250_000)
MAX_FILES_PER_SPLIT_PER_PASS = get_file_limit(
    "SMAP_MAX_FILES_PER_SPLIT_PER_PASS", None
)


# ============================================================================
# IEM point-to-area settings
# ============================================================================

IEM_PTA_VARIABLES = [
    "precip",
    "rh",
    "speed",
    "gust",
    "et",
    "soil04tn",
    "soil04t",
    "soil04tx",
    "soil12tn",
    "soil12t",
    "soil12tx",
    "soil12vwc",
    "soil24tn",
    "soil24t",
    "soil24tx",
    "soil24vwc",
    "soil50tn",
    "soil50t",
    "soil50tx",
    "soil50vwc",
]
# Compatibility alias used by 09_feature_selection.py.
IEM_VARIABLES = IEM_PTA_VARIABLES

IEM_MISSING_VALUE = -99.0
MIN_STATIONS_FOR_KRIGING = 8
IEM_PTA_SAMPLE_MODE = "five_point"
IEM_VARIIOGRAM_MODEL = "spherical"


# ============================================================================
# Artificial holdout settings
# ============================================================================

HOLDOUT_MODES = ["random_cell", "spatial_block"]
RANDOM_CELL_HOLDOUT_FRACTION = 0.25
MIN_OBSERVED_ROWS_PER_RETRIEVAL = 30
MIN_DONOR_ROWS = 20
MIN_HOLDOUT_ROWS = 10

# The spatial holdout divides each retrieval into a 4 x 4 quantile grid and
# selects one contiguous 2 x 2 block, which targets approximately 25% of rows.
SPATIAL_BLOCK_N_BINS = 4
SPATIAL_BLOCK_WIDTH_BINS = 2

# The ridge meta-model is developed using the harder, more realistic holdout.
STACKING_HOLDOUT_MODE = "spatial_block"


# ============================================================================
# ML feature sets and model settings
# ============================================================================

SELECTED_IEM_PTA_FEATURES = [
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

ALL_IEM_PTA_FEATURES = [f"{name}_pta" for name in IEM_PTA_VARIABLES]
SPATIOTEMPORAL_FEATURES = ["x", "y", "sin_doy", "cos_doy", "pass_pm"]

ML_FEATURE_GROUPS = {
    "selected_iem_pta": SELECTED_IEM_PTA_FEATURES,
    "selected_iem_spatiotemporal": (
        SELECTED_IEM_PTA_FEATURES + SPATIOTEMPORAL_FEATURES
    ),
    "all_iem_pta": ALL_IEM_PTA_FEATURES,
    "all_iem_spatiotemporal": ALL_IEM_PTA_FEATURES + SPATIOTEMPORAL_FEATURES,
}

FINAL_ML_FEATURE_GROUP = "all_iem_spatiotemporal"
FINAL_ML_FEATURES = ML_FEATURE_GROUPS[FINAL_ML_FEATURE_GROUP]
STRICT_FINAL_ML_FEATURES = True

CANDIDATE_ML_MODELS = [
    "random_forest",
    "extra_trees",
    "hist_gbdt",
    "xgboost",
    "ffnn_mlp",
]
SELECTED_ML_MODELS = ["xgboost", "hist_gbdt", "random_forest"]

ML_MODEL_SETTINGS = {
    "random_forest": {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "n_jobs": -1,
    },
    "extra_trees": {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 3,
        "n_jobs": -1,
    },
    "hist_gbdt": {
        "max_iter": 350,
        "learning_rate": 0.05,
        "max_leaf_nodes": 31,
        "l2_regularization": 0.01,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
    },
    "ffnn_mlp": {
        "hidden_layer_sizes": (128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,
        "learning_rate_init": 1e-3,
        "max_iter": 400,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
    },
}


# ============================================================================
# GI settings mirrored in gapfill_geostat_common.R
# ============================================================================

SELECTED_INTERPOLATION_METHODS = [
    "centroid_ordinary_kriging",
    "nearest_neighbor_same_day",
    "regression_kriging",
]
CENTROID_OK_NMAX = 30
CENTROID_OK_DETREND = True
CENTROID_OK_TREND_PVALUE = 0.05
CENTROID_OK_TREND_R2 = 0.01
VARIOGRAM_MODEL = "Sph"
VARIOGRAM_INITIAL_RANGE_M = 50_000.0
VARIOGRAM_NUGGET_FRACTION = 0.20
RK_IEM_COVARIATES = [
    "soil04t_pta",
    "soil12vwc_pta",
    "soil24vwc_pta",
    "soil50vwc_pta",
    "precip_pta",
    "et_pta",
    "rh_pta",
]
RK_MAX_COVARIATES = 3
RK_MIN_OBS_PER_PARAMETER = 5


# ============================================================================
# Stacking and final filling
# ============================================================================

BASE_PREDICTION_COLUMNS = [
    "pred_centroid_ordinary_kriging",
    "pred_nearest_neighbor_same_day",
    "pred_regression_kriging",
    "pred_xgboost",
    "pred_hist_gbdt",
    "pred_random_forest",
]
META_EXTRA_FEATURES = ["x", "y", "sin_doy", "cos_doy", "pass_pm"]
META_FEATURE_COLUMNS = BASE_PREDICTION_COLUMNS + META_EXTRA_FEATURES
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
META_GROUP_CV_FOLDS = 5
META_DIAGNOSTIC_TEST_FRACTION = 0.20

GAPFILL_YEARS = ALL_YEARS
FINAL_PRIMARY_METHOD = "nearest_neighbor_same_day"
FINAL_FALLBACK_METHODS = [
    "centroid_ordinary_kriging",
    "regression_kriging",
    "xgboost",
    "hist_gbdt",
    "random_forest",
]
CLIP_FILLED_VALUES = False
CLIP_MIN = 0.0
CLIP_MAX = 0.7


# ============================================================================
# Helpers used by preprocessing scripts
# ============================================================================

def _limit_files(files: list[Path], max_files: int | None) -> list[Path]:
    files = sorted(files)
    return files[:max_files] if max_files is not None else files


def _list_files(folder: Path, suffix: str) -> list[Path]:
    return sorted(folder.glob(f"*{suffix}")) if folder.exists() else []


def get_smap_dirs(pass_name: str) -> dict[str, Path]:
    pass_name = pass_name.lower()
    if pass_name == "am":
        return {"csv": SMAP_DETRENDED_AM_CSV_DIR, "rds": SMAP_DETRENDED_AM_RDS_DIR}
    if pass_name == "pm":
        return {"csv": SMAP_DETRENDED_PM_CSV_DIR, "rds": SMAP_DETRENDED_PM_RDS_DIR}
    raise ValueError("pass_name must be 'am' or 'pm'.")


def list_smap_files(
    pass_name: str,
    file_mode: str = "auto",
    max_files: int | None = MAX_FILES,
) -> list[Path]:
    dirs = get_smap_dirs(pass_name)
    csv_files = _list_files(dirs["csv"], ".csv")
    rds_files = _list_files(dirs["rds"], ".rds")
    mode = file_mode.lower()
    if mode == "auto":
        files = csv_files if csv_files else rds_files
    elif mode == "csv":
        files = csv_files
    elif mode == "rds":
        files = rds_files
    else:
        raise ValueError("file_mode must be 'auto', 'csv', or 'rds'.")
    return _limit_files(files, max_files)


def get_iem_pta_daily_csv_path(date_yyyymmdd: str) -> Path:
    return IEM_PTA_DIR / f"iem_pta_smap_lattice_{date_yyyymmdd}.csv"


def get_full_smap_iem_dirs(pass_name: str) -> dict[str, Path]:
    pass_name = pass_name.lower()
    if pass_name == "am":
        return {
            "complete": FULL_SMAP_IEM_AM_COMPLETE_DIR,
            "observed": FULL_SMAP_IEM_AM_OBSERVED_DIR,
            "missing": FULL_SMAP_IEM_AM_MISSING_DIR,
        }
    if pass_name == "pm":
        return {
            "complete": FULL_SMAP_IEM_PM_COMPLETE_DIR,
            "observed": FULL_SMAP_IEM_PM_OBSERVED_DIR,
            "missing": FULL_SMAP_IEM_PM_MISSING_DIR,
        }
    raise ValueError("pass_name must be 'am' or 'pm'.")


def ensure_output_dirs() -> None:
    folders = [
        GAP_FILLING_DIR,
        SUPPORT_DIR,
        SMAP_LATTICE_DIR,
        IEM_PTA_DIR,
        FULL_SMAP_IEM_DIR,
        FULL_SMAP_IEM_AM_COMPLETE_DIR,
        FULL_SMAP_IEM_AM_OBSERVED_DIR,
        FULL_SMAP_IEM_AM_MISSING_DIR,
        FULL_SMAP_IEM_PM_COMPLETE_DIR,
        FULL_SMAP_IEM_PM_OBSERVED_DIR,
        FULL_SMAP_IEM_PM_MISSING_DIR,
        FEATURE_SCREENING_DIR,
        VALIDATION_HOLDOUT_DIR,
        TEST_HOLDOUT_DIR,
        ML_VALIDATION_DIR,
        INTERP_VALIDATION_DIR,
        COMPARISON_DIR,
        STACKING_DIR,
        ML_TEST_DIR,
        INTERP_TEST_DIR,
        STACKING_TEST_DIR,
        PREDICTION_DIR,
        FINAL_DIR,
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def print_config_summary() -> None:
    """Print the settings most often needed when auditing a run."""
    print("\nSMAP gap-filling configuration")
    print("-" * 72)
    print(f"Project root:             {PROJECT_ROOT}")
    print(f"Data root:                {DATA_ROOT}")
    print(f"Full SMAP + IEM dir:      {FULL_SMAP_IEM_DIR}")
    print(f"Train years:              {TRAIN_YEARS}")
    print(f"Validation years:         {VALIDATION_YEARS}")
    print(f"Test years:               {TEST_YEARS}")
    print(f"Random seed:              {RANDOM_SEED}")
    print(f"Final ML feature group:   {FINAL_ML_FEATURE_GROUP}")
    print(f"Final ML predictors:      {len(FINAL_ML_FEATURES)}")
    print(f"Selected ML models:       {SELECTED_ML_MODELS}")
    print(f"Selected GI methods:      {SELECTED_INTERPOLATION_METHODS}")
    print(f"Validation holdouts:      {VALIDATION_HOLDOUT_PATH}")
    print(f"Test holdouts:            {TEST_HOLDOUT_PATH}")
    print("-" * 72)


ensure_output_dirs()


if __name__ == "__main__":
    print_config_summary()
