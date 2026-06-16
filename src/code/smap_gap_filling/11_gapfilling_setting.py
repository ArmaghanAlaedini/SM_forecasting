#!/usr/bin/env python3
"""
11_gapfilling_setting.py

Manual settings for the 11-family SMAP gap-filling scripts.
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling/03_full_smap_iem_data"
)

PREDICTION_DIR = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling/07_gapfill_predictions"
)

FINAL_DIR = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling/08_gapfilled_final"
)

META_MODEL_PATH = (
    PROJECT_ROOT
    / "src/data/processed/smap_gap_filling"
    / "05_gapfill_model_validation/stacking/meta_model.joblib"
)


# ============================================================
# BASIC COLUMNS
# ============================================================

TARGET = "soil_moisture"
KEY = "smap_pixel_key"
PASSES = ["am", "pm"]


# ============================================================
# YEARS
# ============================================================

ML_TRAIN_YEARS = [2020, 2021, 2022, 2023]
GAPFILL_YEARS  = [2020, 2021, 2022, 2023, 2024, 2025]


# ============================================================
# ML MODEL SELECTION
# ============================================================

ML_MODELS_TO_USE = [
    "xgboost",
    "hist_gbdt",
    "random_forest",
]


# ============================================================
# ML FEATURE SELECTION
# ============================================================

ML_FEATURE_GROUP_NAME = "available_all_iem_spatiotemporal"

ML_FEATURES_TO_USE = [
    "precip_pta",
    "rh_pta",
    "speed_pta",
    "gust_pta",
    "et_pta",
    "soil04tn_pta",
    "soil04t_pta",
    "soil04tx_pta",
    "soil24tn_pta",
    "soil24t_pta",
    "soil24tx_pta",
    "soil50tn_pta",
    "soil50t_pta",
    "soil50tx_pta",
    "x",
    "y",
    "sin_doy",
    "cos_doy",
    "pass_pm",
]

STRICT_ML_FEATURES = False


# ============================================================
# TRAINING SIZE CONTROL
# ============================================================

MAX_ML_TRAIN_ROWS = 250_000
RANDOM_STATE = 42


# ============================================================
# INTERPOLATION METHOD SELECTION
# ============================================================

INTERPOLATION_METHODS_TO_USE = [
    "centroid_ordinary_kriging",
    "nearest_neighbor_same_day",
    "regression_kriging",
]


# ============================================================
# FINAL FILLING RULE
# ============================================================

FINAL_PRIMARY_METHOD = "centroid_ordinary_kriging"

FINAL_FALLBACK_METHODS = [
    "nearest_neighbor_same_day",
    "regression_kriging",
    "xgboost",
    "hist_gbdt",
    "random_forest",
]

CLIP_FILLED_VALUES = False
CLIP_MIN = 0.0
CLIP_MAX = 0.7