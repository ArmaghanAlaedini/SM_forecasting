#!/usr/bin/env python3
"""
11_gapfilling_setting.py

Manual settings for the 11-family SMAP gap-filling scripts.

Edit this file to control:
  - which ML models are used
  - which ML features are used
  - which final filling method is primary
  - which fallback methods are allowed
  - which years are gap-filled

Used by:
  11a_generate_ml_gapfill_predictions.py
  11c_stack_and_finalize_gapfills.py
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


# ============================================================
# BASIC COLUMNS
# ============================================================

TARGET = "soil_moisture"
KEY = "smap_pixel_key"
PASSES = ["am", "pm"]


# ============================================================
# YEARS
# ============================================================

# Production gap filling:
# Use all observed rows from 2020-2025 to train ML, then fill real gaps.
# This is for creating the retrospective completed dataset.
ML_TRAIN_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Years whose real missing pixels should be filled.
GAPFILL_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


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

# Currently available feature subset.
# The missing columns are not included:
#   soil112tn_pta
#   soil112t_pta
#   soil112tx_pta
#   soil112wc_pta
#   soil24wc_pta
#   soil50wc_pta
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

# If False, missing requested features are skipped and reported.
# If True, the script stops if any requested feature is unavailable.
STRICT_ML_FEATURES = False


# ============================================================
# TRAINING SIZE CONTROL
# ============================================================

# Use None on HPC if you want all training rows.
# Use 250_000 or 500_000 on laptop if needed.
MAX_ML_TRAIN_ROWS = 250_000

RANDOM_STATE = 42


# ============================================================
# INTERPOLATION METHOD SELECTION
# ============================================================

INTERPOLATION_METHODS_TO_USE = [
    "centroid_ordinary_kriging",
    "nearest_neighbor_same_day",
]


# ============================================================
# FINAL FILLING RULE
# ============================================================

FINAL_PRIMARY_METHOD = "centroid_ordinary_kriging"

FINAL_FALLBACK_METHODS = [
    "nearest_neighbor_same_day",
    "xgboost",
    "hist_gbdt",
    "random_forest",
]

# Optional clipping. Leave False unless explicitly needed.
CLIP_FILLED_VALUES = False
CLIP_MIN = 0.0
CLIP_MAX = 0.7