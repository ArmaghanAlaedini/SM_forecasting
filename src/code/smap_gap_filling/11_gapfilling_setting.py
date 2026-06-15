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
  - stacking meta-model path

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

# Path to the stacking meta-model saved by 10g.
# 11c will load this and use it to combine base predictions.
# Set to None to fall back to the waterfall rule (no stacking).
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
# YEARS  ← FIXED: 2024 and 2025 excluded from ML training
# ============================================================

# Models must be trained ONLY on 2020-2023.
# 2024 = validation year (used in 10a/10b/10c to pick methods).
# 2025 = test year (never touched during model selection).
ML_TRAIN_YEARS = [2020, 2021, 2022, 2023]

# Years whose real missing pixels should be filled.
# All years are filled, but the models were trained only on 2020-2023.
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
# Use 250_000 on laptop if memory is tight.
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

# Primary method fed into the stacking meta-model.
# Also used as the first waterfall fallback if stacking is disabled.
FINAL_PRIMARY_METHOD = "centroid_ordinary_kriging"

# Waterfall order used ONLY when:
#   (a) META_MODEL_PATH is None, OR
#   (b) the meta-model itself returns NaN for a pixel
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