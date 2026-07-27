#!/usr/bin/env python3
"""Compatibility settings for the 11-family production scripts.

The authoritative settings now live in ``00_config.py``.  This wrapper keeps
older imports working while preventing validation/test/production drift.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().with_name("00_config.py")
spec = importlib.util.spec_from_file_location("cfg", CONFIG_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load configuration from {CONFIG_PATH}")
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

PROJECT_ROOT = cfg.PROJECT_ROOT
INPUT_DIR = cfg.FULL_SMAP_IEM_DIR
PREDICTION_DIR = cfg.PREDICTION_DIR
FINAL_DIR = cfg.FINAL_DIR
META_MODEL_PATH = cfg.META_MODEL_PATH

TARGET = cfg.TARGET
KEY = cfg.KEY
PASSES = cfg.PASSES

ML_TRAIN_YEARS = cfg.TRAIN_YEARS
GAPFILL_YEARS = cfg.GAPFILL_YEARS
ML_MODELS_TO_USE = cfg.SELECTED_ML_MODELS
ML_FEATURE_GROUP_NAME = cfg.FINAL_ML_FEATURE_GROUP
ML_FEATURES_TO_USE = cfg.FINAL_ML_FEATURES
STRICT_ML_FEATURES = cfg.STRICT_FINAL_ML_FEATURES
MAX_ML_TRAIN_ROWS = cfg.MAX_ML_TRAIN_ROWS
RANDOM_STATE = cfg.RANDOM_SEED

INTERPOLATION_METHODS_TO_USE = cfg.SELECTED_INTERPOLATION_METHODS
FINAL_PRIMARY_METHOD = cfg.FINAL_PRIMARY_METHOD
FINAL_FALLBACK_METHODS = cfg.FINAL_FALLBACK_METHODS
CLIP_FILLED_VALUES = cfg.CLIP_FILLED_VALUES
CLIP_MIN = cfg.CLIP_MIN
CLIP_MAX = cfg.CLIP_MAX
