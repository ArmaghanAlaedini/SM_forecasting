#!/usr/bin/env python3
"""
10g_train_stacking_meta_model.py

Train a Ridge regression meta-model (stacking layer) on the meta-training
table produced by 10f.

The meta-model learns how to best combine these base predictions:
    pred_centroid_ordinary_kriging
    pred_nearest_neighbor_same_day
    pred_xgboost
    pred_hist_gbdt
    pred_random_forest

Ridge is chosen because:
  - It handles correlated base predictions well
  - Its coefficients are interpretable (how much it trusts each model)
  - RidgeCV selects regularisation strength via leave-one-out CV automatically
  - It never over-fits on a table of this size

Inputs
------
05_gapfill_model_validation/stacking/meta_training_table.csv   (from 10f)

Outputs
-------
05_gapfill_model_validation/stacking/
    meta_model.joblib          ← load this in 11c
    meta_model_coefficients.csv
    meta_model_cv_report.txt
    meta_model_scatter.pdf     ← predicted vs observed on held-out 20%
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():
    config_path = Path(__file__).resolve().with_name("00_config.py")
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")
    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# SETTINGS
# ============================================================

TARGET = "soil_moisture"
KEY = "smap_pixel_key"

RANDOM_STATE = 42

# Held-out fraction used only for the diagnostic scatter plot.
# The final saved model is re-trained on ALL meta-training rows.
DIAGNOSTIC_TEST_FRACTION = 0.20

# Ridge regularisation candidates explored by RidgeCV.
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

# Base prediction column names expected from 10f.
# The meta-model is trained on whichever of these are actually present
# in the meta_training_table.csv (missing ones are imputed with median).
BASE_PRED_COLS = [
    "pred_centroid_ordinary_kriging",
    "pred_nearest_neighbor_same_day",
    "pred_xgboost",
    "pred_hist_gbdt",
    "pred_random_forest",
]

# Optional: add spatial/temporal covariates to the meta-model.
# Usually adds very little on top of the base predictions.
# Set to [] to keep the meta-model purely combinatorial.
META_EXTRA_FEATURES = ["x", "y", "sin_doy", "cos_doy", "pass_pm"]


# ============================================================
# PATHS
# ============================================================

GAP_FILLING_DIR = cfg.GAP_FILLING_DIR
STACKING_DIR = GAP_FILLING_DIR / "05_gapfill_model_validation" / "stacking"

META_TABLE_PATH = STACKING_DIR / "meta_training_table.csv"

OUT_MODEL_PATH = STACKING_DIR / "meta_model.joblib"
OUT_COEF_PATH = STACKING_DIR / "meta_model_coefficients.csv"
OUT_REPORT_PATH = STACKING_DIR / "meta_model_cv_report.txt"
OUT_SCATTER_PATH = STACKING_DIR / "meta_model_scatter.pdf"


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run 10f_generate_stacking_meta_features.py first."
        )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def make_meta_pipeline(alphas: list[float]) -> Pipeline:
    """
    Impute → scale → RidgeCV pipeline.

    Scaling is important here because the base predictions all live on the
    same [0, ~0.6] scale, but extra features (x, y in metres) differ greatly.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, fit_intercept=True)),
    ])


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature columns actually present in the data."""
    wanted = BASE_PRED_COLS + META_EXTRA_FEATURES
    present = [c for c in wanted if c in df.columns]
    if not present:
        raise ValueError(
            "No recognised feature columns found in meta_training_table.csv. "
            f"Expected some of: {wanted}"
        )
    return present


# ============================================================
# DIAGNOSTIC PLOT
# ============================================================

def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.2, s=6, rasterized=True)
    lims = [min(y_true.min(), y_pred.min()) - 0.01,
            max(y_true.max(), y_pred.max()) + 0.01]
    ax.plot(lims, lims, "r--", linewidth=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed soil moisture")
    ax.set_ylabel("Meta-model prediction")
    ax.set_title("Stacking meta-model: held-out 20%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("10g: Train stacking meta-model")
    print("=" * 70)
    print(f"Meta table:   {META_TABLE_PATH}")
    print(f"Output model: {OUT_MODEL_PATH}")
    print("=" * 70)

    require_file(META_TABLE_PATH)

    meta = pd.read_csv(META_TABLE_PATH, low_memory=False)
    print(f"Meta table rows: {len(meta):,}")

    meta[TARGET] = pd.to_numeric(meta[TARGET], errors="coerce")
    meta = meta[meta[TARGET].notna()].copy()
    print(f"Rows with valid target: {len(meta):,}")

    feature_cols = get_feature_cols(meta)
    print(f"\nMeta-model feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        n_valid = meta[c].notna().sum()
        print(f"  {c:<45}  {n_valid:>8,} non-null  ({100*n_valid/len(meta):.1f}%)")

    for c in feature_cols:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    X_all = meta[feature_cols].to_numpy(dtype=float)
    y_all = meta[TARGET].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Diagnostic split (20 % held out for the scatter plot only)
    # --------------------------------------------------------
    X_tr, X_ts, y_tr, y_ts = train_test_split(
        X_all, y_all,
        test_size=DIAGNOSTIC_TEST_FRACTION,
        random_state=RANDOM_STATE,
    )

    print(f"\nDiagnostic split: {len(X_tr):,} train / {len(X_ts):,} test")

    diag_pipe = make_meta_pipeline(RIDGE_ALPHAS)
    diag_pipe.fit(X_tr, y_tr)

    y_pred_diag = diag_pipe.predict(X_ts)
    diag_metrics = compute_metrics(y_ts, y_pred_diag)

    print("\nDiagnostic metrics on held-out 20%:")
    for k, v in diag_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    chosen_alpha = diag_pipe.named_steps["ridge"].alpha_
    print(f"\nRidgeCV selected alpha: {chosen_alpha}")

    plot_scatter(y_ts, y_pred_diag, OUT_SCATTER_PATH)

    # --------------------------------------------------------
    # Final model: re-train on ALL rows with the chosen alpha
    # --------------------------------------------------------
    print("\nRe-training on all meta-training rows...")

    final_pipe = make_meta_pipeline(RIDGE_ALPHAS)
    final_pipe.fit(X_all, y_all)

    # Extract coefficients (after scaling, so they reflect relative importance)
    ridge = final_pipe.named_steps["ridge"]
    scaler = final_pipe.named_steps["scaler"]

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": ridge.coef_,
        "scaled_std": scaler.scale_,
    })
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    coef_df.to_csv(OUT_COEF_PATH, index=False)
    print(f"\nCoefficients saved: {OUT_COEF_PATH}")
    print(coef_df[["feature", "coefficient"]].to_string(index=False))

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    joblib.dump(final_pipe, OUT_MODEL_PATH)
    print(f"\nMeta-model saved: {OUT_MODEL_PATH}")

    # --------------------------------------------------------
    # Write report
    # --------------------------------------------------------
    report_lines = [
        "Stacking meta-model training report",
        "=" * 50,
        f"Meta table rows:     {len(meta):,}",
        f"Feature columns:     {len(feature_cols)}",
        f"RidgeCV alpha:       {final_pipe.named_steps['ridge'].alpha_}",
        f"Intercept:           {final_pipe.named_steps['ridge'].intercept_:.6f}",
        "",
        "Diagnostic metrics (held-out 20%)",
        "-" * 50,
    ]
    for k, v in diag_metrics.items():
        report_lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    report_lines += [
        "",
        "Coefficients (sorted by abs value)",
        "-" * 50,
        coef_df[["feature", "coefficient"]].to_string(index=False),
        "",
        "Interpretation:",
        "  Positive coefficient → method pushes prediction up.",
        "  Larger |coefficient| → more weight given to that method.",
        "  Ridge shrinks coefficients toward zero, so very small values",
        "  indicate the method adds little beyond what others already capture.",
    ]

    report_text = "\n".join(report_lines)
    OUT_REPORT_PATH.write_text(report_text)
    print(f"Report saved: {OUT_REPORT_PATH}")

    print("\nDone.  Load the meta-model in 11c with:")
    print(f"  import joblib")
    print(f"  meta_model = joblib.load('{OUT_MODEL_PATH}')")


if __name__ == "__main__":
    main()
