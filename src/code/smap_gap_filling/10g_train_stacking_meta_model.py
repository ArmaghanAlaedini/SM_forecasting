#!/usr/bin/env python3
"""Train the ridge stacking meta-model on complete aligned 2024 predictions.

The diagnostic holdout is grouped by complete date-pass retrievals, not random
pixels.  Ridge alpha is selected with grouped cross-validation.  No base-model
prediction is imputed: ``10f`` guarantees that every row contains six finite
base predictions.
"""

from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gapfill_workflow_common import cfg, compute_metrics


META_PATH = cfg.STACKING_DIR / "meta_training_table.csv"
MODEL_PATH = cfg.META_MODEL_PATH
COEFFICIENT_PATH = cfg.STACKING_DIR / "meta_model_coefficients.csv"
REPORT_PATH = cfg.STACKING_DIR / "meta_model_cv_report.txt"
METADATA_PATH = cfg.STACKING_DIR / "meta_model_metadata.json"
SCATTER_PATH = cfg.STACKING_DIR / "meta_model_scatter.pdf"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(fit_intercept=True)),
        ]
    )


def fit_grouped_search(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series,
) -> GridSearchCV:
    n_groups = groups.nunique()
    n_splits = min(cfg.META_GROUP_CV_FOLDS, n_groups)
    if n_splits < 2:
        raise ValueError("At least two date-pass groups are required for ridge CV.")
    search = GridSearchCV(
        estimator=make_pipeline(),
        param_grid={"ridge__alpha": cfg.RIDGE_ALPHAS},
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y, groups=groups)
    return search


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.25, s=8, rasterized=True)
    lower = min(float(np.min(y_true)), float(np.min(y_pred))) - 0.01
    upper = max(float(np.max(y_true)), float(np.max(y_pred))) + 0.01
    ax.plot([lower, upper], [lower, upper], "--", linewidth=1)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Observed SM")
    ax.set_ylabel("Stacked prediction")
    ax.set_title("Grouped 2024 stacking diagnostic")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(SCATTER_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not META_PATH.exists():
        raise FileNotFoundError(f"Meta-training table not found: {META_PATH}")

    meta = pd.read_csv(META_PATH, low_memory=False)
    meta["date"] = pd.to_datetime(meta["date"], errors="raise").dt.normalize()

    required = ["soil_moisture", "date", "pass", *cfg.META_FEATURE_COLUMNS]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Meta-training table is missing columns: {missing}")

    numeric = meta[cfg.META_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    target = pd.to_numeric(meta["soil_moisture"], errors="coerce")
    finite &= np.isfinite(target.to_numpy(dtype=float))
    if not finite.all():
        raise ValueError(
            f"Meta table contains {(~finite).sum():,} incomplete rows. "
            "Re-run 10f; do not impute missing base predictions."
        )

    X = numeric
    y = target.to_numpy(dtype=float)
    groups = meta["date"].dt.strftime("%Y-%m-%d") + "_" + meta["pass"].astype(str)

    unique_groups = np.array(sorted(groups.unique()))
    if len(unique_groups) >= 3:
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        n_test_groups = max(1, int(round(cfg.META_DIAGNOSTIC_TEST_FRACTION * len(unique_groups))))
        n_test_groups = min(n_test_groups, len(unique_groups) - 2)
        test_groups = set(rng.choice(unique_groups, size=n_test_groups, replace=False))
        test_mask = groups.isin(test_groups).to_numpy()
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        diagnostic_search = fit_grouped_search(
            X.iloc[train_idx], y[train_idx], groups.iloc[train_idx]
        )
        diagnostic_prediction = diagnostic_search.predict(X.iloc[test_idx])
        diagnostic_metrics = compute_metrics(y[test_idx], diagnostic_prediction)
        diagnostic_alpha = float(diagnostic_search.best_params_["ridge__alpha"])
        plot_scatter(y[test_idx], diagnostic_prediction)
    else:
        # A full Iowa 2024 run has hundreds of date-pass groups. This fallback
        # only supports tiny smoke-test data sets.
        train_idx = np.arange(len(X))
        test_idx = np.array([], dtype=int)
        diagnostic_search = None
        diagnostic_metrics = {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "r2": np.nan, "n": 0}
        diagnostic_alpha = np.nan

    # Select alpha again using all 2024 groups, then refit on all meta rows.
    final_search = fit_grouped_search(X, y, groups)
    final_pipeline = final_search.best_estimator_
    best_alpha = float(final_search.best_params_["ridge__alpha"])

    scaler = final_pipeline.named_steps["scaler"]
    ridge = final_pipeline.named_steps["ridge"]
    coefficients = pd.DataFrame(
        {
            "feature": cfg.META_FEATURE_COLUMNS,
            "coefficient": ridge.coef_,
            "scaled_std": scaler.scale_,
        }
    )
    coefficients["abs_coef"] = coefficients["coefficient"].abs()
    coefficients = coefficients.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    coefficients.to_csv(COEFFICIENT_PATH, index=False)

    bundle = {
        "pipeline": final_pipeline,
        "feature_columns": list(cfg.META_FEATURE_COLUMNS),
        "base_prediction_columns": list(cfg.BASE_PREDICTION_COLUMNS),
        "meta_extra_features": list(cfg.META_EXTRA_FEATURES),
        "best_alpha": best_alpha,
        "random_seed": cfg.RANDOM_SEED,
        "training_year": cfg.VALIDATION_YEARS[0],
        "holdout_mode": cfg.STACKING_HOLDOUT_MODE,
    }
    joblib.dump(bundle, MODEL_PATH)

    metadata = {
        key: value
        for key, value in bundle.items()
        if key != "pipeline"
    }
    metadata.update(
        {
            "n_rows": int(len(meta)),
            "n_date_pass_groups": int(groups.nunique()),
            "diagnostic_train_rows": int(len(train_idx)),
            "diagnostic_test_rows": int(len(test_idx)),
            "diagnostic_best_alpha": diagnostic_alpha,
            "diagnostic_metrics": diagnostic_metrics,
        }
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    report = [
        "Aligned ridge stacking meta-model report",
        "=" * 54,
        f"Project seed: {cfg.RANDOM_SEED}",
        f"Meta-training rows: {len(meta):,}",
        f"Date-pass groups: {groups.nunique():,}",
        f"Features: {len(cfg.META_FEATURE_COLUMNS)}",
        f"Diagnostic grouped alpha: {diagnostic_alpha}",
        f"Final grouped-CV alpha: {best_alpha}",
        "",
        "Grouped diagnostic metrics",
        "-" * 54,
        *[
            f"{name}: {value:.6f}" if isinstance(value, float) else f"{name}: {value}"
            for name, value in diagnostic_metrics.items()
        ],
        "",
        "Standardized coefficients",
        "-" * 54,
        coefficients[["feature", "coefficient"]].to_string(index=False),
    ]
    REPORT_PATH.write_text("\n".join(report))

    print("\n".join(report[:16]))
    print("\nSaved:")
    print(f"  {MODEL_PATH}")
    print(f"  {COEFFICIENT_PATH}")
    print(f"  {METADATA_PATH}")
    print(f"  {REPORT_PATH}")
    print(f"  {SCATTER_PATH}")


if __name__ == "__main__":
    main()
