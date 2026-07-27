#!/usr/bin/env python3
"""Shared helpers for SMAP gap-filling visualization scripts.

This module keeps all visualization scripts connected to the same ``00_config.py``
and the same output structure used by the corrected modeling workflow.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_gapfill_code_dir(start: Path | None = None) -> Path:
    """Find the folder containing ``00_config.py``.

    The visualization files normally live in ``smap_gap_filling/Visualization``.
    Searching upward also makes them work if the folder is renamed or moved one
    level within the project.
    """
    current = (start or Path(__file__).resolve()).resolve()
    base = current if current.is_dir() else current.parent

    for folder in [base, *base.parents]:
        if (folder / "00_config.py").is_file():
            return folder

    raise FileNotFoundError(
        "Could not find 00_config.py. Keep the Visualization folder inside "
        "src/code/smap_gap_filling, or set the scripts beside that configuration."
    )


def load_config() -> ModuleType:
    code_dir = find_gapfill_code_dir()
    config_path = code_dir / "00_config.py"
    spec = importlib.util.spec_from_file_location("smap_gapfill_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load configuration from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cfg = load_config()
CODE_DIR = find_gapfill_code_dir()
VISUALIZATION_ROOT = Path(cfg.GAP_FILLING_DIR) / "09_final_visualization"
VISUALIZATION_ROOT.mkdir(parents=True, exist_ok=True)


def safe_name(value: object, limit: int = 120) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text[:limit] or "unnamed"


def save_figure(
    fig: plt.Figure,
    output_stem: Path,
    *,
    save_png: bool = True,
    save_pdf: bool = True,
    dpi: int = 240,
) -> None:
    """Save one figure using a suffix-free output stem."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    if save_png:
        fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def date_to_yyyymmdd(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y%m%d")


def normalize_date(value: object) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def complete_file_path(date: object, pass_name: str) -> Path:
    pass_name = pass_name.lower().strip()
    if pass_name not in {"am", "pm"}:
        raise ValueError("pass_name must be 'am' or 'pm'.")
    ymd = date_to_yyyymmdd(date)
    return (
        Path(cfg.FULL_SMAP_IEM_DIR)
        / pass_name
        / "complete"
        / f"smap_iem_{pass_name}_complete_{ymd}.csv"
    )


def gapfilled_file_path(date: object, pass_name: str) -> Path:
    pass_name = pass_name.lower().strip()
    if pass_name not in {"am", "pm"}:
        raise ValueError("pass_name must be 'am' or 'pm'.")
    ymd = date_to_yyyymmdd(date)
    return (
        Path(cfg.FINAL_DIR)
        / pass_name
        / f"smap_iem_{pass_name}_gapfilled_{ymd}.csv"
    )


def iem_pta_file_path(date: object) -> Path:
    ymd = date_to_yyyymmdd(date)
    if hasattr(cfg, "get_iem_pta_daily_csv_path"):
        return Path(cfg.get_iem_pta_daily_csv_path(ymd))
    return Path(cfg.IEM_PTA_DIR) / f"iem_pta_smap_lattice_{ymd}.csv"


def read_spatial_csv(path: Path) -> tuple[pd.DataFrame, gpd.GeoDataFrame | None]:
    """Read a daily CSV and optionally construct polygon geometry.

    Returns both the original DataFrame and a GeoDataFrame when ``geometry_wkt``
    is available. The DataFrame is retained because several final files contain
    categorical status columns that are easiest to work with directly.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()

    if "geometry_wkt" not in frame.columns:
        return frame, None

    geometry = gpd.GeoSeries.from_wkt(
        frame["geometry_wkt"], crs=f"EPSG:{getattr(cfg, 'CRS_EASE', 6933)}"
    )
    geoframe = gpd.GeoDataFrame(
        frame.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs=geometry.crs,
    )
    return frame, geoframe


def read_townships(target_crs=None, *, required: bool = False) -> gpd.GeoDataFrame | None:
    path = Path(getattr(cfg, "TOWNSHIP_SHP_PATH", ""))
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Township shapefile not found: {path}")
        print(f"[warning] Township shapefile not found; boundary overlay skipped: {path}")
        return None

    try:
        townships = gpd.read_file(path, engine="fiona")
    except Exception:
        townships = gpd.read_file(path)

    if townships.crs is None:
        townships = townships.set_crs("EPSG:4326")
    if target_crs is not None and townships.crs != target_crs:
        townships = townships.to_crs(target_crs)
    return townships


def coord_columns(frame: pd.DataFrame, *, prefer_lonlat: bool = False) -> tuple[str, str]:
    lonlat = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("lng", "lat"),
    ]
    projected = [("x", "y"), ("easting", "northing")]
    choices = lonlat + projected if prefer_lonlat else projected + lonlat
    lower = {str(column).lower(): str(column) for column in frame.columns}
    for x_name, y_name in choices:
        x = lower.get(x_name.lower())
        y = lower.get(y_name.lower())
        if x is not None and y is not None:
            return x, y
    raise ValueError("No usable x/y or lon/lat coordinate columns were found.")


def auto_marker_size(n_rows: int) -> float:
    if n_rows <= 500:
        return 42.0
    if n_rows <= 1_500:
        return 28.0
    if n_rows <= 4_000:
        return 17.0
    if n_rows <= 10_000:
        return 9.0
    return 5.0


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def robust_limits(
    series_collection: Iterable[pd.Series],
    *,
    low_quantile: float = 0.02,
    high_quantile: float = 0.98,
) -> tuple[float | None, float | None]:
    parts = [pd.to_numeric(series, errors="coerce").dropna() for series in series_collection]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return None, None
    values = pd.concat(parts, ignore_index=True)
    vmin = float(values.quantile(low_quantile))
    vmax = float(values.quantile(high_quantile))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if vmin == vmax:
        delta = 1e-9 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= delta
        vmax += delta
    return vmin, vmax


_VARIABLE_LABELS = {
    "soil_moisture": "SMAP SM",
    "soil_moisture_filled": "Gap-filled SMAP SM",
    "precip_pta": "IEM precipitation",
    "rh_pta": "IEM relative humidity",
    "speed_pta": "IEM wind speed",
    "gust_pta": "IEM wind gust",
    "et_pta": "IEM evapotranspiration",
    "soil04tn_pta": "4-cm minimum soil temperature",
    "soil04t_pta": "4-cm mean soil temperature",
    "soil04tx_pta": "4-cm maximum soil temperature",
    "soil12tn_pta": "12-cm minimum soil temperature",
    "soil12t_pta": "12-cm mean soil temperature",
    "soil12tx_pta": "12-cm maximum soil temperature",
    "soil12vwc_pta": "12-cm soil VWC",
    "soil24tn_pta": "24-cm minimum soil temperature",
    "soil24t_pta": "24-cm mean soil temperature",
    "soil24tx_pta": "24-cm maximum soil temperature",
    "soil24vwc_pta": "24-cm soil VWC",
    "soil50tn_pta": "50-cm minimum soil temperature",
    "soil50t_pta": "50-cm mean soil temperature",
    "soil50tx_pta": "50-cm maximum soil temperature",
    "soil50vwc_pta": "50-cm soil VWC",
}

_VARIABLE_UNITS = {
    "soil_moisture": r"m$^3$ m$^{-3}$",
    "soil_moisture_filled": r"m$^3$ m$^{-3}$",
    "precip_pta": "input-data units",
    "rh_pta": "%",
    "speed_pta": "input-data units",
    "gust_pta": "input-data units",
    "et_pta": "input-data units",
    "soil12vwc_pta": r"m$^3$ m$^{-3}$",
    "soil24vwc_pta": r"m$^3$ m$^{-3}$",
    "soil50vwc_pta": r"m$^3$ m$^{-3}$",
}


def pretty_variable(column: str) -> str:
    return _VARIABLE_LABELS.get(column, column.replace("_pta", "").replace("_", " "))


def variable_unit(column: str) -> str:
    if column in _VARIABLE_UNITS:
        return _VARIABLE_UNITS[column]
    if "soil" in column and column.endswith("_pta") and "vwc" not in column:
        return "input-data units"
    return ""


def ordered_pta_columns(columns: Iterable[str]) -> list[str]:
    available = set(columns)
    ordered = [f"{name}_pta" for name in getattr(cfg, "IEM_PTA_VARIABLES", [])]
    selected = [column for column in ordered if column in available]
    remaining = sorted(
        column
        for column in available
        if column.endswith("_pta")
        and not column.endswith("_pta_var")
        and column not in selected
    )
    return selected + remaining


def model_family(method: str) -> str:
    method = str(method)
    if method == "stacking":
        return "Stack"
    if method in set(getattr(cfg, "CANDIDATE_ML_MODELS", [])):
        return "ML"
    if method in set(getattr(cfg, "SELECTED_INTERPOLATION_METHODS", [])):
        return "GI"
    return "Other"


def pretty_method(method: str) -> str:
    labels = {
        "stacking": "Ridge stack",
        "xgboost": "XGBoost",
        "hist_gbdt": "Histogram GBDT",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "ffnn_mlp": "FFNN",
        "baseline": "Training-mean baseline",
        "centroid_ordinary_kriging": "Detrended centroid OK",
        "nearest_neighbor_same_day": "Same-day nearest neighbor",
        "regression_kriging": "Regression kriging",
        "observed": "Observed",
        "none": "Unfilled",
    }
    return labels.get(str(method), str(method).replace("_", " "))
