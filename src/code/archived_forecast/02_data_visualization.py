#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_data_visualization.py

Purpose
-------
Visualize yearly weather forecast data for Iowa from yearly parquet files.

What this script makes
----------------------
1) Iowa-wide time series plots (monthly average within each year)
2) Optional spatial maps of yearly mean forecast values

Expected input
--------------
Yearly parquet files in:
    src/data/processed/archived_weather/combined/yearly/

Example filenames:
    weather_models_iowa_2020.parquet
    weather_models_iowa_2021.parquet
    ...
    weather_models_iowa_2025.parquet

Township shapefile:
    src/data/raw/townships/civil_townships_a_ia.shp

Notes
-----
- Viridis color theme is used throughout.
- If polygon WKT is present in the parquet, this script will use it for map plots.
- Otherwise, it falls back to centroid point plots using latitude/longitude.

Run
---
conda activate py312
python src/code/archived_forecast/02_data_visualization.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# USER SETTINGS
# ============================================================

# Years to visualize
YEARS_TO_PLOT = [2020, 2021, 2022, 2023, 2024, 2025]

# If None, all available models in the file will be plotted
MODELS_TO_PLOT = None
# Example:
# MODELS_TO_PLOT = ["hrrr", "gfs", "rap", "nam"]

# Lead hours to include in plots
LEAD_HOURS_TO_PLOT = [24, 48]
# If your data also contains analysis/nowcast, you can add 0

# Main forecast variables to visualize as yearly time series
VARIABLES_TO_PLOT = [
    "temperature_c",
    "dewpoint_c",
    "relative_humidity_percent",
    "wind_speed_10m",
    "wind_gust",
    "precip_accum_mm",
    "precip_rate",
    "surface_pressure_hpa",
    "downward_shortwave_radiation",
    "latent_heat_flux",
    "soil_water_model",   # if available
]

# Spatial maps are optional because they can take longer
MAKE_SPATIAL_MAPS = True

# Fewer variables for maps is usually better
SPATIAL_VARIABLES = [
    "temperature_c",
    "precip_accum_mm",
    "wind_speed_10m",
    "soil_water_model",   # if available
]

# Use monthly averages for a cleaner yearly plot
# If False, plots daily averages instead
USE_MONTHLY_AVERAGE = True

# Figure size and DPI
FIGSIZE = (12, 6)
MAP_FIGSIZE = (10, 8)
DPI = 300

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_ARCHIVED_WEATHER = (
    PROJECT_ROOT / "src" / "data" / "processed" / "archived_weather"
)
COMBINED_YEARLY_DIR = PROCESSED_ARCHIVED_WEATHER / "combined" / "yearly"

FIGURES_DIR = PROCESSED_ARCHIVED_WEATHER / "figures" / "yearly"

TOWNSHIP_SHP = (
    PROJECT_ROOT / "src" / "data" / "raw" / "townships" / "civil_townships_a_ia.shp"
)

# ============================================================
# VIRIDIS THEME HELPERS
# ============================================================

def get_viridis_colors(n):
    """Return n colors from the viridis colormap."""
    if n <= 1:
        return [plt.cm.viridis(0.6)]
    vals = np.linspace(0.1, 0.9, n)
    return [plt.cm.viridis(v) for v in vals]


def set_plot_style():
    """Set a clean matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "grid.color": "#d9d9d9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
    })

# ============================================================
# FILE FINDERS / LOADERS
# ============================================================

def find_yearly_parquet(year: int) -> Path:
    """
    Find the yearly parquet file for a given year.
    Expected pattern includes the year in the filename.
    """
    if not COMBINED_YEARLY_DIR.exists():
        raise FileNotFoundError(
            f"Yearly combined directory not found:\n{COMBINED_YEARLY_DIR}"
        )

    candidates = sorted(COMBINED_YEARLY_DIR.glob(f"*{year}*.parquet"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No yearly parquet file found for {year} in:\n{COMBINED_YEARLY_DIR}"
        )

    # Use the first match
    return candidates[0]


def load_year_data(year: int) -> pd.DataFrame:
    """Load one year's parquet file."""
    path = find_yearly_parquet(year)
    print(f"\nReading yearly parquet for {year}:\n{path}\n")
    df = pd.read_parquet(path)

    # Standardize date column
    if "valid_date" in df.columns:
        df["valid_date"] = pd.to_datetime(df["valid_date"])
    elif "valid_time" in df.columns:
        df["valid_date"] = pd.to_datetime(df["valid_time"]).dt.normalize()
    else:
        raise ValueError("Expected a 'valid_date' or 'valid_time' column.")

    # Standardize longitude column
    if "longitude" not in df.columns and "longitude_180" in df.columns:
        df["longitude"] = df["longitude_180"]

    # Ensure model and lead_hour exist
    for col in ["model", "lead_hour"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_iowa_townships() -> gpd.GeoDataFrame:
    """Load the Iowa township shapefile."""
    if not TOWNSHIP_SHP.exists():
        raise FileNotFoundError(
            f"Township shapefile not found:\n{TOWNSHIP_SHP}"
        )

    gdf = gpd.read_file(TOWNSHIP_SHP)

    # Reproject to WGS84 if needed
    if gdf.crs is None:
        print("Warning: township shapefile CRS is missing. Assuming EPSG:4326.")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    return gdf

# ============================================================
# DATA PREP
# ============================================================

def keep_requested_models(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to selected models if requested."""
    if MODELS_TO_PLOT is None:
        return df.copy()

    return df[df["model"].isin(MODELS_TO_PLOT)].copy()


def keep_requested_leads(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to selected lead hours."""
    return df[df["lead_hour"].isin(LEAD_HOURS_TO_PLOT)].copy()


def available_variables(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return only variables that exist in the dataframe."""
    return [c for c in candidates if c in df.columns]


def prepare_time_aggregation(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Create Iowa-wide daily or monthly average dataframe for a variable.
    Averaging is across all grid cells in Iowa.
    """
    if USE_MONTHLY_AVERAGE:
        temp = df.copy()
        temp["plot_date"] = temp["valid_date"].dt.to_period("M").dt.to_timestamp()

        out = (
            temp.groupby(["plot_date", "model", "lead_hour"], as_index=False)[var]
            .mean()
            .sort_values(["plot_date", "model", "lead_hour"])
        )
    else:
        temp = df.copy()
        temp["plot_date"] = temp["valid_date"]

        out = (
            temp.groupby(["plot_date", "model", "lead_hour"], as_index=False)[var]
            .mean()
            .sort_values(["plot_date", "model", "lead_hour"])
        )

    return out


def get_geometry_column(df: pd.DataFrame):
    """
    Detect a possible polygon WKT column.
    """
    for col in ["polygon_wkt", "grid_wkt", "geometry_wkt", "wkt"]:
        if col in df.columns:
            return col
    return None

# ============================================================
# YEARLY TIME SERIES PLOTS
# ============================================================

def plot_yearly_time_series(df: pd.DataFrame, year: int, outdir: Path):
    """
    Make yearly time series plots for each selected variable.
    One plot per variable; colors distinguish model/lead combinations.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    variables = available_variables(df, VARIABLES_TO_PLOT)
    if len(variables) == 0:
        print(f"No requested variables found for year {year}.")
        return

    for var in variables:
        plot_df = prepare_time_aggregation(df, var)

        combos = (
            plot_df[["model", "lead_hour"]]
            .drop_duplicates()
            .sort_values(["model", "lead_hour"])
            .reset_index(drop=True)
        )

        colors = get_viridis_colors(len(combos))

        fig, ax = plt.subplots(figsize=FIGSIZE)

        for i, row in combos.iterrows():
            m = row["model"]
            lead = row["lead_hour"]

            sub = plot_df[
                (plot_df["model"] == m) &
                (plot_df["lead_hour"] == lead)
            ].copy()

            label = f"{m} | F{lead}"
            ax.plot(
                sub["plot_date"],
                sub[var],
                label=label,
                color=colors[i],
                linewidth=2
            )

        ax.set_title(
            f"{var} | Iowa-wide average forecast | {year}",
            pad=12
        )
        ax.set_xlabel("Date")
        ax.set_ylabel(var)

        # Date formatting
        if USE_MONTHLY_AVERAGE:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        plt.xticks(rotation=45)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()

        out_path = outdir / f"{var}_timeseries_{year}.png"
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")

# ============================================================
# YEARLY SPATIAL MAPS
# ============================================================

def build_spatial_gdf(df: pd.DataFrame, value_col: str) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame for mapping.
    If polygon WKT exists, use polygons.
    Otherwise use point centroids from lon/lat.
    """
    geom_col = get_geometry_column(df)

    if geom_col is not None:
        # Average by model, lead, and polygon text
        id_cols = ["model", "lead_hour", geom_col]
        if "latitude" in df.columns:
            id_cols.append("latitude")
        if "longitude" in df.columns:
            id_cols.append("longitude")

        grp = (
            df.groupby(id_cols, as_index=False)[value_col]
            .mean()
        )

        gdf = gpd.GeoDataFrame(
            grp,
            geometry=grp[geom_col].apply(wkt.loads),
            crs="EPSG:4326"
        )
    else:
        if "latitude" not in df.columns or "longitude" not in df.columns:
            raise ValueError(
                "No polygon WKT found, and latitude/longitude columns are missing."
            )

        grp = (
            df.groupby(["model", "lead_hour", "latitude", "longitude"], as_index=False)[value_col]
            .mean()
        )

        gdf = gpd.GeoDataFrame(
            grp,
            geometry=gpd.points_from_xy(grp["longitude"], grp["latitude"]),
            crs="EPSG:4326"
        )

    return gdf


def plot_yearly_spatial_maps(df: pd.DataFrame, year: int, townships: gpd.GeoDataFrame, outdir: Path):
    """
    Make yearly mean spatial maps for selected variables.
    One map per variable per model per lead hour.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    variables = available_variables(df, SPATIAL_VARIABLES)
    if len(variables) == 0:
        print(f"No spatial variables found for year {year}.")
        return

    for var in variables:
        spatial_gdf = build_spatial_gdf(df, var)

        combos = (
            spatial_gdf[["model", "lead_hour"]]
            .drop_duplicates()
            .sort_values(["model", "lead_hour"])
            .reset_index(drop=True)
        )

        for _, row in combos.iterrows():
            model_name = row["model"]
            lead = row["lead_hour"]

            sub = spatial_gdf[
                (spatial_gdf["model"] == model_name) &
                (spatial_gdf["lead_hour"] == lead)
            ].copy()

            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=MAP_FIGSIZE)

            # Plot township boundaries for context
            townships.boundary.plot(ax=ax, color="lightgray", linewidth=0.3)

            # Plot weather field
            if sub.geom_type.iloc[0] in ["Polygon", "MultiPolygon"]:
                sub.plot(
                    ax=ax,
                    column=var,
                    cmap="viridis",
                    legend=True,
                    linewidth=0.0,
                    alpha=0.9
                )
            else:
                sub.plot(
                    ax=ax,
                    column=var,
                    cmap="viridis",
                    legend=True,
                    markersize=8,
                    alpha=0.85
                )

            ax.set_title(
                f"{var} | Yearly mean | {model_name} | F{lead} | {year}",
                pad=12
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.tight_layout()

            out_path = outdir / f"{var}_map_{model_name}_F{lead}_{year}.png"
            plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
            plt.close()

            print(f"Saved: {out_path}")

# ============================================================
# YEARLY SUMMARY CSV
# ============================================================

def save_yearly_summary_table(df: pd.DataFrame, year: int, outdir: Path):
    """
    Save a tidy yearly summary table:
    monthly Iowa-wide means by model and lead hour.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    vars_found = available_variables(df, VARIABLES_TO_PLOT)
    if len(vars_found) == 0:
        return

    temp = df.copy()
    temp["month"] = temp["valid_date"].dt.month

    summary = (
        temp.groupby(["month", "model", "lead_hour"], as_index=False)[vars_found]
        .mean()
        .sort_values(["month", "model", "lead_hour"])
    )

    out_path = outdir / f"yearly_summary_{year}.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    set_plot_style()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load townships only once if maps are requested
    townships = None
    if MAKE_SPATIAL_MAPS:
        townships = load_iowa_townships()

    for year in YEARS_TO_PLOT:
        print("=" * 70)
        print(f"Processing visualizations for year: {year}")
        print("=" * 70)

        try:
            df = load_year_data(year)
        except Exception as e:
            print(f"Skipping year {year}. Reason:\n{e}\n")
            continue

        df = keep_requested_models(df)
        df = keep_requested_leads(df)

        if df.empty:
            print(f"No data left after filtering for year {year}. Skipping.\n")
            continue

        year_outdir = FIGURES_DIR / str(year)
        year_outdir.mkdir(parents=True, exist_ok=True)

        # 1) Iowa-wide yearly time series
        plot_yearly_time_series(
            df=df,
            year=year,
            outdir=year_outdir / "timeseries"
        )

        # 2) Optional spatial maps
        if MAKE_SPATIAL_MAPS and townships is not None:
            try:
                plot_yearly_spatial_maps(
                    df=df,
                    year=year,
                    townships=townships,
                    outdir=year_outdir / "maps"
                )
            except Exception as e:
                print(f"Spatial maps failed for {year}.\nReason: {e}\n")

        # 3) Save yearly summary table
        save_yearly_summary_table(
            df=df,
            year=year,
            outdir=year_outdir / "tables"
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()