from pathlib import Path
import importlib.util
import re

import pandas as pd
import geopandas as gpd
import matplotlib

# Use non-interactive backend so this also works on HPC.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# 0. Load config
# ============================================================

def load_config():
    """Load 00_config.py even though the filename starts with a number."""
    config_path = Path(__file__).resolve().parent.parent / "00_config.py"

    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# 1. Settings
# ============================================================

# Folder containing daily PTA CSV files.
PTA_DIR = cfg.IEM_PTA_DIR

# Output folder for maps.
FIG_DIR = PTA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Choose a few dates to visualize.
# Use YYYY-MM-DD format.
SELECT_DATES = [
    "2020-01-01",
    "2020-07-01",
    "2025-07-01",
]

# Choose variables to visualize.
# These should match columns in the daily PTA CSV files.
SELECT_VARIABLES = [
    "precip_pta",
    "et_pta",
    "soil12vwc_pta",
    "soil24vwc_pta",
]

# If True, use the same color scale across selected dates
# for the same variable.
USE_SHARED_SCALE_ACROSS_DATES = True


# ============================================================
# 2. File helpers
# ============================================================

def list_pta_files() -> list[Path]:
    """List daily PTA CSV files."""
    files = sorted(PTA_DIR.glob("iem_pta_smap_lattice_*.csv"))

    if not files:
        raise FileNotFoundError(f"No daily PTA CSV files found in:\n{PTA_DIR}")

    return files


def date_to_yyyymmdd(date_string: str) -> str:
    """Convert YYYY-MM-DD to YYYYMMDD."""
    return pd.to_datetime(date_string).strftime("%Y%m%d")


def extract_date_from_file(path: Path) -> str:
    """Extract date from filename as YYYY-MM-DD."""
    match = re.search(r"iem_pta_smap_lattice_(\d{8})\.csv", path.name)

    if match is None:
        raise ValueError(f"Could not extract date from filename: {path.name}")

    return pd.to_datetime(match.group(1), format="%Y%m%d").strftime("%Y-%m-%d")


def get_files_for_selected_dates(files: list[Path], selected_dates: list[str]) -> list[Path]:
    """Return files matching selected dates."""
    file_map = {
        extract_date_from_file(path): path
        for path in files
    }

    selected_files = []

    for date_string in selected_dates:
        date_string = pd.to_datetime(date_string).strftime("%Y-%m-%d")

        if date_string in file_map:
            selected_files.append(file_map[date_string])
        else:
            print(f"Warning: no PTA file found for {date_string}")

    if not selected_files:
        print("No selected dates were found. Falling back to first 3 available files.")
        selected_files = files[:3]

    return selected_files


# ============================================================
# 3. Load daily PTA file
# ============================================================

def load_pta_csv(path: Path) -> gpd.GeoDataFrame:
    """
    Load one daily PTA CSV and convert geometry_wkt to geometry.
    """
    df = pd.read_csv(path)

    if "geometry_wkt" not in df.columns:
        raise ValueError(f"File has no geometry_wkt column:\n{path}")

    geometry = gpd.GeoSeries.from_wkt(df["geometry_wkt"], crs=f"EPSG:{cfg.CRS_EASE}")

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs=f"EPSG:{cfg.CRS_EASE}",
    )

    return gdf


# ============================================================
# 4. Plot helpers
# ============================================================

def clean_axis(ax, title: str) -> None:
    """Apply clean map formatting."""
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_axis_off()
    ax.set_aspect("equal")


def save_pdf(fig, filename: str) -> None:
    """Save figure as PDF."""
    out_path = FIG_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def available_variables(gdf: gpd.GeoDataFrame, requested_vars: list[str]) -> list[str]:
    """Keep only variables that exist in the data."""
    existing = []

    for var in requested_vars:
        if var in gdf.columns:
            existing.append(var)
        else:
            print(f"Warning: variable not found and skipped: {var}")

    return existing


# ============================================================
# 5. Figure A: multiple variables for one date
# ============================================================

def plot_variables_for_one_date(path: Path, variables: list[str]) -> None:
    """
    Plot several PTA variables for one date.
    """
    date_string = extract_date_from_file(path)
    gdf = load_pta_csv(path)

    variables = available_variables(gdf, variables)

    if not variables:
        print(f"No requested variables available for {date_string}. Skipping.")
        return

    n = len(variables)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4.2 * n, 5.0),
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, variables):
        gdf.plot(
            column=var,
            ax=ax,
            legend=True,
            linewidth=0.0,
            cmap="viridis",
            missing_kwds={
                "color": "lightgray",
                "label": "Missing",
            },
        )

        clean_axis(ax, var)

    fig.suptitle(f"IEM PTA Kriged Variables on SMAP Lattice — {date_string}", fontsize=14)

    save_pdf(fig, f"pta_variables_{date_to_yyyymmdd(date_string)}.pdf")


# ============================================================
# 6. Figure B: one variable across multiple dates
# ============================================================

def plot_one_variable_across_dates(paths: list[Path], variable: str) -> None:
    """
    Plot one variable for multiple dates side by side.

    This is useful for checking whether daily maps change sensibly.
    """
    gdfs = []
    dates = []

    for path in paths:
        gdf = load_pta_csv(path)
        date_string = extract_date_from_file(path)

        if variable not in gdf.columns:
            print(f"Warning: {variable} not found in {path.name}. Skipping.")
            continue

        gdfs.append(gdf)
        dates.append(date_string)

    if not gdfs:
        print(f"No files contained {variable}. Skipping.")
        return

    if USE_SHARED_SCALE_ACROSS_DATES:
        values = pd.concat([gdf[variable] for gdf in gdfs], ignore_index=True)
        vmin = values.quantile(0.02)
        vmax = values.quantile(0.98)
    else:
        vmin = None
        vmax = None

    n = len(gdfs)

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(4.2 * n, 5.0),
        constrained_layout=True,
    )

    if n == 1:
        axes = [axes]

    for ax, gdf, date_string in zip(axes, gdfs, dates):
        gdf.plot(
            column=variable,
            ax=ax,
            legend=True,
            linewidth=0.0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            missing_kwds={
                "color": "lightgray",
                "label": "Missing",
            },
        )

        clean_axis(ax, date_string)

    fig.suptitle(f"{variable} Across Selected Dates", fontsize=14)

    save_pdf(fig, f"pta_{variable}_across_dates.pdf")


# ============================================================
# 7. Quick summary
# ============================================================

def print_file_summary(path: Path) -> None:
    """
    Print basic information for one daily PTA file.
    """
    gdf = load_pta_csv(path)
    date_string = extract_date_from_file(path)

    pta_cols = [c for c in gdf.columns if c.endswith("_pta")]

    print("\nDaily PTA file summary")
    print("-" * 60)
    print(f"Date:       {date_string}")
    print(f"File:       {path.name}")
    print(f"Rows:       {len(gdf)}")
    print(f"CRS:        {gdf.crs}")
    print(f"PTA cols:   {pta_cols}")
    print("-" * 60)


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    files = list_pta_files()
    selected_files = get_files_for_selected_dates(files, SELECT_DATES)

    print(f"PTA folder: {PTA_DIR}")
    print(f"Figure folder: {FIG_DIR}")
    print(f"Daily files found: {len(files)}")
    print(f"Selected files: {[p.name for p in selected_files]}")

    # Print one file summary.
    print_file_summary(selected_files[0])

    # Plot multiple variables for each selected date.
    for path in selected_files:
        plot_variables_for_one_date(path, SELECT_VARIABLES)

    # Plot each selected variable across dates.
    for variable in SELECT_VARIABLES:
        plot_one_variable_across_dates(selected_files, variable)

    print("\nDone.")


if __name__ == "__main__":
    main()