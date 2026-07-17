#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta
import importlib.util

import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# 0. USER SETTINGS
# ============================================================

START_DATE = "2020-01-01"   # first day to plot
N_DAYS = 50                # number of days to plot (change this)
PASS_TO_PLOT = "am"         # "am" or "pm"

# What to include in each daily PDF:
INCLUDE_SMAP = True         # include soil_moisture
INCLUDE_PTA = True          # include *_pta columns

# Usually you only want the actual kriged variables,
# not the variance or n_samples columns.
INCLUDE_PTA_VAR = False     # include *_pta_var if True
INCLUDE_N_SAMPLES = False   # include *_n_samples if True

# Reversed colormap: higher values = darker colors
CMAP = "viridis_r"

# Use robust limits so outliers do not dominate the color scale
USE_ROBUST_COLOR_LIMITS = True
LOW_Q = 0.02
HIGH_Q = 0.98

# Optional manual limits. Leave as None for automatic.
VMIN = None
VMAX = None

# Light borders for polygons
DRAW_BORDERS = False
BORDER_COLOR = "white"
BORDER_WIDTH = 0.05

# Output folder name
OUTPUT_TAG = "complete_all_variables"


# ============================================================
# 1. VARIABLE NOTES
# ============================================================

"""
This script now uses only the COMPLETE daily files.

It creates ONE PDF PER DAY, and inside each daily PDF it plots:
- soil_moisture
- all kriged IEM PTA variables (columns ending in _pta)

By default it does NOT plot:
- *_pta_var
- *_n_samples

You can turn those on above if you want.

Typical actual plotted variables are:
- soil_moisture
- precip_pta
- rh_pta
- speed_pta
- gust_pta
- et_pta
- soil04tn_pta
- soil04t_pta
- soil04tx_pta
- soil12tn_pta
- soil12t_pta
- soil12tx_pta
- soil12vwc_pta
- soil24tn_pta
- soil24t_pta
- soil24tx_pta
- soil24vwc_pta
- soil50tn_pta
- soil50t_pta
- soil50tx_pta
- soil50vwc_pta
"""


# ============================================================
# 2. LOAD CONFIG
# ============================================================

def load_config():
    config_path = Path(__file__).resolve().parent.parent / "00_config.py"

    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# 3. PATH HELPERS
# ============================================================

def get_gap_filling_dir() -> Path:
    if hasattr(cfg, "GAP_FILLING_DIR"):
        return Path(cfg.GAP_FILLING_DIR)
    return Path(cfg.PROCESSED_DIR) / "smap_gap_filling"


def get_full_smap_iem_dir() -> Path:
    if hasattr(cfg, "FULL_SMAP_IEM_DIR"):
        return Path(cfg.FULL_SMAP_IEM_DIR)
    return get_gap_filling_dir() / "03_full_smap_iem_data"


FULL_DIR = get_full_smap_iem_dir()

FIG_DIR = (
    FULL_DIR
    / "figures"
    / f"{OUTPUT_TAG}_{PASS_TO_PLOT}_{START_DATE.replace('-', '')}_n{N_DAYS}"
)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. DATE / FILE HELPERS
# ============================================================

def make_dates(start_date: str, n_days: int) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def date_to_yyyymmdd(date_string: str) -> str:
    return pd.to_datetime(date_string).strftime("%Y%m%d")


def get_complete_file(date_string: str, pass_name: str) -> Path:
    yyyymmdd = date_to_yyyymmdd(date_string)
    filename = f"smap_iem_{pass_name.lower()}_complete_{yyyymmdd}.csv"
    return FULL_DIR / pass_name.lower() / "complete" / filename


# ============================================================
# 5. LOADING DAILY DATA
# ============================================================

def load_daily_gdf(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found:\n{path}")

    df = pd.read_csv(path)

    if "geometry_wkt" not in df.columns:
        raise ValueError(f"'geometry_wkt' column not found in:\n{path}")

    geometry = gpd.GeoSeries.from_wkt(
        df["geometry_wkt"],
        crs=f"EPSG:{getattr(cfg, 'CRS_EASE', 6933)}"
    )

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs=f"EPSG:{getattr(cfg, 'CRS_EASE', 6933)}"
    )

    return gdf


def load_requested_days() -> dict[str, gpd.GeoDataFrame | None]:
    records = {}
    for date_string in make_dates(START_DATE, N_DAYS):
        path = get_complete_file(date_string, PASS_TO_PLOT)
        if not path.exists():
            print(f"[missing file] {path}")
            records[date_string] = None
            continue

        try:
            records[date_string] = load_daily_gdf(path)
            print(f"[loaded] {path}")
        except Exception as exc:
            print(f"[failed] {path}: {exc}")
            records[date_string] = None

    return records


# ============================================================
# 6. VARIABLE SELECTION
# ============================================================

def get_pta_value_columns(columns: list[str]) -> list[str]:
    """
    Return actual kriged PTA value columns, e.g. precip_pta, rh_pta, soil04t_pta.
    Excludes *_pta_var and *_n_samples.
    """
    cols = []

    # Prefer cfg.IEM_VARIABLES ordering if available
    if hasattr(cfg, "IEM_VARIABLES"):
        ordered = []
        for base in cfg.IEM_VARIABLES:
            c = f"{base}_pta"
            if c in columns:
                ordered.append(c)
        cols.extend(ordered)

    # Add any remaining *_pta columns not already included
    remaining = [
        c for c in columns
        if c.endswith("_pta")
        and not c.endswith("_pta_var")
        and c not in cols
    ]
    cols.extend(sorted(remaining))

    return cols


def get_pta_var_columns(columns: list[str]) -> list[str]:
    return sorted([c for c in columns if c.endswith("_pta_var")])


def get_nsample_columns(columns: list[str]) -> list[str]:
    return sorted([c for c in columns if c.endswith("_n_samples")])


def choose_variables(gdf: gpd.GeoDataFrame) -> list[str]:
    cols = list(gdf.columns)
    variables = []

    if INCLUDE_SMAP and "soil_moisture" in cols:
        variables.append("soil_moisture")

    if INCLUDE_PTA:
        variables.extend(get_pta_value_columns(cols))

    if INCLUDE_PTA_VAR:
        variables.extend(get_pta_var_columns(cols))

    if INCLUDE_N_SAMPLES:
        variables.extend(get_nsample_columns(cols))

    # remove duplicates while preserving order
    seen = set()
    out = []
    for v in variables:
        if v not in seen:
            out.append(v)
            seen.add(v)

    return out


# ============================================================
# 7. COLOR LIMITS
# ============================================================

def get_numeric_values(gdf: gpd.GeoDataFrame | None, variable: str) -> pd.Series:
    if gdf is None or variable not in gdf.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(gdf[variable], errors="coerce")


def compute_variable_limits(records: dict, variables: list[str]) -> dict[str, tuple[float | None, float | None]]:
    limits = {}

    for var in variables:
        if VMIN is not None and VMAX is not None:
            limits[var] = (float(VMIN), float(VMAX))
            continue

        vals_list = []
        for _, gdf in records.items():
            vals = get_numeric_values(gdf, var).dropna()
            if len(vals) > 0:
                vals_list.append(vals)

        if not vals_list:
            limits[var] = (None, None)
            continue

        combined = pd.concat(vals_list, ignore_index=True)

        if USE_ROBUST_COLOR_LIMITS:
            vmin = combined.quantile(LOW_Q)
            vmax = combined.quantile(HIGH_Q)
        else:
            vmin = combined.min()
            vmax = combined.max()

        if pd.isna(vmin) or pd.isna(vmax):
            limits[var] = (None, None)
            continue

        vmin = float(vmin)
        vmax = float(vmax)

        if vmin == vmax:
            eps = 1e-9 if vmin == 0 else abs(vmin) * 1e-6
            vmin -= eps
            vmax += eps

        limits[var] = (vmin, vmax)

    return limits


# ============================================================
# 8. PLOTTING
# ============================================================

def pretty_name(var: str) -> str:
    return var.replace("_", " ")


def plot_one_variable(
    gdf: gpd.GeoDataFrame,
    variable: str,
    date_string: str,
    pass_name: str,
    vmin: float | None,
    vmax: float | None,
):
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f8fa")

    if gdf is None or len(gdf) == 0:
        ax.text(
            0.5, 0.5, "No data loaded",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=16
        )
        ax.set_axis_off()
        return fig

    if variable not in gdf.columns:
        ax.text(
            0.5, 0.5, f"Variable not found:\n{variable}",
            ha="center", va="center",
            transform=ax.transAxes, fontsize=16
        )
        ax.set_axis_off()
        return fig

    plot_gdf = gdf.copy()
    plot_gdf[variable] = pd.to_numeric(plot_gdf[variable], errors="coerce")

    n_total = len(plot_gdf)
    n_nonmissing = int(plot_gdf[variable].notna().sum())
    n_missing = int(plot_gdf[variable].isna().sum())

    edgecolor = BORDER_COLOR if DRAW_BORDERS else "none"
    linewidth = BORDER_WIDTH if DRAW_BORDERS else 0.0

    if n_nonmissing == 0:
        plot_gdf.plot(
            ax=ax,
            facecolor="#d9dee5",
            edgecolor=edgecolor,
            linewidth=linewidth
        )
    else:
        plot_gdf.plot(
            column=variable,
            ax=ax,
            cmap=CMAP,
            vmin=vmin,
            vmax=vmax,
            edgecolor=edgecolor,
            linewidth=linewidth,
            missing_kwds={
                "color": "#d9dee5",
                "edgecolor": "none",
                "label": "NA"
            }
        )

    ax.set_axis_off()
    ax.set_aspect("equal")

    title = pretty_name(variable)
    subtitle = f"{date_string} | {pass_name.upper()} | complete lattice"
    note = f"non-missing: {n_nonmissing:,} | NA: {n_missing:,} | total cells: {n_total:,}"

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18, color="#263238")

    fig.text(
        0.5, 0.925, subtitle,
        ha="center", va="center",
        fontsize=11, color="#455a64"
    )
    fig.text(
        0.5, 0.895, note,
        ha="center", va="center",
        fontsize=10, color="#607d8b"
    )

    if n_nonmissing > 0 and vmin is not None and vmax is not None:
        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=CMAP)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.045,
            pad=0.035,
            shrink=0.75
        )
        cbar.set_label(variable, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.88])
    return fig


# ============================================================
# 9. SAVE DAILY PDFS
# ============================================================

def save_daily_pdfs(records: dict, variables: list[str], limits: dict[str, tuple[float | None, float | None]]) -> None:
    for date_string, gdf in records.items():
        if gdf is None:
            print(f"[skip] {date_string}: no data")
            continue

        out_pdf = FIG_DIR / f"complete_{PASS_TO_PLOT}_{date_to_yyyymmdd(date_string)}_all_variables.pdf"

        with PdfPages(out_pdf) as pdf:
            for var in variables:
                vmin, vmax = limits.get(var, (None, None))
                fig = plot_one_variable(
                    gdf=gdf,
                    variable=var,
                    date_string=date_string,
                    pass_name=PASS_TO_PLOT,
                    vmin=vmin,
                    vmax=vmax,
                )
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        print(f"[saved] {out_pdf}")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    print("\nLoading complete daily files")
    print("-" * 80)
    print(f"Start date : {START_DATE}")
    print(f"N days     : {N_DAYS}")
    print(f"Pass       : {PASS_TO_PLOT}")
    print(f"Output dir : {FIG_DIR}")
    print("-" * 80)

    records = load_requested_days()

    first_gdf = next((g for g in records.values() if g is not None), None)
    if first_gdf is None:
        raise RuntimeError("No daily complete files were loaded.")

    variables = choose_variables(first_gdf)
    if not variables:
        raise RuntimeError("No variables were selected for plotting.")

    print("\nVariables to plot:")
    for v in variables:
        print(f"  - {v}")

    limits = compute_variable_limits(records, variables)

    print("\nColor limits:")
    for v in variables:
        vmin, vmax = limits[v]
        print(f"  - {v}: {vmin} to {vmax}")

    save_daily_pdfs(records, variables, limits)

    print("\nDone.")


if __name__ == "__main__":
    main()