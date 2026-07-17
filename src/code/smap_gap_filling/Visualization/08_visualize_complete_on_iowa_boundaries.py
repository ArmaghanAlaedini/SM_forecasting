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


# ============================================================
# 0. USER SETTINGS
# ============================================================

START_DATE = "2020-03-01"
N_DAYS = 10

PASS_TO_PLOT = "am"

# Use "AUTO" to plot soil_moisture plus all actual *_pta variables.
# Or use a list, e.g. ["soil_moisture", "precip_pta", "soil12vwc_pta"]
VARIABLES_TO_PLOT = "AUTO"

# Higher values darker.
CMAP = "viridis_r"

USE_ROBUST_COLOR_LIMITS = True
LOW_Q = 0.02
HIGH_Q = 0.98

DRAW_PIXEL_BORDERS = False
DRAW_TOWNSHIP_BOUNDARIES = True
DRAW_IOWA_OUTLINE = True

SAVE_PDF = True
SAVE_PNG = False
PNG_DPI = 250


# ============================================================
# 1. LOAD CONFIG
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
# 2. PATHS
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
    / f"complete_on_iowa_boundaries_{PASS_TO_PLOT}_{START_DATE.replace('-', '')}_n{N_DAYS}"
)

FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DATE / FILE HELPERS
# ============================================================

def make_dates(start_date: str, n_days: int) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def date_to_yyyymmdd(date_string: str) -> str:
    return pd.to_datetime(date_string).strftime("%Y%m%d")


def complete_file(date_string: str, pass_name: str) -> Path:
    yyyymmdd = date_to_yyyymmdd(date_string)
    return FULL_DIR / pass_name / "complete" / f"smap_iem_{pass_name}_complete_{yyyymmdd}.csv"


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_complete_gdf(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)

    if "geometry_wkt" not in df.columns:
        raise ValueError(f"geometry_wkt missing in {path}")

    geometry = gpd.GeoSeries.from_wkt(
        df["geometry_wkt"],
        crs=f"EPSG:{getattr(cfg, 'CRS_EASE', 6933)}",
    )

    gdf = gpd.GeoDataFrame(
        df.drop(columns=["geometry_wkt"]),
        geometry=geometry,
        crs=f"EPSG:{getattr(cfg, 'CRS_EASE', 6933)}",
    )

    return gdf


def load_records() -> dict[str, gpd.GeoDataFrame]:
    records = {}

    for date_string in make_dates(START_DATE, N_DAYS):
        path = complete_file(date_string, PASS_TO_PLOT)

        try:
            records[date_string] = load_complete_gdf(path)
            print(f"[loaded] {path}")
        except Exception as exc:
            print(f"[skip] {date_string}: {exc}")

    return records


# ============================================================
# 5. LOAD IOWA / TOWNSHIP BOUNDARIES
# ============================================================

def find_township_shapefile() -> Path | None:
    candidate_names = [
        "TOWNSHIP_SHP",
        "TOWNSHIP_SHAPEFILE",
        "IOWA_TOWNSHIP_SHP",
        "TOWNSHIPS_SHP",
    ]

    for name in candidate_names:
        if hasattr(cfg, name):
            p = Path(getattr(cfg, name))
            if p.exists():
                return p

    search_roots = []

    for name in ["RAW_DIR", "DATA_ROOT", "PROJECT_ROOT"]:
        if hasattr(cfg, name):
            search_roots.append(Path(getattr(cfg, name)))

    for root in search_roots:
        if not root.exists():
            continue

        shp_files = list(root.rglob("*.shp"))

        for p in shp_files:
            lower = p.name.lower()
            if "township" in lower or "civil" in lower:
                return p

    return None


def load_townships(target_crs) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    shp = find_township_shapefile()

    if shp is None:
        print("[warning] Township shapefile not found. Maps will be drawn without Iowa boundaries.")
        return None, None

    townships = gpd.read_file(shp, engine="fiona")

    if townships.crs is None:
        townships = townships.set_crs("EPSG:4326")

    townships = townships.to_crs(target_crs)

    iowa_outline = townships.dissolve()

    return townships, iowa_outline


# ============================================================
# 6. VARIABLE SELECTION
# ============================================================

def choose_variables(first_gdf: gpd.GeoDataFrame) -> list[str]:
    cols = list(first_gdf.columns)

    if VARIABLES_TO_PLOT != "AUTO":
        return list(VARIABLES_TO_PLOT)

    variables = []

    if "soil_moisture" in cols:
        variables.append("soil_moisture")

    if hasattr(cfg, "IEM_VARIABLES"):
        for base in cfg.IEM_VARIABLES:
            c = f"{base}_pta"
            if c in cols:
                variables.append(c)

    remaining_pta = sorted([
        c for c in cols
        if c.endswith("_pta")
        and not c.endswith("_pta_var")
        and c not in variables
    ])

    variables.extend(remaining_pta)

    return variables


# ============================================================
# 7. COLOR LIMITS
# ============================================================

def get_values(gdf: gpd.GeoDataFrame, variable: str) -> pd.Series:
    if gdf is None or variable not in gdf.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(gdf[variable], errors="coerce")


def compute_limits(records: dict[str, gpd.GeoDataFrame], variables: list[str]) -> dict[str, tuple[float | None, float | None]]:
    limits = {}

    for variable in variables:
        all_values = []

        for gdf in records.values():
            vals = get_values(gdf, variable).dropna()
            if len(vals) > 0:
                all_values.append(vals)

        if not all_values:
            limits[variable] = (None, None)
            continue

        combined = pd.concat(all_values, ignore_index=True)

        if USE_ROBUST_COLOR_LIMITS:
            vmin = combined.quantile(LOW_Q)
            vmax = combined.quantile(HIGH_Q)
        else:
            vmin = combined.min()
            vmax = combined.max()

        if pd.isna(vmin) or pd.isna(vmax):
            limits[variable] = (None, None)
            continue

        vmin = float(vmin)
        vmax = float(vmax)

        if vmin == vmax:
            eps = 1e-9 if vmin == 0 else abs(vmin) * 1e-6
            vmin -= eps
            vmax += eps

        limits[variable] = (vmin, vmax)

    return limits


# ============================================================
# 8. PLOTTING
# ============================================================

def pretty_name(variable: str) -> str:
    names = {
        "soil_moisture": "SMAP soil moisture",
        "precip_pta": "IEM PTA precipitation",
        "rh_pta": "IEM PTA relative humidity",
        "speed_pta": "IEM PTA wind speed",
        "gust_pta": "IEM PTA wind gust",
        "et_pta": "IEM PTA evapotranspiration",
    }

    if variable in names:
        return names[variable]

    return variable.replace("_", " ")


def plot_map(
    gdf: gpd.GeoDataFrame,
    townships: gpd.GeoDataFrame | None,
    iowa_outline: gpd.GeoDataFrame | None,
    date_string: str,
    variable: str,
    vmin: float | None,
    vmax: float | None,
):
    plot_gdf = gdf.copy()

    if variable not in plot_gdf.columns:
        raise ValueError(f"{variable} not found in complete file.")

    plot_gdf[variable] = pd.to_numeric(plot_gdf[variable], errors="coerce")

    n_total = len(plot_gdf)
    n_nonmissing = int(plot_gdf[variable].notna().sum())
    n_missing = int(plot_gdf[variable].isna().sum())

    fig, ax = plt.subplots(figsize=(10.5, 8.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f8fa")

    edgecolor = "white" if DRAW_PIXEL_BORDERS else "none"
    linewidth = 0.04 if DRAW_PIXEL_BORDERS else 0.0

    if n_nonmissing == 0:
        plot_gdf.plot(
            ax=ax,
            facecolor="#d9dee5",
            edgecolor=edgecolor,
            linewidth=linewidth,
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
                "label": "NA",
            },
        )

    if DRAW_TOWNSHIP_BOUNDARIES and townships is not None:
        townships.boundary.plot(
            ax=ax,
            color="#263238",
            linewidth=0.18,
            alpha=0.45,
        )

    if DRAW_IOWA_OUTLINE and iowa_outline is not None:
        iowa_outline.boundary.plot(
            ax=ax,
            color="#111111",
            linewidth=1.1,
            alpha=0.95,
        )

    ax.set_axis_off()
    ax.set_aspect("equal")

    title = pretty_name(variable)
    subtitle = f"{date_string} | {PASS_TO_PLOT.upper()} | complete lattice over Iowa townships"
    note = f"non-missing: {n_nonmissing:,} | NA: {n_missing:,} | total SMAP lattice cells: {n_total:,}"

    ax.set_title(title, fontsize=18, fontweight="bold", pad=18, color="#263238")

    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="center",
        fontsize=11,
        color="#455a64",
    )

    fig.text(
        0.5,
        0.895,
        note,
        ha="center",
        va="center",
        fontsize=10,
        color="#607d8b",
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
            shrink=0.75,
        )
        cbar.set_label(variable, fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.88])

    return fig


# ============================================================
# 9. MAIN
# ============================================================

def main():
    print("\nVisualizing completed SMAP + IEM files on Iowa boundaries")
    print("-" * 80)
    print(f"Start date: {START_DATE}")
    print(f"N days:     {N_DAYS}")
    print(f"Pass:       {PASS_TO_PLOT}")
    print(f"Full dir:   {FULL_DIR}")
    print(f"Figure dir: {FIG_DIR}")
    print("-" * 80)

    records = load_records()

    if not records:
        raise RuntimeError("No complete files loaded.")

    first_gdf = next(iter(records.values()))
    townships, iowa_outline = load_townships(first_gdf.crs)

    variables = choose_variables(first_gdf)

    print("\nVariables to plot:")
    for variable in variables:
        print(f"  - {variable}")

    limits = compute_limits(records, variables)

    for date_string, gdf in records.items():
        yyyymmdd = date_to_yyyymmdd(date_string)

        for variable in variables:
            vmin, vmax = limits[variable]

            fig = plot_map(
                gdf=gdf,
                townships=townships,
                iowa_outline=iowa_outline,
                date_string=date_string,
                variable=variable,
                vmin=vmin,
                vmax=vmax,
            )

            var_dir = FIG_DIR / variable
            var_dir.mkdir(parents=True, exist_ok=True)

            stem = f"{variable}_{PASS_TO_PLOT}_{yyyymmdd}_iowa_boundaries"

            if SAVE_PDF:
                out_pdf = var_dir / f"{stem}.pdf"
                fig.savefig(out_pdf, bbox_inches="tight")
                print(f"[saved] {out_pdf}")

            if SAVE_PNG:
                out_png = var_dir / f"{stem}.png"
                fig.savefig(out_png, bbox_inches="tight", dpi=PNG_DPI)
                print(f"[saved] {out_png}")

            plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()