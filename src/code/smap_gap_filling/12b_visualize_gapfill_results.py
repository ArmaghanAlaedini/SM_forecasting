#!/usr/bin/env python3
"""
12b_visualize_gapfill_results.py

Visualize final gap-filled SMAP outputs from the 11-family.

This script creates:
  - original vs final gap-filled map for one selected day/pass
  - original missingness map
  - fill-status map
  - fill-method map
  - histogram of original observed vs final filled values
  - daily gap-fill summary time series

Optional:
  - overlay IEM/station point-support locations as black dots

Outputs:
  src/data/processed/smap_gap_filling/09_final_visualization/gapfilling/

This script does not modify model outputs.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER CONTROLS
# ============================================================

# Change this day whenever you want.
# Format can be "YYYY-MM-DD" or "YYYYMMDD".
SELECTED_DATE = "2025-05-28"

# Use "am" or "pm".
PASS_NAME = "am"

# If True, ignores SELECTED_DATE and chooses the day/pass with the most original missing pixels.
AUTO_PICK_DAY_WITH_MOST_MISSING = False

# Figure appearance.
MARKER_SIZE = None  # None = automatic
MAKE_PDF_TOO = True


# ============================================================
# STATION POINT OVERLAY CONTROLS
# ============================================================

# Turn this on if you want IEM/station locations shown on maps.
SHOW_STATION_POINTS = True

# Use None to let the script search for a station CSV.
# Or give an exact path, for example:
# STATION_POINTS_PATH = "src/data/processed/isu_stations/your_station_file.csv"
STATION_POINTS_PATH = None

# If station data have lon/lat and the SMAP file also has lon/lat,
# this makes maps use lon/lat so the black station dots align correctly.
PREFER_LON_LAT_FOR_STATION_OVERLAY = True

STATION_POINT_SIZE = 18
STATION_POINT_ALPHA = 0.90
STATION_POINT_LABEL = "IEM stations"


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / "11_gapfilling_setting.py"

if SETTINGS_PATH.exists():
    spec = importlib.util.spec_from_file_location("gapfill_settings", SETTINGS_PATH)
    settings = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load settings file: {SETTINGS_PATH}")

    spec.loader.exec_module(settings)

    PROJECT_ROOT = settings.PROJECT_ROOT
    TARGET = settings.TARGET
    KEY = settings.KEY
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    TARGET = "soil_moisture"
    KEY = "smap_pixel_key"

BASE_DIR = PROJECT_ROOT / "src/data/processed/smap_gap_filling"

ORIGINAL_DIR = BASE_DIR / "03_full_smap_iem_data"
FINAL_DIR = BASE_DIR / "08_gapfilled_final"

SUMMARY_PATH = FINAL_DIR / "gapfill_summary_by_file.csv"
OVERALL_SUMMARY_PATH = FINAL_DIR / "gapfill_overall_summary.csv"

OUT_DIR = BASE_DIR / "09_final_visualization/gapfilling"
FIG_DIR = OUT_DIR / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_date_string(x: str) -> tuple[str, str]:
    x = str(x).strip()

    if re.fullmatch(r"\d{8}", x):
        ymd = x
        pretty = f"{x[:4]}-{x[4:6]}-{x[6:8]}"
        return ymd, pretty

    dt = pd.to_datetime(x)
    ymd = dt.strftime("%Y%m%d")
    pretty = dt.strftime("%Y-%m-%d")

    return ymd, pretty


def safe_name(x: str) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9._-]+", "_", x)
    return x.strip("_")[:120]


def save_figure(fig: plt.Figure, path_png: Path) -> None:
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=230, bbox_inches="tight")

    if MAKE_PDF_TOO:
        fig.savefig(path_png.with_suffix(".pdf"), bbox_inches="tight")

    plt.close(fig)


def find_one_file(folder: Path, ymd: str) -> Path:
    candidates = sorted(folder.glob(f"*{ymd}*.csv"))

    if not candidates:
        raise FileNotFoundError(f"No file found for {ymd} in {folder}")

    return candidates[0]


def get_coord_pair(
    df: pd.DataFrame,
    options: list[tuple[str, str]],
) -> tuple[str, str] | None:
    lower_to_real = {str(c).lower(): c for c in df.columns}

    for x_raw, y_raw in options:
        x = lower_to_real.get(x_raw.lower())
        y = lower_to_real.get(y_raw.lower())

        if x is not None and y is not None:
            return x, y

    return None


def get_coord_cols(df: pd.DataFrame) -> tuple[str, str]:
    lonlat_options = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("long", "lat"),
        ("lng", "lat"),
        ("lon_dd", "lat_dd"),
        ("longitude_dd", "latitude_dd"),
        ("station_lon", "station_lat"),
        ("station_longitude", "station_latitude"),
    ]

    xy_options = [
        ("x", "y"),
        ("X", "Y"),
        ("easting", "northing"),
        ("EASTING", "NORTHING"),
    ]

    if SHOW_STATION_POINTS and PREFER_LON_LAT_FOR_STATION_OVERLAY:
        lonlat = get_coord_pair(df, lonlat_options)
        if lonlat is not None:
            return lonlat

    xy = get_coord_pair(df, xy_options)
    if xy is not None:
        return xy

    lonlat = get_coord_pair(df, lonlat_options)
    if lonlat is not None:
        return lonlat

    raise ValueError("No coordinate columns found. Need x/y or lon/lat.")


def auto_marker_size(n: int) -> float:
    if MARKER_SIZE is not None:
        return float(MARKER_SIZE)

    if n <= 500:
        return 38
    if n <= 1500:
        return 26
    if n <= 4000:
        return 16
    if n <= 10000:
        return 9

    return 5


def load_selected_day() -> tuple[pd.DataFrame, pd.DataFrame, str, str, Path, Path]:
    pass_name = PASS_NAME.lower().strip()

    if pass_name not in {"am", "pm"}:
        raise ValueError("PASS_NAME must be either 'am' or 'pm'.")

    if AUTO_PICK_DAY_WITH_MOST_MISSING:
        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(f"Missing summary file: {SUMMARY_PATH}")

        summary = pd.read_csv(SUMMARY_PATH)
        summary["date"] = pd.to_datetime(summary["date"], errors="coerce")

        sub = summary[
            summary["pass"].astype(str).str.lower().eq(pass_name)
            & (summary["n_missing_original"] > 0)
        ].copy()

        if sub.empty:
            raise RuntimeError(f"No day with missing pixels found for pass={pass_name}")

        pick = sub.sort_values("n_missing_original", ascending=False).iloc[0]
        pretty_date = pick["date"].strftime("%Y-%m-%d")
        ymd = pick["date"].strftime("%Y%m%d")
    else:
        ymd, pretty_date = normalize_date_string(SELECTED_DATE)

    original_folder = ORIGINAL_DIR / pass_name / "complete"
    final_folder = FINAL_DIR / pass_name

    original_path = find_one_file(original_folder, ymd)
    final_path = find_one_file(final_folder, ymd)

    original = pd.read_csv(original_path, low_memory=False)
    final = pd.read_csv(final_path, low_memory=False)

    original = original.loc[:, ~original.columns.duplicated()].copy()
    final = final.loc[:, ~final.columns.duplicated()].copy()

    return original, final, ymd, pretty_date, original_path, final_path


def combined_limits(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    vals = pd.concat(
        [
            pd.to_numeric(a, errors="coerce"),
            pd.to_numeric(b, errors="coerce"),
        ],
        ignore_index=True,
    )

    vals = vals[np.isfinite(vals)]

    if vals.empty:
        return 0.0, 1.0

    lo = np.nanpercentile(vals, 2)
    hi = np.nanpercentile(vals, 98)

    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(vals.min())
        hi = float(vals.max())

    if lo == hi:
        hi = lo + 1e-6

    return float(lo), float(hi)


def prepare_axes(ax, xcol: str, ycol: str) -> None:
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)


# ============================================================
# STATION POINT HELPERS
# ============================================================

_STATION_CACHE: pd.DataFrame | None = None


def find_station_file() -> Path | None:
    if STATION_POINTS_PATH is not None:
        p = Path(STATION_POINTS_PATH).expanduser()

        if not p.is_absolute():
            p = PROJECT_ROOT / p

        if p.exists():
            return p

        raise FileNotFoundError(f"Requested station file does not exist: {p}")

    search_dirs = [
        PROJECT_ROOT / "src/data/processed/isu_stations",
        PROJECT_ROOT / "src/data/processed/iem_stations",
        PROJECT_ROOT / "src/data/processed/iem_point_to_area",
        PROJECT_ROOT / "src/data/processed/iem_stations",
        PROJECT_ROOT / "src/data/raw/isu_stations",
        PROJECT_ROOT / "src/data/raw/iem_stations",
        PROJECT_ROOT / "src/data/raw",
        PROJECT_ROOT / "src/data/processed",
    ]

    candidates: list[Path] = []

    for d in search_dirs:
        if d.exists():
            candidates.extend(sorted(d.rglob("*station*.csv")))
            candidates.extend(sorted(d.rglob("*stations*.csv")))
            candidates.extend(sorted(d.rglob("*iem*.csv")))

    candidates = [p for p in candidates if p.is_file()]

    if not candidates:
        return None

    preferred = [
        p for p in candidates
        if any(
            k in p.name.lower()
            for k in ["station", "stations", "location", "metadata", "site"]
        )
    ]

    return preferred[0] if preferred else candidates[0]


def load_station_points() -> pd.DataFrame | None:
    global _STATION_CACHE

    if not SHOW_STATION_POINTS:
        return None

    if _STATION_CACHE is not None:
        return _STATION_CACHE

    station_path = find_station_file()

    if station_path is None:
        print(
            "\nWarning: SHOW_STATION_POINTS=True, but no station CSV was found.\n"
            "Set STATION_POINTS_PATH manually in 12b_visualize_gapfill_results.py.\n"
        )
        _STATION_CACHE = pd.DataFrame()
        return _STATION_CACHE

    print(f"\nReading station points:\n  {station_path}")

    stations = pd.read_csv(station_path, low_memory=False)
    stations = stations.loc[:, ~stations.columns.duplicated()].copy()

    _STATION_CACHE = stations
    return stations


def station_xy_for_map(
    stations: pd.DataFrame | None,
    map_df: pd.DataFrame,
    xcol: str,
    ycol: str,
) -> tuple[pd.Series | None, pd.Series | None]:
    if stations is None or stations.empty:
        return None, None

    # Case 1: station file has exactly the same coordinate columns used by the map.
    if xcol in stations.columns and ycol in stations.columns:
        sx = pd.to_numeric(stations[xcol], errors="coerce")
        sy = pd.to_numeric(stations[ycol], errors="coerce")
        return sx, sy

    # Case 2: map uses lon/lat and station file has common lon/lat column names.
    lonlat_options = [
        ("lon", "lat"),
        ("longitude", "latitude"),
        ("long", "lat"),
        ("lng", "lat"),
        ("lon_dd", "lat_dd"),
        ("longitude_dd", "latitude_dd"),
        ("station_lon", "station_lat"),
        ("station_longitude", "station_latitude"),
    ]

    station_lonlat = get_coord_pair(stations, lonlat_options)

    if station_lonlat is not None:
        sx_col, sy_col = station_lonlat

        map_x_lower = str(xcol).lower()
        map_y_lower = str(ycol).lower()

        map_is_lonlat = (
            map_x_lower in {"lon", "longitude", "long", "lng", "lon_dd", "longitude_dd"}
            and map_y_lower in {"lat", "latitude", "lat_dd", "latitude_dd"}
        )

        if map_is_lonlat:
            sx = pd.to_numeric(stations[sx_col], errors="coerce")
            sy = pd.to_numeric(stations[sy_col], errors="coerce")
            return sx, sy

    print(
        "\nWarning: Station coordinates could not be aligned with map coordinates.\n"
        f"Map uses: {xcol}, {ycol}\n"
        "Try setting PREFER_LON_LAT_FOR_STATION_OVERLAY=True, or set STATION_POINTS_PATH "
        "to a station file with matching x/y coordinates.\n"
    )

    return None, None


def overlay_station_points(
    ax,
    stations: pd.DataFrame | None,
    map_df: pd.DataFrame,
    xcol: str,
    ycol: str,
) -> bool:
    if not SHOW_STATION_POINTS:
        return False

    if stations is None or stations.empty:
        return False

    sx, sy = station_xy_for_map(stations, map_df, xcol, ycol)

    if sx is None or sy is None:
        return False

    sx = pd.to_numeric(sx, errors="coerce")
    sy = pd.to_numeric(sy, errors="coerce")

    good = np.isfinite(sx) & np.isfinite(sy)

    if not good.any():
        print("Warning: station file was found, but no finite station coordinates were available.")
        return False

    mx = pd.to_numeric(map_df[xcol], errors="coerce")
    my = pd.to_numeric(map_df[ycol], errors="coerce")

    if not np.isfinite(mx).any() or not np.isfinite(my).any():
        print("Warning: map coordinate columns are not finite, so station overlay was skipped.")
        return False

    xmin, xmax = np.nanmin(mx), np.nanmax(mx)
    ymin, ymax = np.nanmin(my), np.nanmax(my)

    xpad = 0.05 * max(xmax - xmin, 1e-9)
    ypad = 0.05 * max(ymax - ymin, 1e-9)

    in_extent = (
        good
        & (sx >= xmin - xpad)
        & (sx <= xmax + xpad)
        & (sy >= ymin - ypad)
        & (sy <= ymax + ypad)
    )

    if not in_extent.any():
        print("Warning: station points loaded, but none fall inside the map extent.")
        return False

    ax.scatter(
        sx[in_extent],
        sy[in_extent],
        s=STATION_POINT_SIZE,
        c="black",
        marker="o",
        alpha=STATION_POINT_ALPHA,
        linewidths=0.25,
        edgecolors="white",
        label=STATION_POINT_LABEL,
        zorder=20,
    )

    return True


# ============================================================
# MAP PLOTS
# ============================================================

def plot_value_map(
    ax,
    df: pd.DataFrame,
    value_col: str,
    xcol: str,
    ycol: str,
    title: str,
    vmin: float,
    vmax: float,
):
    size = auto_marker_size(len(df))

    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    val = pd.to_numeric(df[value_col], errors="coerce")

    # Draw all pixels lightly first so gaps are visually obvious.
    ax.scatter(
        x,
        y,
        s=size,
        marker="s",
        c="lightgray",
        alpha=0.45,
        linewidths=0,
    )

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)

    sc = ax.scatter(
        x[mask],
        y[mask],
        c=val[mask],
        s=size,
        marker="s",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )

    ax.set_title(title)
    prepare_axes(ax, xcol, ycol)

    return sc


def plot_before_after(
    original: pd.DataFrame,
    final: pd.DataFrame,
    pretty_date: str,
    pass_name: str,
) -> None:
    xcol, ycol = get_coord_cols(final)

    original_value = TARGET
    final_value = "soil_moisture_filled"

    if original_value not in original.columns:
        raise ValueError(f"Original file does not contain {original_value}")

    if final_value not in final.columns:
        raise ValueError(f"Final file does not contain {final_value}")

    vmin, vmax = combined_limits(original[original_value], final[final_value])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)

    _ = plot_value_map(
        axes[0],
        original,
        original_value,
        xcol,
        ycol,
        f"Original SMAP with gaps\n{pretty_date} | {pass_name.upper()}",
        vmin,
        vmax,
    )

    sc1 = plot_value_map(
        axes[1],
        final,
        final_value,
        xcol,
        ycol,
        f"Final gap-filled SMAP\n{pretty_date} | {pass_name.upper()}",
        vmin,
        vmax,
    )

    stations = load_station_points()
    added0 = overlay_station_points(axes[0], stations, original, xcol, ycol)
    added1 = overlay_station_points(axes[1], stations, final, xcol, ycol)

    if added0:
        axes[0].legend(loc="best", frameon=True)
    if added1:
        axes[1].legend(loc="best", frameon=True)

    cbar = fig.colorbar(sc1, ax=axes, shrink=0.82)
    cbar.set_label("Soil moisture")

    out = FIG_DIR / f"before_after_gapfill_{pass_name}_{safe_name(pretty_date)}.png"
    save_figure(fig, out)


def plot_original_missingness(
    original: pd.DataFrame,
    pretty_date: str,
    pass_name: str,
) -> None:
    xcol, ycol = get_coord_cols(original)
    size = auto_marker_size(len(original))

    x = pd.to_numeric(original[xcol], errors="coerce")
    y = pd.to_numeric(original[ycol], errors="coerce")
    observed = pd.to_numeric(original[TARGET], errors="coerce").notna()

    fig, ax = plt.subplots(figsize=(8.5, 7))

    ax.scatter(
        x[observed],
        y[observed],
        s=size,
        marker="s",
        c="#2b8cbe",
        alpha=0.78,
        linewidths=0,
        label="Observed",
    )

    ax.scatter(
        x[~observed],
        y[~observed],
        s=size,
        marker="s",
        c="#f03b20",
        alpha=0.78,
        linewidths=0,
        label="Original missing",
    )

    stations = load_station_points()
    overlay_station_points(ax, stations, original, xcol, ycol)

    ax.set_title(f"Original SMAP missingness\n{pretty_date} | {pass_name.upper()}")
    prepare_axes(ax, xcol, ycol)
    ax.legend(loc="best", frameon=True)

    out = FIG_DIR / f"original_missingness_{pass_name}_{safe_name(pretty_date)}.png"
    save_figure(fig, out)


def plot_fill_status(
    final: pd.DataFrame,
    pretty_date: str,
    pass_name: str,
) -> None:
    if "fill_status" not in final.columns:
        raise ValueError("Final file does not contain fill_status")

    xcol, ycol = get_coord_cols(final)
    size = auto_marker_size(len(final))

    x = pd.to_numeric(final[xcol], errors="coerce")
    y = pd.to_numeric(final[ycol], errors="coerce")

    colors = {
        "observed": "#2b8cbe",
        "filled": "#31a354",
        "unfilled": "#f03b20",
    }

    fig, ax = plt.subplots(figsize=(8.5, 7))

    for status, color in colors.items():
        mask = final["fill_status"].astype(str).eq(status)

        if mask.any():
            ax.scatter(
                x[mask],
                y[mask],
                s=size,
                marker="s",
                c=color,
                alpha=0.78,
                linewidths=0,
                label=status,
            )

    stations = load_station_points()
    overlay_station_points(ax, stations, final, xcol, ycol)

    ax.set_title(f"Fill status\n{pretty_date} | {pass_name.upper()}")
    prepare_axes(ax, xcol, ycol)
    ax.legend(loc="best", frameon=True)

    out = FIG_DIR / f"fill_status_{pass_name}_{safe_name(pretty_date)}.png"
    save_figure(fig, out)


def plot_fill_method(
    final: pd.DataFrame,
    pretty_date: str,
    pass_name: str,
) -> None:
    if "fill_method" not in final.columns:
        raise ValueError("Final file does not contain fill_method")

    xcol, ycol = get_coord_cols(final)
    size = auto_marker_size(len(final))

    x = pd.to_numeric(final[xcol], errors="coerce")
    y = pd.to_numeric(final[ycol], errors="coerce")

    methods = sorted(final["fill_method"].astype(str).fillna("unknown").unique())

    palette = [
        "#2b8cbe",
        "#31a354",
        "#756bb1",
        "#e6550d",
        "#636363",
        "#969696",
        "#de2d26",
        "#3182bd",
    ]

    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    for i, method in enumerate(methods):
        mask = final["fill_method"].astype(str).eq(method)

        if not mask.any():
            continue

        ax.scatter(
            x[mask],
            y[mask],
            s=size,
            marker="s",
            c=palette[i % len(palette)],
            alpha=0.78,
            linewidths=0,
            label=method,
        )

    stations = load_station_points()
    overlay_station_points(ax, stations, final, xcol, ycol)

    ax.set_title(f"Fill method\n{pretty_date} | {pass_name.upper()}")
    prepare_axes(ax, xcol, ycol)
    ax.legend(loc="best", frameon=True, fontsize=8)

    out = FIG_DIR / f"fill_method_{pass_name}_{safe_name(pretty_date)}.png"
    save_figure(fig, out)


# ============================================================
# HISTOGRAM AND SUMMARY PLOTS
# ============================================================

def plot_histogram(
    original: pd.DataFrame,
    final: pd.DataFrame,
    pretty_date: str,
    pass_name: str,
) -> None:
    final_value = "soil_moisture_filled"

    if TARGET not in original.columns:
        raise ValueError(f"Original file does not contain {TARGET}")

    if final_value not in final.columns:
        raise ValueError(f"Final file does not contain {final_value}")

    orig_vals = pd.to_numeric(original[TARGET], errors="coerce").dropna()
    final_vals = pd.to_numeric(final[final_value], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.hist(orig_vals, bins=45, alpha=0.62, label="Original observed")
    ax.hist(final_vals, bins=45, alpha=0.45, label="Final filled")

    ax.set_xlabel("Soil moisture")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Distribution before and after gap filling\n{pretty_date} | {pass_name.upper()}")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()

    out = FIG_DIR / f"hist_original_vs_filled_{pass_name}_{safe_name(pretty_date)}.png"
    save_figure(fig, out)


def plot_daily_summary(pass_name: str) -> None:
    if not SUMMARY_PATH.exists():
        print(f"Skipping daily summary plot. Missing: {SUMMARY_PATH}")
        return

    summary = pd.read_csv(SUMMARY_PATH)
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")

    sub = summary[summary["pass"].astype(str).str.lower().eq(pass_name)].copy()

    if sub.empty:
        print(f"Skipping daily summary plot. No rows for pass={pass_name}.")
        return

    sub = sub.sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 5.8))

    ax.plot(sub["date"], sub["n_observed_original"], label="Original observed", linewidth=1.1)
    ax.plot(sub["date"], sub["n_missing_original"], label="Original missing", linewidth=1.1)
    ax.plot(sub["date"], sub["n_filled"], label="Filled", linewidth=1.1)
    ax.plot(sub["date"], sub["n_unfilled"], label="Still unfilled", linewidth=1.1)

    ax.set_xlabel("Date")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Daily gap-filling summary | {pass_name.upper()}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)

    out = FIG_DIR / f"daily_gapfill_summary_{pass_name}.png"
    save_figure(fig, out)


def write_selected_day_summary(
    original: pd.DataFrame,
    final: pd.DataFrame,
    pretty_date: str,
    pass_name: str,
    original_path: Path,
    final_path: Path,
) -> None:
    lines = []

    lines.append("Selected gap-fill visualization summary")
    lines.append("=" * 60)
    lines.append(f"Date: {pretty_date}")
    lines.append(f"Pass: {pass_name}")
    lines.append(f"Original file: {original_path}")
    lines.append(f"Final file: {final_path}")
    lines.append(f"Station overlay enabled: {SHOW_STATION_POINTS}")
    lines.append("")

    n_rows = len(final)
    n_original_observed = int(pd.to_numeric(original[TARGET], errors="coerce").notna().sum())
    n_original_missing = int(pd.to_numeric(original[TARGET], errors="coerce").isna().sum())

    lines.append(f"Rows: {n_rows:,}")
    lines.append(f"Original observed: {n_original_observed:,}")
    lines.append(f"Original missing: {n_original_missing:,}")

    if "fill_status" in final.columns:
        lines.append("")
        lines.append("Fill status counts:")
        for k, v in final["fill_status"].value_counts(dropna=False).items():
            lines.append(f"  {k}: {v:,}")

    if "fill_method" in final.columns:
        lines.append("")
        lines.append("Fill method counts:")
        for k, v in final["fill_method"].value_counts(dropna=False).items():
            lines.append(f"  {k}: {v:,}")

    if "soil_moisture_filled" in final.columns:
        vals = pd.to_numeric(final["soil_moisture_filled"], errors="coerce")
        lines.append("")
        lines.append("soil_moisture_filled summary:")
        lines.append(f"  min:  {vals.min():.6f}")
        lines.append(f"  mean: {vals.mean():.6f}")
        lines.append(f"  max:  {vals.max():.6f}")

    out = OUT_DIR / f"selected_day_summary_{pass_name}_{safe_name(pretty_date)}.txt"
    out.write_text("\n".join(lines))

    print(f"Saved selected day summary:\n  {out}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("12b: Visualize final gap-filled results")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Original dir:  {ORIGINAL_DIR}")
    print(f"Final dir:     {FINAL_DIR}")
    print(f"Output dir:    {OUT_DIR}")
    print(f"Selected date: {SELECTED_DATE}")
    print(f"Pass:          {PASS_NAME}")
    print(f"Auto pick day: {AUTO_PICK_DAY_WITH_MOST_MISSING}")
    print(f"Show stations: {SHOW_STATION_POINTS}")
    print("=" * 80)

    original, final, ymd, pretty_date, original_path, final_path = load_selected_day()
    pass_name = PASS_NAME.lower().strip()

    print("\nSelected files:")
    print(f"  Original: {original_path}")
    print(f"  Final:    {final_path}")

    write_selected_day_summary(
        original=original,
        final=final,
        pretty_date=pretty_date,
        pass_name=pass_name,
        original_path=original_path,
        final_path=final_path,
    )

    plot_before_after(original, final, pretty_date, pass_name)
    plot_original_missingness(original, pretty_date, pass_name)
    plot_fill_status(final, pretty_date, pass_name)
    plot_fill_method(final, pretty_date, pass_name)
    plot_histogram(original, final, pretty_date, pass_name)
    plot_daily_summary(pass_name)

    print("\nSaved figures to:")
    print(f"  {FIG_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()