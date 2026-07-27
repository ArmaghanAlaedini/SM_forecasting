#!/usr/bin/env python3
"""Visualize final gap-filled SMAP products from the corrected workflow.

The script reads exact file names:

* original: ``smap_iem_<pass>_complete_<YYYYMMDD>.csv``
* final: ``smap_iem_<pass>_gapfilled_<YYYYMMDD>.csv``

If ``--date`` is omitted, the date/pass with the most original missing pixels is
selected from ``gapfill_summary_by_file.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from visualization_common import (
    VISUALIZATION_ROOT,
    auto_marker_size,
    cfg,
    complete_file_path,
    coord_columns,
    gapfilled_file_path,
    model_family,
    pretty_method,
    read_spatial_csv,
    robust_limits,
    safe_name,
    save_figure,
)


OUT_DIR = VISUALIZATION_ROOT / "06_final_gapfill"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_PATH = cfg.FINAL_DIR / "gapfill_summary_by_file.csv"
OVERALL_SUMMARY_PATH = cfg.FINAL_DIR / "gapfill_overall_summary.csv"

METHOD_COLORS = {
    "observed": "#4C78A8",
    "stacking": "#59A14F",
    "nearest_neighbor_same_day": "#F28E2B",
    "centroid_ordinary_kriging": "#B07AA1",
    "regression_kriging": "#E15759",
    "xgboost": "#76B7B2",
    "hist_gbdt": "#EDC948",
    "random_forest": "#9C755F",
    "none": "#BAB0AC",
}
STATUS_COLORS = {"observed": "#4C78A8", "filled": "#59A14F", "unfilled": "#E15759"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="YYYY-MM-DD or YYYYMMDD; default: auto-pick")
    parser.add_argument("--pass-name", choices=["am", "pm"], default="am")
    parser.add_argument("--no-stations", action="store_true")
    return parser.parse_args()


def choose_date(pass_name: str, requested_date: str | None) -> str:
    if requested_date is not None:
        return pd.to_datetime(requested_date).strftime("%Y-%m-%d")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"No --date was supplied and the summary file is missing: {SUMMARY_PATH}"
        )
    summary = pd.read_csv(SUMMARY_PATH)
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
    summary["n_missing_original"] = pd.to_numeric(
        summary["n_missing_original"], errors="coerce"
    )
    candidates = summary[
        summary["pass"].astype(str).str.lower().eq(pass_name)
        & summary["date"].notna()
        & summary["n_missing_original"].gt(0)
    ].copy()
    if candidates.empty:
        raise RuntimeError(f"No gap-filled day with original gaps was found for pass={pass_name}.")
    row = candidates.sort_values(["n_missing_original", "date"], ascending=[False, True]).iloc[0]
    return row["date"].strftime("%Y-%m-%d")


def load_aligned_day(date: str, pass_name: str):
    original_path = complete_file_path(date, pass_name)
    final_path = gapfilled_file_path(date, pass_name)
    original, original_geo = read_spatial_csv(original_path)
    final, final_geo = read_spatial_csv(final_path)

    key = cfg.KEY
    for name, frame in [("original", original), ("final", final)]:
        if key not in frame.columns:
            raise ValueError(f"{name} file does not contain {key}: {date} {pass_name}")
        frame[key] = frame[key].astype(str)
        if frame[key].duplicated().any():
            raise ValueError(f"{name} file contains duplicate {key} values.")

    original_keys = set(original[key])
    final_keys = set(final[key])
    if original_keys != final_keys:
        raise ValueError(
            "Original and final files do not contain the same SMAP pixel keys. "
            f"Only original={len(original_keys-final_keys)}, only final={len(final_keys-original_keys)}"
        )

    final_order = final.set_index(key)
    original = original.set_index(key).loc[final_order.index].reset_index()
    final = final_order.reset_index()

    # Rebuild GeoDataFrames in aligned order when polygon geometry is available.
    if original_geo is not None:
        original_geo[key] = original_geo[key].astype(str)
        original_geo = original_geo.set_index(key).loc[final[key]].reset_index()
    if final_geo is not None:
        final_geo[key] = final_geo[key].astype(str)
        final_geo = final_geo.set_index(key).loc[final[key]].reset_index()

    required = [cfg.TARGET, "soil_moisture_filled", "fill_status", "fill_method", "stacking_eligible"]
    missing = [column for column in required if column not in final.columns and column != cfg.TARGET]
    if missing:
        raise ValueError(f"Final gap-filled file is missing required columns: {missing}")
    return original, final, original_geo, final_geo, original_path, final_path


def load_station_points(target_crs, disabled: bool) -> gpd.GeoDataFrame | None:
    if disabled:
        return None
    candidates = [
        Path(getattr(cfg, "IEM_STATIONS_FULL_PATH", "")),
        Path(getattr(cfg, "IEM_STATIONS_FULL_FALLBACK_PATH", "")),
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        print("[warning] Station overlay skipped: configured station file was not found.")
        return None
    stations = pd.read_csv(path, low_memory=False)
    stations = stations.loc[:, ~stations.columns.duplicated()].copy()
    lon_candidates = ["lon", "longitude", "station_lon"]
    lat_candidates = ["lat", "latitude", "station_lat"]
    lon = next((column for column in lon_candidates if column in stations), None)
    lat = next((column for column in lat_candidates if column in stations), None)
    if lon is None or lat is None:
        print(f"[warning] Station overlay skipped: no lon/lat columns in {path}")
        return None
    stations[lon] = pd.to_numeric(stations[lon], errors="coerce")
    stations[lat] = pd.to_numeric(stations[lat], errors="coerce")
    stations = stations.dropna(subset=[lon, lat]).drop_duplicates([lon, lat])
    points = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations[lon], stations[lat]),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        points = points.to_crs(target_crs)
    return points


def overlay_stations(ax, stations: gpd.GeoDataFrame | None) -> None:
    if stations is not None and not stations.empty:
        stations.plot(ax=ax, color="black", markersize=15, alpha=0.85, label="IEM stations")


def plot_continuous(
    ax,
    frame: pd.DataFrame,
    geoframe: gpd.GeoDataFrame | None,
    column: str,
    *,
    vmin: float,
    vmax: float,
    stations: gpd.GeoDataFrame | None,
):
    values = pd.to_numeric(frame[column], errors="coerce")
    if geoframe is not None:
        spatial = geoframe.copy()
        spatial[column] = values.to_numpy()
        spatial.plot(
            column=column,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            missing_kwds={"color": "0.82"},
        )
        overlay_stations(ax, stations)
        ax.set_axis_off()
        ax.set_aspect("equal")
    else:
        xcol, ycol = coord_columns(frame)
        x = pd.to_numeric(frame[xcol], errors="coerce")
        y = pd.to_numeric(frame[ycol], errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        ax.scatter(
            x[valid], y[valid], c=values[valid], cmap="viridis", vmin=vmin, vmax=vmax,
            marker="s", s=auto_marker_size(len(frame)), linewidths=0,
        )
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)


def plot_before_after(
    original: pd.DataFrame,
    final: pd.DataFrame,
    original_geo,
    final_geo,
    stations,
    date: str,
    pass_name: str,
) -> None:
    vmin, vmax = robust_limits(
        [original[cfg.TARGET], final["soil_moisture_filled"]]
    )
    if vmin is None or vmax is None:
        raise ValueError("No finite SM values were available for the before/after map.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.7), constrained_layout=True)
    plot_continuous(
        axes[0], original, original_geo, cfg.TARGET, vmin=vmin, vmax=vmax, stations=stations
    )
    plot_continuous(
        axes[1], final, final_geo, "soil_moisture_filled", vmin=vmin, vmax=vmax, stations=stations
    )
    axes[0].set_title("Original SMAP SM")
    axes[1].set_title("Final gap-filled SMAP SM")
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
    cbar = fig.colorbar(mappable, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label(r"SM (m$^3$ m$^{-3}$)")
    fig.suptitle(f"Before and after gap filling — {date} {pass_name.upper()}", fontsize=15)
    save_figure(fig, FIG_DIR / f"before_after_{pass_name}_{date.replace('-', '')}")


def plot_category(
    frame: pd.DataFrame,
    geoframe: gpd.GeoDataFrame | None,
    column: str,
    colors: dict[str, str],
    stations,
    title: str,
    filename: str,
) -> None:
    categories = frame[column].fillna("none").astype(str)
    fig, ax = plt.subplots(figsize=(9.5, 7.3))
    if geoframe is not None:
        spatial = geoframe.copy()
        spatial[column] = categories.to_numpy()
        for category in sorted(categories.unique()):
            mask = spatial[column].eq(category)
            spatial.loc[mask].plot(
                ax=ax,
                color=colors.get(category, "0.55"),
                linewidth=0,
                label=pretty_method(category),
            )
        overlay_stations(ax, stations)
        ax.set_axis_off()
        ax.set_aspect("equal")
    else:
        xcol, ycol = coord_columns(frame)
        x = pd.to_numeric(frame[xcol], errors="coerce")
        y = pd.to_numeric(frame[ycol], errors="coerce")
        for category in sorted(categories.unique()):
            mask = categories.eq(category) & np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[mask], y[mask], color=colors.get(category, "0.55"), marker="s",
                s=auto_marker_size(len(frame)), linewidths=0, label=pretty_method(category),
            )
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / filename)


def plot_stacking_eligibility(final, final_geo, stations, date, pass_name):
    original_missing = pd.to_numeric(final[cfg.TARGET], errors="coerce").isna()
    category = pd.Series("observed", index=final.index, dtype=object)
    category.loc[original_missing & final["stacking_eligible"].astype(bool)] = "stacking_eligible"
    category.loc[original_missing & ~final["stacking_eligible"].astype(bool)] = "fallback_required"
    frame = final.copy()
    frame["eligibility_class"] = category
    colors = {
        "observed": "#4C78A8",
        "stacking_eligible": "#59A14F",
        "fallback_required": "#F28E2B",
    }
    labels = {
        "observed": "Observed",
        "stacking_eligible": "Eligible for six-model stack",
        "fallback_required": "Required fallback",
    }

    # Use the general categorical mapper, but replace the internal labels with
    # the more descriptive stacking-eligibility labels.
    fig, ax = plt.subplots(figsize=(9.5, 7.3))
    if final_geo is not None:
        spatial = final_geo.copy()
        spatial["eligibility_class"] = category.to_numpy()
        for category_name in ["observed", "stacking_eligible", "fallback_required"]:
            mask = spatial["eligibility_class"].eq(category_name)
            spatial.loc[mask].plot(
                ax=ax, color=colors[category_name], linewidth=0, label=labels[category_name]
            )
        overlay_stations(ax, stations)
        ax.set_axis_off()
        ax.set_aspect("equal")
    else:
        xcol, ycol = coord_columns(frame)
        x = pd.to_numeric(frame[xcol], errors="coerce")
        y = pd.to_numeric(frame[ycol], errors="coerce")
        for category_name in ["observed", "stacking_eligible", "fallback_required"]:
            mask = category.eq(category_name) & np.isfinite(x) & np.isfinite(y)
            ax.scatter(
                x[mask], y[mask], c=colors[category_name], marker="s",
                s=auto_marker_size(len(frame)), linewidths=0, label=labels[category_name],
            )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Stacking eligibility — {date} {pass_name.upper()}")
    fig.tight_layout()
    save_figure(fig, FIG_DIR / f"stacking_eligibility_{pass_name}_{date.replace('-', '')}")


def plot_distribution(original, final, date, pass_name):
    observed = pd.to_numeric(original[cfg.TARGET], errors="coerce").dropna()
    filled = pd.to_numeric(final["soil_moisture_filled"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    ax.hist(observed, bins=45, alpha=0.62, label="Original observed")
    ax.hist(filled, bins=45, alpha=0.45, label="Final complete field")
    ax.set_xlabel(r"SM (m$^3$ m$^{-3}$)")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"SM distribution — {date} {pass_name.upper()}")
    ax.grid(axis="y", alpha=0.22)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, FIG_DIR / f"distribution_{pass_name}_{date.replace('-', '')}")


def plot_daily_summary(pass_name: str) -> None:
    if not SUMMARY_PATH.exists():
        return
    summary = pd.read_csv(SUMMARY_PATH)
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce")
    sub = summary[summary["pass"].astype(str).str.lower().eq(pass_name)].sort_values("date")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for column, label in [
        ("n_missing_original", "Original gaps"),
        ("n_stacking_eligible", "Stacking eligible"),
        ("n_filled", "Filled"),
        ("n_unfilled", "Unfilled"),
    ]:
        if column in sub:
            ax.plot(sub["date"], sub[column], label=label, linewidth=1.2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Pixel count")
    ax.set_title(f"Daily gap-filling coverage — {pass_name.upper()}")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    save_figure(fig, FIG_DIR / f"daily_gapfill_coverage_{pass_name}")


def write_summary(original, final, date, pass_name, original_path, final_path):
    lines = [
        "Final gap-fill visualization summary",
        "=" * 56,
        f"Date: {date}",
        f"Pass: {pass_name.upper()}",
        f"Original file: {original_path}",
        f"Final file: {final_path}",
        f"Project seed: {cfg.RANDOM_SEED}",
        "",
        f"Rows: {len(final):,}",
        f"Original observed: {pd.to_numeric(original[cfg.TARGET], errors='coerce').notna().sum():,}",
        f"Original missing: {pd.to_numeric(original[cfg.TARGET], errors='coerce').isna().sum():,}",
        f"Stacking eligible: {final['stacking_eligible'].astype(bool).sum():,}",
        "",
        "Fill status counts:",
    ]
    for key, value in final["fill_status"].value_counts(dropna=False).items():
        lines.append(f"  {key}: {value:,}")
    lines.append("")
    lines.append("Fill method counts:")
    for key, value in final["fill_method"].value_counts(dropna=False).items():
        lines.append(f"  {pretty_method(str(key))}: {value:,}")
    values = pd.to_numeric(final["soil_moisture_filled"], errors="coerce")
    lines.extend(
        [
            "",
            "soil_moisture_filled:",
            f"  min:  {values.min():.6f}",
            f"  mean: {values.mean():.6f}",
            f"  max:  {values.max():.6f}",
        ]
    )
    path = OUT_DIR / f"selected_day_summary_{pass_name}_{date.replace('-', '')}.txt"
    path.write_text("\n".join(lines))
    print(f"[saved] {path}")


def main() -> None:
    args = parse_args()
    date = choose_date(args.pass_name, args.date)
    (
        original,
        final,
        original_geo,
        final_geo,
        original_path,
        final_path,
    ) = load_aligned_day(date, args.pass_name)

    target_crs = final_geo.crs if final_geo is not None else None
    stations = load_station_points(target_crs, args.no_stations)

    print("12b: Visualize final gap-filled products")
    print(f"Date:          {date}")
    print(f"Pass:          {args.pass_name.upper()}")
    print(f"Original file: {original_path}")
    print(f"Final file:    {final_path}")
    print(f"Output:        {OUT_DIR}")

    write_summary(original, final, date, args.pass_name, original_path, final_path)
    plot_before_after(
        original, final, original_geo, final_geo, stations, date, args.pass_name
    )
    plot_category(
        final,
        final_geo,
        "fill_status",
        STATUS_COLORS,
        stations,
        f"Fill status — {date} {args.pass_name.upper()}",
        f"fill_status_{args.pass_name}_{date.replace('-', '')}",
    )
    plot_category(
        final,
        final_geo,
        "fill_method",
        METHOD_COLORS,
        stations,
        f"Fill method — {date} {args.pass_name.upper()}",
        f"fill_method_{args.pass_name}_{date.replace('-', '')}",
    )
    plot_stacking_eligibility(final, final_geo, stations, date, args.pass_name)
    plot_distribution(original, final, date, args.pass_name)
    plot_daily_summary(args.pass_name)
    print(f"\nSaved figures to: {FIG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
