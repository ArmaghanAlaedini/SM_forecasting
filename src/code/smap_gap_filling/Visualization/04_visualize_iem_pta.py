#!/usr/bin/env python3
"""Visualize daily IEM point-to-area outputs on the SMAP lattice.

The script reads the exact daily file names created by ``03_iem_pta_kriging.py``
and writes figures to ``09_final_visualization/02_iem_pta``.
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
import pandas as pd

from visualization_common import (
    VISUALIZATION_ROOT,
    iem_pta_file_path,
    normalize_date,
    ordered_pta_columns,
    pretty_variable,
    read_spatial_csv,
    read_townships,
    robust_limits,
    safe_name,
    save_figure,
    variable_unit,
)


DEFAULT_DATES = ["2020-01-01", "2020-07-01", "2025-07-01"]
DEFAULT_VARIABLES = ["precip_pta", "et_pta", "soil12vwc_pta", "soil24vwc_pta"]
OUT_DIR = VISUALIZATION_ROOT / "02_iem_pta"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    parser.add_argument("--all-variables", action="store_true")
    parser.add_argument("--no-townships", action="store_true")
    return parser.parse_args()


def load_days(dates: list[str]) -> dict[str, gpd.GeoDataFrame]:
    records: dict[str, gpd.GeoDataFrame] = {}
    for raw_date in dates:
        date = normalize_date(raw_date)
        path = iem_pta_file_path(date)
        if not path.exists():
            print(f"[warning] PTA file not found for {date}: {path}")
            continue
        _, gdf = read_spatial_csv(path)
        if gdf is None:
            raise ValueError(f"PTA file must contain geometry_wkt: {path}")
        records[date] = gdf
        print(f"[loaded] {date}: {path.name}")
    if not records:
        raise FileNotFoundError("None of the requested IEM PTA daily files were found.")
    return records


def add_boundary(ax, townships: gpd.GeoDataFrame | None) -> None:
    if townships is not None:
        townships.boundary.plot(ax=ax, color="0.35", linewidth=0.22, alpha=0.75)


def plot_variables_for_date(
    date: str,
    gdf: gpd.GeoDataFrame,
    variables: list[str],
    limits: dict[str, tuple[float | None, float | None]],
    townships: gpd.GeoDataFrame | None,
) -> None:
    variables = [variable for variable in variables if variable in gdf.columns]
    if not variables:
        return
    ncols = min(3, len(variables))
    nrows = (len(variables) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows), constrained_layout=True
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, variable in zip(axes_list, variables):
        vmin, vmax = limits[variable]
        gdf.plot(
            column=variable,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            missing_kwds={"color": "0.82"},
        )
        add_boundary(ax, townships)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(pretty_variable(variable), fontsize=11)
        if vmin is not None and vmax is not None:
            mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
            label = variable_unit(variable)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.015)
            if label:
                cbar.set_label(label)

    for ax in axes_list[len(variables):]:
        ax.set_visible(False)

    fig.suptitle(f"IEM variables translated to SMAP support — {date}", fontsize=14)
    save_figure(fig, OUT_DIR / f"iem_pta_variables_{date.replace('-', '')}")


def plot_variable_across_dates(
    variable: str,
    records: dict[str, gpd.GeoDataFrame],
    limits: tuple[float | None, float | None],
    townships: gpd.GeoDataFrame | None,
) -> None:
    available = [(date, gdf) for date, gdf in records.items() if variable in gdf.columns]
    if not available:
        return
    fig, axes = plt.subplots(
        1, len(available), figsize=(4.8 * len(available), 4.9), constrained_layout=True
    )
    axes_list = list(axes) if hasattr(axes, "__len__") else [axes]
    vmin, vmax = limits
    for ax, (date, gdf) in zip(axes_list, available):
        gdf.plot(
            column=variable,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            missing_kwds={"color": "0.82"},
        )
        add_boundary(ax, townships)
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(date)

    if vmin is not None and vmax is not None:
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
        cbar = fig.colorbar(mappable, ax=axes_list, shrink=0.72, pad=0.015)
        unit = variable_unit(variable)
        if unit:
            cbar.set_label(unit)
    fig.suptitle(pretty_variable(variable), fontsize=14)
    save_figure(fig, OUT_DIR / f"iem_pta_{safe_name(variable)}_across_dates")


def main() -> None:
    args = parse_args()
    records = load_days(args.dates)
    first = next(iter(records.values()))
    variables = ordered_pta_columns(first.columns) if args.all_variables else args.variables
    variables = [variable for variable in variables if any(variable in gdf for gdf in records.values())]
    if not variables:
        raise ValueError("None of the requested variables are present in the PTA files.")

    limits = {
        variable: robust_limits(
            [gdf[variable] for gdf in records.values() if variable in gdf.columns]
        )
        for variable in variables
    }
    townships = None if args.no_townships else read_townships(first.crs)

    print("04: Visualize IEM PTA outputs")
    print(f"Dates:     {list(records)}")
    print(f"Variables: {variables}")
    print(f"Output:    {OUT_DIR}")

    for date, gdf in records.items():
        plot_variables_for_date(date, gdf, variables, limits, townships)
    for variable in variables:
        plot_variable_across_dates(variable, records, limits[variable], townships)
    print("Done.")


if __name__ == "__main__":
    main()
