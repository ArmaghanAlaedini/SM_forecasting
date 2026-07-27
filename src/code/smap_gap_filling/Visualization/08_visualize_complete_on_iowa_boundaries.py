#!/usr/bin/env python3
"""Plot completed SMAP + IEM fields with Iowa township boundaries.

The script uses the exact completed-file names produced by ``05_full_smap_iem.py``
and the township path from ``00_config.py``. Outputs are written to
``09_final_visualization/04_complete_iowa``.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

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
    complete_file_path,
    coord_columns,
    ordered_pta_columns,
    pretty_variable,
    read_spatial_csv,
    read_townships,
    robust_limits,
    safe_name,
    save_figure,
    variable_unit,
)


OUT_DIR = VISUALIZATION_ROOT / "04_complete_iowa"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2020-03-01")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--pass-name", choices=["am", "pm"], default="am")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["soil_moisture", "precip_pta", "soil12vwc_pta"],
    )
    parser.add_argument("--all-variables", action="store_true")
    return parser.parse_args()


def date_range(start_date: str, n_days: int) -> list[str]:
    if n_days <= 0:
        raise ValueError("--days must be positive.")
    start = pd.to_datetime(start_date).normalize()
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def load_records(dates: list[str], pass_name: str):
    records = {}
    for date in dates:
        path = complete_file_path(date, pass_name)
        if not path.exists():
            print(f"[missing] {path}")
            continue
        frame, geoframe = read_spatial_csv(path)
        records[date] = (frame, geoframe, path)
        print(f"[loaded] {date}: {path.name}")
    if not records:
        raise FileNotFoundError("No requested completed files were found.")
    return records


def plot_one(
    date: str,
    pass_name: str,
    frame: pd.DataFrame,
    geoframe,
    variable: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    values = pd.to_numeric(frame[variable], errors="coerce")

    if geoframe is not None:
        plot_frame = geoframe.copy()
        plot_frame[variable] = values.to_numpy()
        townships = read_townships(plot_frame.crs, required=True)
        assert townships is not None
        plot_frame.plot(
            column=variable,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            missing_kwds={"color": "0.84"},
        )
        townships.boundary.plot(ax=ax, color="0.28", linewidth=0.28, alpha=0.78)
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
        print(
            "[warning] geometry_wkt is absent, so township boundaries cannot be "
            f"overlaid for {date}."
        )

    if vmin is not None and vmax is not None:
        mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.72, pad=0.02)
        unit = variable_unit(variable)
        if unit:
            cbar.set_label(unit)

    missing = int(values.isna().sum())
    ax.set_title(
        f"{pretty_variable(variable)}\n{date} | {pass_name.upper()} | missing={missing:,}",
        fontsize=13,
    )
    fig.tight_layout()
    save_figure(
        fig,
        OUT_DIR
        / f"complete_iowa_{pass_name}_{date.replace('-', '')}_{safe_name(variable)}",
    )


def main() -> None:
    args = parse_args()
    records = load_records(date_range(args.start_date, args.days), args.pass_name)
    first = next(iter(records.values()))[0]
    variables = (
        (["soil_moisture"] if "soil_moisture" in first else [])
        + ordered_pta_columns(first.columns)
        if args.all_variables
        else args.variables
    )
    variables = [variable for variable in variables if any(variable in f for f, _, _ in records.values())]
    if not variables:
        raise ValueError("None of the requested variables are available.")

    limits = {
        variable: robust_limits(
            [frame[variable] for frame, _, _ in records.values() if variable in frame]
        )
        for variable in variables
    }

    print("08: Visualize completed fields on Iowa boundaries")
    print(f"Dates:     {list(records)}")
    print(f"Pass:      {args.pass_name.upper()}")
    print(f"Variables: {variables}")
    print(f"Output:    {OUT_DIR}")

    for date, (frame, geoframe, _) in records.items():
        for variable in variables:
            if variable in frame.columns:
                plot_one(
                    date,
                    args.pass_name,
                    frame,
                    geoframe,
                    variable,
                    *limits[variable],
                )
    print("Done.")


if __name__ == "__main__":
    main()
