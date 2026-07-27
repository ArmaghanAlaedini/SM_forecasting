#!/usr/bin/env python3
"""Create daily multi-panel PDFs for completed SMAP + IEM data.

This replaces the misleadingly named ``06_visualize_full_smap_iem_one_day.py``:
the script can visualize any consecutive date range and creates one PDF per
available date/pass.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from visualization_common import (
    VISUALIZATION_ROOT,
    auto_marker_size,
    complete_file_path,
    coord_columns,
    normalize_date,
    ordered_pta_columns,
    pretty_variable,
    read_spatial_csv,
    robust_limits,
    safe_name,
    variable_unit,
)


OUT_DIR = VISUALIZATION_ROOT / "03_complete_daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_N_DAYS = 10
DEFAULT_PASS = "am"
VARIABLES_PER_PAGE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--days", type=int, default=DEFAULT_N_DAYS)
    parser.add_argument("--pass-name", choices=["am", "pm"], default=DEFAULT_PASS)
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Default: SM plus all actual *_pta columns.",
    )
    return parser.parse_args()


def requested_dates(start_date: str, n_days: int) -> list[str]:
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
        raise FileNotFoundError("No requested completed SMAP + IEM files were found.")
    return records


def choose_variables(frame: pd.DataFrame, requested: list[str] | None) -> list[str]:
    if requested is not None:
        missing = [column for column in requested if column not in frame.columns]
        if missing:
            print(f"[warning] Requested variables not found and skipped: {missing}")
        return [column for column in requested if column in frame.columns]
    variables = ["soil_moisture"] if "soil_moisture" in frame.columns else []
    variables.extend(ordered_pta_columns(frame.columns))
    return variables


def draw_map(ax, frame: pd.DataFrame, geoframe, variable: str, vmin, vmax):
    values = pd.to_numeric(frame[variable], errors="coerce")
    if geoframe is not None:
        plot_frame = geoframe.copy()
        plot_frame[variable] = values.to_numpy()
        plot_frame.plot(
            column=variable,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            missing_kwds={"color": "0.84"},
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
    else:
        xcol, ycol = coord_columns(frame)
        x = pd.to_numeric(frame[xcol], errors="coerce")
        y = pd.to_numeric(frame[ycol], errors="coerce")
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
        ax.scatter(
            x[valid], y[valid], c=values[valid], cmap="viridis", vmin=vmin, vmax=vmax,
            s=auto_marker_size(len(frame)), marker="s", linewidths=0,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(alpha=0.15)

    missing = int(values.isna().sum())
    ax.set_title(f"{pretty_variable(variable)}\nmissing={missing:,}", fontsize=10)


def create_daily_pdf(
    date: str,
    pass_name: str,
    frame: pd.DataFrame,
    geoframe,
    variables: list[str],
    limits: dict[str, tuple[float | None, float | None]],
) -> None:
    output = OUT_DIR / f"complete_daily_{pass_name}_{date.replace('-', '')}.pdf"
    with PdfPages(output) as pdf:
        for start in range(0, len(variables), VARIABLES_PER_PAGE):
            page_variables = variables[start : start + VARIABLES_PER_PAGE]
            ncols = 3
            nrows = 2
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8.5), constrained_layout=True)
            axes_list = list(axes.flat)
            for ax, variable in zip(axes_list, page_variables):
                vmin, vmax = limits[variable]
                draw_map(ax, frame, geoframe, variable, vmin, vmax)
                if vmin is not None and vmax is not None:
                    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
                    cbar = fig.colorbar(mappable, ax=ax, shrink=0.74, pad=0.015)
                    unit = variable_unit(variable)
                    if unit:
                        cbar.set_label(unit)
            for ax in axes_list[len(page_variables):]:
                ax.set_visible(False)
            fig.suptitle(
                f"Completed SMAP + IEM data — {date} {pass_name.upper()}", fontsize=15
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"[saved] {output}")


def main() -> None:
    args = parse_args()
    dates = requested_dates(args.start_date, args.days)
    records = load_records(dates, args.pass_name)
    first_frame = next(iter(records.values()))[0]
    variables = choose_variables(first_frame, args.variables)
    if not variables:
        raise ValueError("No variables were selected for plotting.")

    limits = {
        variable: robust_limits(
            [frame[variable] for frame, _, _ in records.values() if variable in frame]
        )
        for variable in variables
    }

    print("06: Visualize completed daily SMAP + IEM data")
    print(f"Dates loaded: {len(records)}")
    print(f"Pass:         {args.pass_name.upper()}")
    print(f"Variables:    {len(variables)}")
    print(f"Output:       {OUT_DIR}")

    for date, (frame, geoframe, _) in records.items():
        available = [variable for variable in variables if variable in frame.columns]
        create_daily_pdf(date, args.pass_name, frame, geoframe, available, limits)
    print("Done.")


if __name__ == "__main__":
    main()
