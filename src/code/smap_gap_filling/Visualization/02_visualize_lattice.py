#!/usr/bin/env python3
"""Visualize the Iowa SMAP lattice and civil-township boundaries.

Inputs
------
* ``support/smap_lattice/smap_lattice_iowa.parquet``
* the township shapefile configured by ``TOWNSHIP_SHP_PATH``

Outputs
-------
``09_final_visualization/01_lattice/``
"""

from __future__ import annotations

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization_common import VISUALIZATION_ROOT, cfg, read_townships, save_figure


OUT_DIR = VISUALIZATION_ROOT / "01_lattice"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LATTICE_PATH = cfg.SMAP_LATTICE_DIR / "smap_lattice_iowa.parquet"


def load_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not LATTICE_PATH.exists():
        raise FileNotFoundError(f"SMAP lattice not found: {LATTICE_PATH}")
    lattice = gpd.read_parquet(LATTICE_PATH)
    if lattice.crs is None:
        lattice = lattice.set_crs(f"EPSG:{cfg.CRS_EASE}")
    townships = read_townships(lattice.crs, required=True)
    assert townships is not None
    return lattice, townships


def set_extent(ax, layer: gpd.GeoDataFrame, pad: float = 0.035) -> None:
    xmin, ymin, xmax, ymax = layer.total_bounds
    dx = max(xmax - xmin, 1.0) * pad
    dy = max(ymax - ymin, 1.0) * pad
    ax.set_xlim(xmin - dx, xmax + dx)
    ax.set_ylim(ymin - dy, ymax + dy)
    ax.set_aspect("equal")
    ax.set_axis_off()


def plot_side_by_side(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    lattice.plot(ax=axes[0], facecolor="none", edgecolor="0.20", linewidth=0.45)
    townships.boundary.plot(ax=axes[1], color="0.25", linewidth=0.30)
    set_extent(axes[0], townships)
    set_extent(axes[1], townships)
    axes[0].set_title(f"SMAP lattice ({len(lattice):,} pixels)")
    axes[1].set_title(f"Iowa civil townships ({len(townships):,})")
    fig.suptitle("Spatial supports used in the SMAP gap-filling workflow", fontsize=15)
    save_figure(fig, OUT_DIR / "lattice_and_townships_side_by_side")


def plot_overlay(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.8))
    townships.plot(ax=ax, facecolor="0.96", edgecolor="0.50", linewidth=0.25)
    lattice.boundary.plot(ax=ax, color="0.10", linewidth=0.50)
    set_extent(ax, townships)
    ax.set_title("SMAP lattice over Iowa civil townships")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "lattice_over_townships")


def plot_zoom(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    xmin, ymin, xmax, ymax = townships.total_bounds
    xc, yc = (xmin + xmax) / 2, (ymin + ymax) / 2
    width = (xmax - xmin) * 0.32
    height = (ymax - ymin) * 0.32
    x0, x1 = xc - width / 2, xc + width / 2
    y0, y1 = yc - height / 2, yc + height / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    townships.cx[x0:x1, y0:y1].plot(
        ax=ax, facecolor="0.97", edgecolor="0.40", linewidth=0.45
    )
    lattice.cx[x0:x1, y0:y1].boundary.plot(ax=ax, color="0.10", linewidth=0.90)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Zoomed comparison of pixel and township boundaries")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / "lattice_township_zoom")


def main() -> None:
    lattice, townships = load_layers()
    print("02: Visualize SMAP lattice")
    print(f"Lattice:   {LATTICE_PATH}")
    print(f"Pixels:    {len(lattice):,}")
    print(f"Townships: {len(townships):,}")
    print(f"Output:    {OUT_DIR}")
    plot_side_by_side(lattice, townships)
    plot_overlay(lattice, townships)
    plot_zoom(lattice, townships)
    print("Done.")


if __name__ == "__main__":
    main()
