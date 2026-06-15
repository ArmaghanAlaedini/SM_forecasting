from pathlib import Path
import importlib.util
import os

# Help avoid common PROJ path issues.
try:
    from pyproj import datadir
    os.environ["PROJ_DATA"] = datadir.get_data_dir()
except Exception:
    pass

import geopandas as gpd
import matplotlib.pyplot as plt


# ============================================================
# 0. Load config
# ============================================================

def load_config():
    """Load 00_config.py even though the filename starts with a number."""
    config_path = Path(__file__).resolve().with_name("00_config.py")

    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# 1. Paths
# ============================================================

LATTICE_PATH = cfg.SMAP_LATTICE_DIR / "smap_lattice_iowa.parquet"
TOWNSHIP_SHP_PATH = cfg.TOWNSHIP_SHP_PATH

FIG_DIR = cfg.SMAP_LATTICE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================

def read_townships(path: Path) -> gpd.GeoDataFrame:
    """
    Read township shapefile.

    Fiona is used first because it avoids some pyogrio/PROJ mismatch issues.
    """
    try:
        return gpd.read_file(path, engine="fiona")
    except Exception:
        return gpd.read_file(path)


def load_data() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load lattice and township layers.
    """
    if not LATTICE_PATH.exists():
        raise FileNotFoundError(f"Lattice file not found:\n{LATTICE_PATH}")

    if not TOWNSHIP_SHP_PATH.exists():
        raise FileNotFoundError(f"Township shapefile not found:\n{TOWNSHIP_SHP_PATH}")

    lattice = gpd.read_parquet(LATTICE_PATH)
    townships = read_townships(TOWNSHIP_SHP_PATH)

    if townships.crs != lattice.crs:
        townships = townships.to_crs(lattice.crs)

    return lattice, townships


# ============================================================
# 3. Summaries
# ============================================================

def print_summary(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    """
    Print quick layer summaries.
    """
    print("\nSMAP lattice")
    print("-" * 60)
    print(f"CRS:        {lattice.crs}")
    print(f"Pixels:     {len(lattice)}")
    print(f"Bounds:     {lattice.total_bounds}")
    print(f"Columns:    {list(lattice.columns)}")

    print("\nIowa townships")
    print("-" * 60)
    print(f"CRS:        {townships.crs}")
    print(f"Townships:  {len(townships)}")
    print(f"Bounds:     {townships.total_bounds}")
    print(f"Columns:    {list(townships.columns)}")


# ============================================================
# 4. Plot helpers
# ============================================================

def clean_axis(ax, title: str) -> None:
    """
    Minimal map style.
    """
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_axis_off()
    ax.set_aspect("equal")


def set_iowa_extent(ax, townships: gpd.GeoDataFrame, pad_fraction: float = 0.04) -> None:
    """
    Set map extent using township bounds plus a small padding.
    """
    xmin, ymin, xmax, ymax = townships.total_bounds
    x_pad = (xmax - xmin) * pad_fraction
    y_pad = (ymax - ymin) * pad_fraction

    ax.set_xlim(xmin - x_pad, xmax + x_pad)
    ax.set_ylim(ymin - y_pad, ymax + y_pad)


def save_pdf(fig, filename: str) -> None:
    """
    Save figure as PDF only.
    """
    out_path = FIG_DIR / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# 5. Figure 1: side-by-side
# ============================================================

def plot_side_by_side(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    """
    Side-by-side comparison of SMAP lattice and Iowa townships.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    lattice.plot(
        ax=axes[0],
        facecolor="none",
        edgecolor="#1f3b73",
        linewidth=0.35,
    )
    set_iowa_extent(axes[0], townships)
    clean_axis(axes[0], "SMAP Lattice")

    townships.plot(
        ax=axes[1],
        facecolor="#f7f7f7",
        edgecolor="#4a4a4a",
        linewidth=0.25,
    )
    set_iowa_extent(axes[1], townships)
    clean_axis(axes[1], "Iowa Civil Townships")

    fig.suptitle("SMAP Lattice and Iowa Township Boundaries", fontsize=15, y=0.98)
    plt.tight_layout()

    save_pdf(fig, "01_lattice_townships_side_by_side.pdf")


# ============================================================
# 6. Figure 2: full overlay
# ============================================================

def plot_overlay(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    """
    Overlay SMAP lattice on Iowa township boundaries.
    """
    fig, ax = plt.subplots(figsize=(9.5, 8))

    townships.plot(
        ax=ax,
        facecolor="#f7f7f7",
        edgecolor="#5a5a5a",
        linewidth=0.25,
    )

    lattice.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#1f3b73",
        linewidth=0.45,
        alpha=0.9,
    )

    set_iowa_extent(ax, townships)
    clean_axis(ax, "SMAP Lattice Overlaid on Iowa Townships")
    plt.tight_layout()

    save_pdf(fig, "02_lattice_townships_overlay.pdf")


# ============================================================
# 7. Figure 3: zoomed overlay
# ============================================================

def plot_zoomed_overlay(lattice: gpd.GeoDataFrame, townships: gpd.GeoDataFrame) -> None:
    """
    Zoomed view to show mismatch between SMAP pixels and township boundaries.
    """
    xmin, ymin, xmax, ymax = townships.total_bounds

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    x_buffer = (xmax - xmin) * 0.16
    y_buffer = (ymax - ymin) * 0.16

    x0, x1 = x_center - x_buffer, x_center + x_buffer
    y0, y1 = y_center - y_buffer, y_center + y_buffer

    lattice_zoom = lattice.cx[x0:x1, y0:y1]
    townships_zoom = townships.cx[x0:x1, y0:y1]

    fig, ax = plt.subplots(figsize=(8, 8))

    townships_zoom.plot(
        ax=ax,
        facecolor="#f7f7f7",
        edgecolor="#4a4a4a",
        linewidth=0.45,
    )

    lattice_zoom.plot(
        ax=ax,
        facecolor="none",
        edgecolor="#1f3b73",
        linewidth=0.9,
        alpha=0.95,
    )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    clean_axis(ax, "Zoomed Overlay: SMAP Pixels and Townships")
    plt.tight_layout()

    save_pdf(fig, "03_lattice_townships_overlay_zoom.pdf")


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    lattice, townships = load_data()

    print_summary(lattice, townships)

    plot_side_by_side(lattice, townships)
    plot_overlay(lattice, townships)
    plot_zoomed_overlay(lattice, townships)

    print("\nDone. PDF figures saved only.")


if __name__ == "__main__":
    main()