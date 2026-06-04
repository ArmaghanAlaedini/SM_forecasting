from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


# ============================================================
# Settings
# ============================================================

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

MODEL = "hrrr"

# HRRR is approximately 3 km.
# This creates 3 km x 3 km square cells around each centroid.
CELL_SIZE_M = 3000
HALF_SIZE_M = CELL_SIZE_M / 2

PROJECTED_CRS = "EPSG:5070"   # USA Contiguous Albers, meters
OUTPUT_CRS = "EPSG:4326"

PROJECT_ROOT = Path.cwd()

YEARLY_DIR = PROJECT_ROOT / "src" / "data" / "processed" / "archived_weather" / "yearly"

TOWNSHIP_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "raw"
    / "townships"
    / "civil_townships_a_ia.shp"
)


# ============================================================
# Helpers
# ============================================================

def make_square_around_point(point, half_size_m: float):
    """Create a square polygon around one projected point."""
    x = point.x
    y = point.y

    return box(
        x - half_size_m,
        y - half_size_m,
        x + half_size_m,
        y + half_size_m,
    )


def load_iowa_boundary() -> gpd.GeoDataFrame:
    """Load township shapefile and dissolve to one Iowa boundary."""
    townships = gpd.read_file(TOWNSHIP_FILE)

    if townships.crs is None:
        raise ValueError("Township CRS is missing. Please set it before using this script.")

    townships = townships.to_crs(PROJECTED_CRS)

    iowa = gpd.GeoDataFrame(
        geometry=[townships.union_all()],
        crs=PROJECTED_CRS,
    )

    return iowa


def rebuild_hrrr_geometry_for_year(year: int, iowa_boundary_projected: gpd.GeoDataFrame) -> None:
    """Rebuild approximate HRRR grid polygons for one year from pixel parquet."""
    pixel_path = YEARLY_DIR / f"{MODEL}_weather_iowa_pixels_{year}.parquet"
    out_path = YEARLY_DIR / f"weather_model_grids_iowa_{year}.geoparquet"
    backup_path = YEARLY_DIR / f"weather_model_grids_iowa_{year}_old_tiny_polygons.geoparquet"

    if not pixel_path.exists():
        print(f"Skipping {year}; missing pixel file:")
        print(pixel_path)
        return

    print("=" * 70)
    print(f"Rebuilding HRRR geometry for {year}")
    print(f"Reading: {pixel_path}")

    cols = [
        "model",
        "product",
        "model_resolution_note",
        "grid_id",
        "latitude",
        "longitude",
    ]

    df = pd.read_parquet(pixel_path, columns=cols)

    df = (
        df[df["model"].astype(str) == MODEL]
        .drop_duplicates(subset=["model", "product", "grid_id"])
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )

    print(f"Unique HRRR grid points: {len(df):,}")

    # Start with centroid points in lon/lat.
    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=OUTPUT_CRS,
    )

    # Project to meters so a 3 km square actually means 3 km.
    points_m = points.to_crs(PROJECTED_CRS)

    # Build square cells in meters.
    points_m["geometry"] = [
        make_square_around_point(pt, HALF_SIZE_M)
        for pt in points_m.geometry
    ]

    points_m["grid_polygon_method"] = (
        "approx_3km_square_from_hrrr_centroid_projected_epsg5070"
    )
    points_m["grid_polygon_crs"] = OUTPUT_CRS

    # Clip to Iowa boundary for your Iowa-specific output.
    # This makes edge cells follow the Iowa border.
    cells_m = gpd.clip(points_m, iowa_boundary_projected)

    # Back to lon/lat for easier plotting and consistency with your previous files.
    cells = cells_m.to_crs(OUTPUT_CRS)

    cells = cells[
        [
            "model",
            "product",
            "model_resolution_note",
            "grid_id",
            "latitude",
            "longitude",
            "grid_polygon_method",
            "grid_polygon_crs",
            "geometry",
        ]
    ].copy()

    # Backup old geometry file before overwriting.
    if out_path.exists() and not backup_path.exists():
        out_path.rename(backup_path)
        print(f"Backed up old geometry file to:")
        print(backup_path)

    cells.to_parquet(out_path, index=False)

    # Quick area check.
    area_km2 = cells.to_crs(PROJECTED_CRS).geometry.area / 1_000_000

    print(f"Saved rebuilt geometry:")
    print(out_path)
    print("Area km² summary:")
    print(area_km2.describe())


# ============================================================
# Main
# ============================================================

def main():
    iowa_boundary = load_iowa_boundary()

    for year in YEARS:
        rebuild_hrrr_geometry_for_year(year, iowa_boundary)

    print("=" * 70)
    print("Done rebuilding HRRR geometry.")


if __name__ == "__main__":
    main()