

from pathlib import Path
import importlib.util
import os

# Help avoid common PROJ path issues.
try:
    from pyproj import datadir
    os.environ["PROJ_DATA"] = datadir.get_data_dir()
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box


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
# 1. SMAP NC4 variable candidates
# ============================================================

AM_VARS = {
    "sm": [
        "Soil_Moisture_Retrieval_Data_AM/soil_moisture",
        "Soil_Moisture_Retrieval_Data_AM/soil_moisture_dca",
        "Soil_Moisture_Retrieval_Data_AM/soil_moisture_scah",
        "Soil_Moisture_Retrieval_Data_AM/soil_moisture_scav",
    ],
    "lon": [
        "Soil_Moisture_Retrieval_Data_AM/longitude",
        "Soil_Moisture_Retrieval_Data_AM/longitude_centroid",
    ],
    "lat": [
        "Soil_Moisture_Retrieval_Data_AM/latitude",
        "Soil_Moisture_Retrieval_Data_AM/latitude_centroid",
    ],
}

PM_VARS = {
    "sm": [
        "Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm",
        "Soil_Moisture_Retrieval_Data_PM/soil_moisture_dca_pm",
        "Soil_Moisture_Retrieval_Data_PM/soil_moisture_scah_pm",
        "Soil_Moisture_Retrieval_Data_PM/soil_moisture_scav_pm",
        "Soil_Moisture_Retrieval_Data_PM/soil_moisture",
    ],
    "lon": [
        "Soil_Moisture_Retrieval_Data_PM/longitude_pm",
        "Soil_Moisture_Retrieval_Data_PM/longitude_centroid_pm",
        "Soil_Moisture_Retrieval_Data_PM/longitude",
    ],
    "lat": [
        "Soil_Moisture_Retrieval_Data_PM/latitude_pm",
        "Soil_Moisture_Retrieval_Data_PM/latitude_centroid_pm",
        "Soil_Moisture_Retrieval_Data_PM/latitude",
    ],
}


# ============================================================
# 2. NetCDF helpers
# ============================================================

def list_nc_files() -> list[Path]:
    """List raw SMAP NC4 files."""
    raw_dir = cfg.RAW_SMAP_NC_DIR

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw SMAP NC4 directory does not exist:\n{raw_dir}")

    patterns = ["*.nc4", "*.NC4", "*.nc", "*.NC"]
    files: list[Path] = []

    for pattern in patterns:
        files.extend(raw_dir.glob(pattern))

    files = sorted(set(files))

    if cfg.LATTICE_SCAN_FILES is not None:
        files = files[:cfg.LATTICE_SCAN_FILES]

    if not files:
        raise FileNotFoundError(f"No NC4 files found in:\n{raw_dir}")

    return files


def collect_variable_paths(group, prefix: str = "") -> list[str]:
    """Recursively list all variables inside an NC4 file."""
    paths = []

    for name in group.variables:
        full_name = f"{prefix}/{name}" if prefix else name
        paths.append(full_name)

    for group_name, subgroup in group.groups.items():
        subgroup_prefix = f"{prefix}/{group_name}" if prefix else group_name
        paths.extend(collect_variable_paths(subgroup, subgroup_prefix))

    return paths


def get_variable(group, path: str):
    """Get NetCDF variable by full path."""
    parts = path.split("/")
    current = group

    for part in parts[:-1]:
        current = current.groups[part]

    return current.variables[parts[-1]]


def find_first_available(available: list[str], candidates: list[str]) -> str | None:
    """Find first candidate variable that exists."""
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def to_float_array(values) -> np.ndarray:
    """Convert NetCDF/masked array to float numpy array."""
    if np.ma.isMaskedArray(values):
        values = np.ma.filled(values, np.nan)

    return np.squeeze(np.asarray(values, dtype=float))


def valid_coord_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Valid coordinate mask."""
    return (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= -180)
        & (lon <= 180)
        & (lat >= -90)
        & (lat <= 90)
    )


def iowa_bbox_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Mask coordinates inside Iowa bounding box."""
    west, south, east, north = cfg.IOWA_BBOX

    return (
        (lon >= west)
        & (lon <= east)
        & (lat >= south)
        & (lat <= north)
    )


def read_retrieval_from_nc(nc_file: Path, pass_name: str) -> pd.DataFrame | None:
    """
    Read one AM or PM retrieval from one NC4 file.

    Returns only pixels with valid soil moisture and valid coordinates
    inside the Iowa bounding box.
    """
    try:
        import netCDF4
    except ImportError as exc:
        raise ImportError("netCDF4 is required. Install with: pip install netCDF4") from exc

    var_set = AM_VARS if pass_name == "am" else PM_VARS

    try:
        with netCDF4.Dataset(nc_file) as ds:
            available = collect_variable_paths(ds)

            sm_var = find_first_available(available, var_set["sm"])
            lon_var = find_first_available(available, var_set["lon"])
            lat_var = find_first_available(available, var_set["lat"])

            if sm_var is None or lon_var is None or lat_var is None:
                return None

            sm = to_float_array(get_variable(ds, sm_var)[:])
            lon = to_float_array(get_variable(ds, lon_var)[:])
            lat = to_float_array(get_variable(ds, lat_var)[:])

            if sm.shape != lon.shape or sm.shape != lat.shape:
                if sm.T.shape == lon.shape and sm.T.shape == lat.shape:
                    sm = sm.T
                elif lon.T.shape == sm.shape and lat.T.shape == sm.shape:
                    lon = lon.T
                    lat = lat.T
                else:
                    return None

            rows, cols = np.indices(sm.shape)

            mask = (
                np.isfinite(sm)
                & valid_coord_mask(lon, lat)
                & iowa_bbox_mask(lon, lat)
            )

            if mask.sum() == 0:
                return None

            df = pd.DataFrame({
                "grid_row": rows[mask].ravel(),
                "grid_col": cols[mask].ravel(),
                "soil_moisture": sm[mask].ravel(),
                "lon": lon[mask].ravel(),
                "lat": lat[mask].ravel(),
                "pass": pass_name,
                "source_file": nc_file.name,
                "sm_variable": sm_var,
                "lon_variable": lon_var,
                "lat_variable": lat_var,
            })

            return df

    except Exception as exc:
        print(f"Failed reading {nc_file.name} {pass_name.upper()}: {exc}")
        return None


# ============================================================
# 3. Select max-coverage retrieval
# ============================================================

def find_max_coverage_retrieval() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scan first N NC4 files and choose the date-pass retrieval with
    the largest number of valid SMAP pixels over Iowa.
    """
    files = list_nc_files()
    summary_rows = []
    best_df = None
    best_count = -1

    print(f"Raw NC4 files scanned: {len(files)}")
    print("Searching for retrieval with maximum valid Iowa pixels...")

    for file_idx, nc_file in enumerate(files, start=1):
        for pass_name in cfg.PASSES:
            df = read_retrieval_from_nc(nc_file, pass_name)

            n_pixels = 0 if df is None else len(df)

            summary_rows.append({
                "file": nc_file.name,
                "pass": pass_name,
                "valid_iowa_pixels": n_pixels,
            })

            print(
                f"[{file_idx}/{len(files)}] {nc_file.name} "
                f"{pass_name.upper()} | valid pixels: {n_pixels}"
            )

            if df is not None and n_pixels > best_count:
                best_df = df
                best_count = n_pixels

    summary = pd.DataFrame(summary_rows)

    if best_df is None or best_count <= 0:
        raise RuntimeError("No usable SMAP retrieval found in scanned NC4 files.")

    print("\nSelected max-coverage retrieval")
    print("-" * 60)
    print(f"Source file: {best_df['source_file'].iloc[0]}")
    print(f"Pass:        {best_df['pass'].iloc[0].upper()}")
    print(f"Pixels:      {best_count}")
    print("-" * 60)

    return best_df, summary


# ============================================================
# 4. Build lattice polygons
# ============================================================

def build_lattice(selected_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Build SMAP square polygons from selected retrieval centroid coordinates.
    """
    centroids = gpd.GeoDataFrame(
        selected_df.copy(),
        geometry=gpd.points_from_xy(selected_df["lon"], selected_df["lat"]),
        crs=cfg.CRS_WGS84,
    ).to_crs(cfg.CRS_EASE)

    centroids["x"] = centroids.geometry.x
    centroids["y"] = centroids.geometry.y

    centroids["smap_pixel_key"] = (
        centroids["x"].round(0).astype("int64").astype(str)
        + "_"
        + centroids["y"].round(0).astype("int64").astype(str)
    )

    half = cfg.SMAP_CELLSIZE_M / 2.0

    polygons = [
        box(x - half, y - half, x + half, y + half)
        for x, y in zip(centroids["x"], centroids["y"])
    ]

    lattice = gpd.GeoDataFrame(
        centroids.drop(columns="geometry"),
        geometry=polygons,
        crs=cfg.CRS_EASE,
    )

    lattice["cell_area_m2"] = lattice.geometry.area

    lattice = lattice.drop_duplicates(subset=["smap_pixel_key"]).copy()

        # Repair one known interior missing SMAP lattice pixel.
    missing_key = "-9264785_4823814"

    if missing_key not in set(lattice["smap_pixel_key"].astype(str)):
        half = cfg.SMAP_CELLSIZE_M / 2.0

        x_missing = -9264784.560657388
        y_missing = 4823813.511327039

        missing_row = {
            "smap_pixel_key": missing_key,
            "grid_row": 29,
            "grid_col": 10,
            "lon": -96.02178192138672,
            "lat": 41.19828414916992,
            "x": x_missing,
            "y": y_missing,
            "cell_area_m2": cfg.SMAP_CELLSIZE_M ** 2,
            "pass": lattice["pass"].iloc[0],
            "source_file": lattice["source_file"].iloc[0],
            "sm_variable": lattice["sm_variable"].iloc[0],
            "lon_variable": lattice["lon_variable"].iloc[0],
            "lat_variable": lattice["lat_variable"].iloc[0],
            "geometry": box(
                x_missing - half,
                y_missing - half,
                x_missing + half,
                y_missing + half,
            ),
        }

        lattice = pd.concat(
            [lattice, gpd.GeoDataFrame([missing_row], geometry="geometry", crs=cfg.CRS_EASE)],
            ignore_index=True,
        )

    keep_cols = [
        "smap_pixel_key",
        "grid_row",
        "grid_col",
        "lon",
        "lat",
        "x",
        "y",
        "cell_area_m2",
        "pass",
        "source_file",
        "sm_variable",
        "lon_variable",
        "lat_variable",
        "geometry",
    ]

    lattice = lattice[keep_cols]
    lattice = lattice.sort_values(["y", "x"]).reset_index(drop=True)

    return lattice


# ============================================================
# 5. Save outputs
# ============================================================

def save_outputs(lattice: gpd.GeoDataFrame, summary: pd.DataFrame) -> None:
    """Save lattice and scan summary."""
    out_dir = cfg.SMAP_LATTICE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "smap_lattice_iowa.parquet"
    csv_path = out_dir / "smap_lattice_iowa.csv"
    rds_path = out_dir / "smap_lattice_iowa_wkt.rds"
    summary_path = out_dir / "smap_lattice_scan_summary.csv"

    lattice.to_parquet(parquet_path, index=False)

    csv_df = lattice.copy()
    csv_df["geometry_wkt"] = csv_df.geometry.to_wkt()
    csv_df = pd.DataFrame(csv_df.drop(columns="geometry"))
    csv_df.to_csv(csv_path, index=False)

    try:
        import pyreadr
        pyreadr.write_rds(str(rds_path), csv_df)
        wrote_rds = True
    except Exception as exc:
        print(f"RDS not written: {exc}")
        wrote_rds = False

    summary.to_csv(summary_path, index=False)

    print("\nSaved outputs")
    print("-" * 60)
    print(f"GeoParquet: {parquet_path}")
    print(f"CSV/WKT:    {csv_path}")
    if wrote_rds:
        print(f"RDS/WKT:    {rds_path}")
    print(f"Summary:    {summary_path}")
    print(f"Pixels:     {len(lattice)}")
    print("-" * 60)


# ============================================================
# 6. Main
# ============================================================

def main() -> None:
    cfg.print_config_summary()

    selected_df, summary = find_max_coverage_retrieval()
    lattice = build_lattice(selected_df)

    print("\nFinal lattice summary")
    print("-" * 60)
    print(f"Pixels:       {len(lattice)}")
    print(f"CRS:          EPSG:{cfg.CRS_EASE}")
    print(f"Lon range:    {lattice['lon'].min():.4f} to {lattice['lon'].max():.4f}")
    print(f"Lat range:    {lattice['lat'].min():.4f} to {lattice['lat'].max():.4f}")
    print(f"Source file:  {lattice['source_file'].iloc[0]}")
    print(f"Source pass:  {lattice['pass'].iloc[0].upper()}")
    print("-" * 60)

    save_outputs(lattice, summary)

    print("\nDone.")


if __name__ == "__main__":
    main()