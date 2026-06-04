from __future__ import annotations

import gc
import importlib.util
import os
import shutil
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from herbie import Herbie
# from shapely.geometry import Polygon
from shapely.geometry import box


# ============================================================
# 1. Load config
# ============================================================

CONFIG_PATH = Path(__file__).with_name("00_config.py")

spec = importlib.util.spec_from_file_location("weather_config", CONFIG_PATH)
cfg = importlib.util.module_from_spec(spec)

if spec.loader is None:
    raise RuntimeError(f"Could not load config file: {CONFIG_PATH}")

spec.loader.exec_module(cfg)

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# 2. Basic helpers
# ============================================================

def lon_to_360(lon: float) -> float:
    """Convert longitude from -180/180 style to 0/360 style."""
    return lon % 360


def lon_to_180_series(lon: pd.Series) -> pd.Series:
    """Convert longitude series from 0/360 style to -180/180 style."""
    return ((lon + 180) % 360) - 180


def output_suffix() -> str:
    """Return output file suffix based on config."""
    if cfg.OUTPUT_FORMAT == "parquet":
        return ".parquet"
    if cfg.OUTPUT_FORMAT == "csv":
        return ".csv"
    raise ValueError(f"Unsupported output format: {cfg.OUTPUT_FORMAT}")


def save_table(df: pd.DataFrame, path: Path) -> None:
    """Save a regular pandas table."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file suffix: {path.suffix}")


def save_geoparquet(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Save a GeoDataFrame as GeoParquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(path, index=False)


def all_years_from_range() -> list[int]:
    """Return all years covered by START_DATE to END_DATE."""
    start_year = pd.Timestamp(cfg.START_DATE).year
    end_year = pd.Timestamp(cfg.END_DATE).year
    return list(range(start_year, end_year + 1))


def year_list() -> list[int]:
    """
    Return selected years.

    cfg.YEARS_TO_PROCESS = None means all years.
    """
    all_years = all_years_from_range()

    if cfg.YEARS_TO_PROCESS is None:
        return all_years

    selected = [int(y) for y in cfg.YEARS_TO_PROCESS]
    valid = [y for y in selected if y in all_years]

    if not valid:
        raise ValueError(
            f"YEARS_TO_PROCESS={cfg.YEARS_TO_PROCESS} does not overlap with "
            f"{cfg.START_DATE} to {cfg.END_DATE}."
        )

    return valid


def model_items_to_process():
    """
    Return selected model configs.

    cfg.MODELS_TO_PROCESS = None means all models.
    cfg.MODELS_TO_PROCESS = ["hrrr"] means selected models.
    """
    if cfg.MODELS_TO_PROCESS is None:
        return cfg.MODEL_CONFIGS.items()

    if isinstance(cfg.MODELS_TO_PROCESS, str):
        selected = {cfg.MODELS_TO_PROCESS}
    else:
        selected = set(cfg.MODELS_TO_PROCESS)

    available = set(cfg.MODEL_CONFIGS.keys())
    missing = selected - available

    if missing:
        raise ValueError(
            f"Unknown model(s): {missing}. Available models: {sorted(available)}"
        )

    return [
        (model_key, model_cfg)
        for model_key, model_cfg in cfg.MODEL_CONFIGS.items()
        if model_key in selected
    ]


def get_dates_for_year(year: int) -> pd.DatetimeIndex:
    """
    Return dates for one year, clipped to START_DATE and END_DATE.
    """
    start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(cfg.START_DATE))
    end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(cfg.END_DATE))

    dates = pd.date_range(start, end, freq=cfg.DATE_FREQ)

    if cfg.TEST_N_DAYS is not None:
        all_dates = pd.date_range(cfg.START_DATE, cfg.END_DATE, freq=cfg.DATE_FREQ)
        allowed_dates = set(all_dates[: cfg.TEST_N_DAYS])
        dates = pd.DatetimeIndex([d for d in dates if d in allowed_dates])

    return dates


def make_valid_times_for_model(
    dates: pd.DatetimeIndex,
    model_cfg: dict[str, Any],
) -> list[pd.Timestamp]:
    """
    Make valid times for a model.

    Example:
        date = 2020-01-01
        valid_hours_utc = [0]
        valid_time = 2020-01-01 00:00

    RAP may use [3], giving valid_time = 03:00 UTC.
    """
    valid_times = []

    for date in dates:
        for hour in model_cfg["valid_hours_utc"]:
            valid_times.append(date + pd.Timedelta(hours=hour))

    return valid_times


def get_max_workers() -> int:
    """
    Number of worker processes.

    Priority:
    1. Environment variable WEATHER_MAX_WORKERS
    2. cfg.MAX_WORKERS
    3. default 1

    Keep this 1 on your laptop. On HPC try 2, then 4.
    """
    env_value = os.environ.get("WEATHER_MAX_WORKERS")
    if env_value is not None:
        return max(1, int(env_value))

    return max(1, int(getattr(cfg, "MAX_WORKERS", 1)))


def get_max_pending_tasks(max_workers: int) -> int:
    """
    Number of submitted tasks allowed to sit in queue at once.

    Keeping this small prevents many completed DataFrames from waiting
    in memory before the parent process writes them.
    """
    default_value = max(2, max_workers * 2)
    return max(1, int(getattr(cfg, "MAX_PENDING_TASKS", default_value)))


def get_max_tasks_per_child() -> int | None:
    """
    Restart each worker after this many tasks.

    This helps with memory growth from cfgrib/xarray/Herbie.
    """
    value = getattr(cfg, "MAX_TASKS_PER_CHILD", 8)

    if value is None:
        return None

    value = int(value)
    if value <= 0:
        return None

    return value


# ============================================================
# 3. Product candidates
# ============================================================

def get_product_candidates(model_cfg: dict[str, Any]) -> list[str]:
    """Return product candidates for a model."""
    candidates = model_cfg["product_candidates"]

    if not candidates:
        raise ValueError("No product candidates were provided.")

    return list(candidates)


# ============================================================
# 4. Spatial subsetting
# ============================================================

def dataset_uses_360_longitude(ds) -> bool:
    """Return True if dataset longitudes appear to be 0 to 360."""
    return float(ds["longitude"].max()) > 180.0


def subset_to_bbox(ds):
    """
    Keep only grid points inside Iowa bounding box.

    Herbie/xarray latitude and longitude are model grid-point coordinates.
    Here they are treated as grid-cell centers / representative points.
    """
    if "latitude" not in ds and "latitude" not in ds.coords:
        raise ValueError("Dataset has no latitude coordinate.")

    if "longitude" not in ds and "longitude" not in ds.coords:
        raise ValueError("Dataset has no longitude coordinate.")

    if dataset_uses_360_longitude(ds):
        west = lon_to_360(cfg.BBOX["west"])
        east = lon_to_360(cfg.BBOX["east"])
    else:
        west = cfg.BBOX["west"]
        east = cfg.BBOX["east"]

    bbox_mask = (
        (ds["latitude"] >= cfg.BBOX["south"])
        & (ds["latitude"] <= cfg.BBOX["north"])
        & (ds["longitude"] >= west)
        & (ds["longitude"] <= east)
    )

    return ds.where(bbox_mask, drop=True)


# ============================================================
# 5. Xarray to DataFrame
# ============================================================

def get_numeric_data_vars(ds) -> list[str]:
    """Return numeric xarray data variables."""
    data_vars = []

    for var_name in ds.data_vars:
        var = ds[var_name]
        if np.issubdtype(var.dtype, np.number):
            data_vars.append(var_name)

    return data_vars


def one_dataset_to_table(ds) -> pd.DataFrame:
    """
    Convert one xarray Dataset into a pandas DataFrame.
    """
    ds = ds.squeeze(drop=True)

    data_vars = get_numeric_data_vars(ds)

    if not data_vars:
        raise ValueError("Dataset has no numeric weather variables.")

    df = ds[data_vars].to_dataframe().reset_index()
    df = df.dropna(subset=data_vars, how="all").copy()

    keep_cols = []

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            keep_cols.append(col)

    keep_cols += data_vars
    df = df[keep_cols].copy()

    # Avoid duplicate grid-coordinate rows from scalar GRIB coordinates.
    # This assumes requested variables are all comparable at the same grid point.
    df = df.groupby(["latitude", "longitude"], as_index=False).first()

    return df


def merge_dataset_tables(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple xarray/cfgrib tables returned from the same model file.

    cfgrib sometimes returns multiple hypercubes, so Herbie may return
    a list of xarray Datasets instead of one Dataset.
    """
    if not tables:
        raise ValueError("No tables to merge.")

    base = tables[0]

    for next_table in tables[1:]:
        shared_keys = ["latitude", "longitude"]

        duplicate_cols = [
            col for col in next_table.columns
            if col in base.columns and col not in shared_keys
        ]

        next_table = next_table.drop(columns=duplicate_cols)
        base = base.merge(next_table, on=shared_keys, how="outer")

    return base


def ds_or_list_to_table(ds_or_list) -> pd.DataFrame:
    """
    Convert one xarray Dataset or a list of xarray Datasets into one table.
    """
    if isinstance(ds_or_list, list):
        tables = []

        for one_ds in ds_or_list:
            ds_subset = subset_to_bbox(one_ds)
            table = one_dataset_to_table(ds_subset)
            tables.append(table)

        df = merge_dataset_tables(tables)

    else:
        ds_subset = subset_to_bbox(ds_or_list)
        df = one_dataset_to_table(ds_subset)

    if df.empty:
        return df

    if df["longitude"].max() > 180:
        df["longitude"] = lon_to_180_series(df["longitude"])

    # Exact final filter in -180/180 coordinates.
    df = df[
        (df["latitude"] >= cfg.BBOX["south"])
        & (df["latitude"] <= cfg.BBOX["north"])
        & (df["longitude"] >= cfg.BBOX["west"])
        & (df["longitude"] <= cfg.BBOX["east"])
    ].copy()

    return df


# ============================================================
# 6. Standardized variables
# ============================================================

def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first candidate column that exists in df."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def k_to_c(x: pd.Series) -> pd.Series:
    """Kelvin to Celsius."""
    return x - 273.15


def c_to_f(x: pd.Series) -> pd.Series:
    """Celsius to Fahrenheit."""
    return (x * 9.0 / 5.0) + 32.0


def mm_to_in(x: pd.Series) -> pd.Series:
    """Millimeters to inches."""
    return x / 25.4


def mps_to_mph(x: pd.Series) -> pd.Series:
    """Meters per second to miles per hour."""
    return x * 2.2369362920544


def add_standard_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert model-specific raw GRIB columns into cleaned station-like variables.

    Raw columns are dropped later by keep_pixel_columns().
    """

    # Instantaneous / snapshot 2-m temperature.
    temp_col = first_existing(df, ["t2m", "tmp", "t"])
    if temp_col is not None:
        df["temperature_c"] = k_to_c(df[temp_col])
        df["temperature_f"] = c_to_f(df["temperature_c"])

    # Optional max/min temperature fields, if available.
    high_col = first_existing(df, ["tmax", "mx2t", "maxt", "max_t", "max_tmp"])
    if high_col is not None:
        df["high_temperature_c"] = k_to_c(df[high_col])
        df["high_temperature_f"] = c_to_f(df["high_temperature_c"])

    low_col = first_existing(df, ["tmin", "mn2t", "mint", "min_t", "min_tmp"])
    if low_col is not None:
        df["low_temperature_c"] = k_to_c(df[low_col])
        df["low_temperature_f"] = c_to_f(df["low_temperature_c"])

    # Dew point.
    dew_col = first_existing(df, ["d2m", "dpt"])
    if dew_col is not None:
        df["dewpoint_c"] = k_to_c(df[dew_col])
        df["dewpoint_f"] = c_to_f(df["dewpoint_c"])

    # Relative humidity.
    rh_col = first_existing(df, ["r2", "rh", "r"])
    if rh_col is not None:
        df["relative_humidity_percent"] = df[rh_col]

    # 10-m wind components and speed.
    u_col = first_existing(df, ["u10", "u"])
    v_col = first_existing(df, ["v10", "v"])

    if u_col is not None:
        df["wind_u_10m_mps"] = df[u_col]

    if v_col is not None:
        df["wind_v_10m_mps"] = df[v_col]

    if u_col is not None and v_col is not None:
        df["wind_speed_10m_mps"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
        df["wind_speed_10m_mph"] = mps_to_mph(df["wind_speed_10m_mps"])

    # Gust.
    gust_col = first_existing(df, ["gust"])
    if gust_col is not None:
        df["wind_gust_mps"] = df[gust_col]
        df["wind_gust_mph"] = mps_to_mph(df["wind_gust_mps"])

    # Precipitation rate.
    prate_col = first_existing(df, ["prate"])
    if prate_col is not None:
        df["precip_rate"] = df[prate_col]

    # Accumulated precipitation.
    apcp_col = first_existing(df, ["tp", "apcp"])
    if apcp_col is not None:
        # For liquid water, kg/m^2 is equivalent to mm.
        df["precip_accum_mm"] = df[apcp_col]
        df["precip_accum_in"] = mm_to_in(df["precip_accum_mm"])

    # Downward shortwave radiation.
    sw_col = first_existing(df, ["sdswrf", "dswrf"])
    if sw_col is not None:
        df["downward_shortwave_radiation_wm2"] = df[sw_col]

    # Latent heat flux; related to evapotranspiration but not identical to ET.
    lht_col = first_existing(df, ["slhtf", "lhtfl"])
    if lht_col is not None:
        df["latent_heat_flux_wm2"] = df[lht_col]

    # Pressure.
    pressure_col = first_existing(df, ["sp", "pres"])
    if pressure_col is not None:
        df["surface_pressure_hpa"] = df[pressure_col] / 100.0

    # Optional model land-surface moisture variables.
    if "mstav" in df.columns:
        df["moisture_availability_percent"] = df["mstav"]

    if "cnwat" in df.columns:
        df["canopy_water"] = df["cnwat"]

    # soil_water_model is a forecast-model land-surface variable.
    # It is not the same as SMAP volumetric soil moisture or IEM station VWC.
    soilw_col = first_existing(df, ["soilw", "sotw"])
    if soilw_col is not None:
        df["soil_water_model"] = df[soilw_col]

    soilt_col = first_existing(df, ["soilt", "tsoil", "st"])
    if soilt_col is not None:
        df["soil_temperature_c"] = k_to_c(df[soilt_col])
        df["soil_temperature_f"] = c_to_f(df["soil_temperature_c"])

    return df


# ============================================================
# 7. Grid IDs and output columns
# ============================================================

def clean_id_text(x: pd.Series) -> pd.Series:
    """Clean text so it is safe inside grid_id."""
    return (
        x.astype("string")
        .fillna("unknown")
        .str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
        .str.strip("_")
    )


def add_grid_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stable model/product-specific grid-point ID.

    This is the join key to the combined grid GeoParquet file.
    """
    model_part = clean_id_text(df["model"])
    product_part = clean_id_text(df["product"])

    df["grid_id"] = (
        model_part
        + "_"
        + product_part
        + "_"
        + df["latitude"].round(5).astype(str)
        + "_"
        + df["longitude"].round(5).astype(str)
    )

    return df


def keep_pixel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only cleaned final columns.

    Missing columns are added as NA so all yearly files have the same schema.
    """
    for col in cfg.PIXEL_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[cfg.PIXEL_OUTPUT_COLUMNS].copy()
    return df


def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make output schema stable and reduce memory.

    Strings:
        pandas string

    Integer ID columns:
        nullable Int16/Int64

    Latitude/longitude:
        float64 for stable coordinate joins and grid polygon building

    Weather values:
        float32 to reduce memory and file size
    """
    string_cols = [
        "valid_date",
        "valid_time",
        "init_time",
        "model",
        "product",
        "model_resolution_note",
        "forecast_type",
        "grid_id",
        "grid_polygon_method",
        "grid_polygon_crs",
        "soil_water_model_note",
    ]

    int16_cols = [
        "valid_hour_utc",
        "lead_hour",
    ]

    float64_cols = [
        "latitude",
        "longitude",
    ]

    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    for col in int16_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

    for col in float64_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    numeric_cols = [
        col for col in df.columns
        if col not in string_cols
        and col not in int16_cols
        and col not in float64_cols
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    return df


# ============================================================
# 8. Grid geometry
# ============================================================
# ============================================================
# 8. Grid geometry
# ============================================================

_IOWA_BOUNDARY_PROJECTED = None


def estimate_half_spacing_degrees(values: pd.Series, fallback: float = 0.01) -> float:
    """
    Estimate half spacing for regular latitude/longitude grids.

    This is appropriate for regular lat/lon models such as GFS.
    It is not appropriate for curvilinear grids such as HRRR.
    """
    vals = np.sort(pd.Series(values).dropna().unique())

    if len(vals) < 2:
        return fallback

    diffs = np.diff(vals)
    diffs = diffs[diffs > 1e-12]

    if len(diffs) == 0:
        return fallback

    return float(np.median(diffs) / 2.0)


def model_uses_regular_latlon(model: str) -> bool:
    """Return True for models whose grid is regular latitude/longitude."""
    regular_models = {
        str(x).lower()
        for x in getattr(cfg, "REGULAR_LATLON_MODELS", ["gfs"])
    }

    return str(model).lower() in regular_models


def get_model_cell_size_m(model: str, product: str) -> float:
    """
    Return approximate model grid-cell size in meters for projected-grid models.

    This is used for HRRR/RAP/NAM-style grids, not for regular lat/lon GFS.
    """
    model = str(model).lower()
    product = str(product)

    default_size = float(getattr(cfg, "DEFAULT_GRID_CELL_SIZE_M", 9000))
    size_cfg = getattr(cfg, "MODEL_GRID_CELL_SIZE_M", {})
    model_size = size_cfg.get(model, default_size)

    if isinstance(model_size, dict):
        if product in model_size:
            return float(model_size[product])

        for product_key, size_value in model_size.items():
            if str(product_key) in product:
                return float(size_value)

        return default_size

    return float(model_size)


def load_iowa_boundary_projected():
    """
    Load township shapefile, dissolve to one Iowa boundary, and project to meters.

    Returns None if clipping is disabled or the township file is unavailable.
    """
    global _IOWA_BOUNDARY_PROJECTED

    if not getattr(cfg, "CLIP_GRID_POLYGONS_TO_IOWA", True):
        return None

    if _IOWA_BOUNDARY_PROJECTED is not None:
        return _IOWA_BOUNDARY_PROJECTED

    township_file = getattr(cfg, "TOWNSHIP_FILE", None)

    if township_file is None:
        print("WARNING: cfg.TOWNSHIP_FILE is not defined. Grid polygons will not be clipped.")
        return None

    township_file = Path(township_file)

    if not township_file.exists():
        print("WARNING: Township shapefile not found. Grid polygons will not be clipped.")
        print(township_file)
        return None

    townships = gpd.read_file(township_file)

    if townships.crs is None:
        raise ValueError(
            "Township CRS is missing. Please define the shapefile CRS before clipping."
        )

    projected_crs = getattr(cfg, "GRID_PROJECTED_CRS", "EPSG:5070")
    townships = townships.to_crs(projected_crs)

    try:
        boundary_geom = townships.geometry.union_all()
    except AttributeError:
        boundary_geom = townships.geometry.unary_union

    _IOWA_BOUNDARY_PROJECTED = gpd.GeoDataFrame(
        geometry=[boundary_geom],
        crs=projected_crs,
    )

    return _IOWA_BOUNDARY_PROJECTED


def clip_cells_to_iowa(cells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clip grid cells to Iowa boundary if available.

    Cells may be in EPSG:4326; clipping is done in projected CRS.
    """
    iowa_boundary = load_iowa_boundary_projected()

    if iowa_boundary is None:
        return cells

    original_crs = cells.crs
    projected_crs = getattr(cfg, "GRID_PROJECTED_CRS", "EPSG:5070")

    cells_m = cells.to_crs(projected_crs)
    clipped_m = gpd.clip(cells_m, iowa_boundary)

    if clipped_m.empty:
        print("WARNING: Clipping removed all grid cells. Using unclipped cells instead.")
        return cells

    return clipped_m.to_crs(original_crs)


def build_regular_latlon_cells(sub: pd.DataFrame, model: str, product: str) -> gpd.GeoDataFrame:
    """
    Build rectangular cells for regular latitude/longitude grids such as GFS.
    """
    output_crs = getattr(cfg, "GRID_POLYGON_CRS", "EPSG:4326")

    half_lat = estimate_half_spacing_degrees(sub["latitude"])
    half_lon = estimate_half_spacing_degrees(sub["longitude"])

    cells = sub.copy()

    cells["geometry"] = [
        box(
            lon - half_lon,
            lat - half_lat,
            lon + half_lon,
            lat + half_lat,
        )
        for lon, lat in zip(cells["longitude"], cells["latitude"])
    ]

    cells = gpd.GeoDataFrame(
        cells,
        geometry="geometry",
        crs=output_crs,
    )

    cells["grid_polygon_method"] = (
        f"regular_latlon_rectangle_from_{model}_{product}_centroid_spacing"
    )
    cells["grid_polygon_crs"] = output_crs

    cells = clip_cells_to_iowa(cells)

    return cells


def build_projected_square_cells(sub: pd.DataFrame, model: str, product: str) -> gpd.GeoDataFrame:
    """
    Build approximate square grid cells in meters for curvilinear/projected grids.

    This is appropriate for HRRR/RAP/NAM-style model grids.
    """
    output_crs = getattr(cfg, "GRID_POLYGON_CRS", "EPSG:4326")
    projected_crs = getattr(cfg, "GRID_PROJECTED_CRS", "EPSG:5070")

    cell_size_m = get_model_cell_size_m(model=model, product=product)
    half_size_m = cell_size_m / 2.0

    points = gpd.GeoDataFrame(
        sub.copy(),
        geometry=gpd.points_from_xy(sub["longitude"], sub["latitude"]),
        crs=output_crs,
    )

    points_m = points.to_crs(projected_crs)

    points_m["geometry"] = [
        box(
            point.x - half_size_m,
            point.y - half_size_m,
            point.x + half_size_m,
            point.y + half_size_m,
        )
        for point in points_m.geometry
    ]

    method_crs = str(projected_crs).replace(":", "")
    points_m["grid_polygon_method"] = (
        f"approx_{int(cell_size_m)}m_square_from_"
        f"{model}_{product}_centroid_projected_{method_crs}"
    )
    points_m["grid_polygon_crs"] = output_crs

    cells = points_m.to_crs(output_crs)
    cells = clip_cells_to_iowa(cells)

    return cells


def build_grid_geometry_from_pixel_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Build approximate model grid-cell polygons from model grid-point centroids.

    GFS:
        regular latitude/longitude rectangles.

    HRRR/RAP/NAM:
        projected meter-based approximate square cells.

    Important:
        These are approximate support polygons, not official native model-cell boundaries.
    """
    needed = [
        "model",
        "product",
        "model_resolution_note",
        "grid_id",
        "latitude",
        "longitude",
    ]

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build grid geometry; missing columns: {missing}")

    geom_df = (
        df[needed]
        .drop_duplicates(subset=["model", "product", "grid_id"])
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )

    if geom_df.empty:
        raise ValueError("Cannot build grid geometry from an empty centroid table.")

    out_parts = []

    for (model, product), sub in geom_df.groupby(["model", "product"], dropna=False):
        model = str(model)
        product = str(product)

        if model_uses_regular_latlon(model):
            cells = build_regular_latlon_cells(
                sub=sub,
                model=model,
                product=product,
            )
        else:
            cells = build_projected_square_cells(
                sub=sub,
                model=model,
                product=product,
            )

        out_parts.append(cells)

    out = pd.concat(out_parts, ignore_index=True)
    out = gpd.GeoDataFrame(
        out,
        geometry="geometry",
        crs=getattr(cfg, "GRID_POLYGON_CRS", "EPSG:4326"),
    )

    out = out[
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

    out = out.drop_duplicates(
        subset=["model", "product", "grid_id"],
        keep="last",
    )

    try:
        projected_crs = getattr(cfg, "GRID_PROJECTED_CRS", "EPSG:5070")
        area_km2 = out.to_crs(projected_crs).geometry.area / 1_000_000
        print("Grid-cell area km² summary:")
        print(area_km2.describe())
    except Exception as e:
        print(f"WARNING: Could not compute grid-cell area summary. Reason: {e}")

    return out

# def estimate_half_spacing(values: pd.Series, fallback: float = 0.01) -> float:
#     """
#     Estimate half grid spacing from sorted unique coordinate values.

#     This is an approximation in degrees.
#     """
#     vals = np.sort(pd.Series(values).dropna().unique())

#     if len(vals) < 2:
#         return fallback

#     diffs = np.diff(vals)
#     diffs = diffs[diffs > 1e-12]

#     if len(diffs) == 0:
#         return fallback

#     return float(np.median(diffs) / 2.0)


# def make_rect_polygon(
#     lon: float,
#     lat: float,
#     half_lon: float,
#     half_lat: float,
# ) -> Polygon:
#     """
#     Build an approximate rectangular grid-cell polygon in EPSG:4326.

#     Coordinates are ordered as lon/lat.

#     These are approximate polygons from grid-point spacing, not official native
#     model grid-cell boundaries.
#     """
#     return Polygon(
#         [
#             (lon - half_lon, lat - half_lat),
#             (lon + half_lon, lat - half_lat),
#             (lon + half_lon, lat + half_lat),
#             (lon - half_lon, lat + half_lat),
#             (lon - half_lon, lat - half_lat),
#         ]
#     )


# def build_grid_geometry_from_pixel_df(df: pd.DataFrame) -> gpd.GeoDataFrame:
#     """
#     Build grid geometry from one successful pixel-level weather chunk.

#     Output:
#         one row = one model/product/grid_id
#         geometry = approximate grid-cell polygon

#     Important:
#         These polygons are approximate model grid-cell footprints built from
#         grid-point spacing in latitude/longitude. The grid points are treated
#         as cell centers / representative points.
#     """
#     needed = [
#         "model",
#         "product",
#         "model_resolution_note",
#         "grid_id",
#         "latitude",
#         "longitude",
#     ]

#     missing = [c for c in needed if c not in df.columns]
#     if missing:
#         raise ValueError(f"Cannot build grid geometry; missing columns: {missing}")

#     geom_df = (
#         df[needed]
#         .drop_duplicates(subset=["model", "product", "grid_id"])
#         .dropna(subset=["latitude", "longitude"])
#         .copy()
#     )

#     out_parts = []

#     for (model, product), sub in geom_df.groupby(["model", "product"], dropna=False):
#         half_lat = estimate_half_spacing(sub["latitude"])
#         half_lon = estimate_half_spacing(sub["longitude"])

#         sub = sub.copy()
#         sub["geometry"] = [
#             make_rect_polygon(lon, lat, half_lon, half_lat)
#             for lon, lat in zip(sub["longitude"], sub["latitude"])
#         ]

#         sub["grid_polygon_method"] = cfg.GRID_POLYGON_METHOD
#         sub["grid_polygon_crs"] = cfg.GRID_POLYGON_CRS

#         out_parts.append(sub)

#     out = pd.concat(out_parts, ignore_index=True)

#     gdf = gpd.GeoDataFrame(
#         out,
#         geometry="geometry",
#         crs=cfg.GRID_POLYGON_CRS,
#     )

#     return gdf[
#         [
#             "model",
#             "product",
#             "model_resolution_note",
#             "grid_id",
#             "latitude",
#             "longitude",
#             "grid_polygon_method",
#             "grid_polygon_crs",
#             "geometry",
#         ]
#     ]


# ============================================================
# 9. Herbie extraction
# ============================================================

def try_open_with_products(
    model_key: str,
    model_cfg: dict[str, Any],
    init_time: pd.Timestamp,
    lead_hour: int,
):
    """
    Try model product candidates until one works.

    Herbie downloads/opens the matching NOAA forecast file and extracts only
    variables matching model_cfg["search_string"].
    """
    errors = []

    for product in get_product_candidates(model_cfg):
        try:
            H = Herbie(
                init_time,
                model=model_cfg["model"],
                product=product,
                fxx=lead_hour,
                save_dir=model_cfg["raw_dir"],
            )

            ds_or_list = H.xarray(model_cfg["search_string"])
            return ds_or_list, product

        except Exception as e:
            errors.append(f"{product}: {e}")

    raise RuntimeError(
        f"All product candidates failed for model={model_key}, "
        f"init_time={init_time}, lead_hour={lead_hour}. "
        f"Errors: {' | '.join(errors)}"
    )


def extract_one_model_lead(
    model_key: str,
    model_cfg: dict[str, Any],
    valid_time: pd.Timestamp,
    lead_hour: int,
    forecast_type: str,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """
    Extract one model, one valid time, and one lead hour.

    Example:
        valid_time = 2020-05-26 00:00
        lead_hour = 24
        init_time = 2020-05-25 00:00
    """
    init_time = valid_time - pd.Timedelta(hours=lead_hour)

    try:
        ds_or_list, product_used = try_open_with_products(
            model_key=model_key,
            model_cfg=model_cfg,
            init_time=init_time,
            lead_hour=lead_hour,
        )

        df = ds_or_list_to_table(ds_or_list)

        # Free xarray objects as soon as possible.
        del ds_or_list
        gc.collect()

        if df.empty:
            raise ValueError("No rows left after spatial subsetting.")

        df["valid_date"] = valid_time.date().isoformat()
        df["valid_time"] = valid_time.isoformat()
        df["init_time"] = init_time.isoformat()
        df["valid_hour_utc"] = valid_time.hour
        df["model"] = model_key
        df["product"] = product_used
        df["model_resolution_note"] = cfg.MODEL_RESOLUTION_NOTES.get(model_key, pd.NA)
        df["lead_hour"] = lead_hour
        df["forecast_type"] = forecast_type
        df["grid_polygon_method"] = cfg.GRID_POLYGON_METHOD
        df["grid_polygon_crs"] = cfg.GRID_POLYGON_CRS
        df["soil_water_model_note"] = cfg.SOIL_WATER_MODEL_NOTE

        df = add_standard_variables(df)
        df = add_grid_id(df)
        df = keep_pixel_columns(df)
        df = standardize_dtypes(df)

        return df, None

    except Exception as e:
        missing = {
            "valid_date": valid_time.date().isoformat(),
            "valid_time": valid_time.isoformat(),
            "init_time": init_time.isoformat(),
            "valid_hour_utc": valid_time.hour,
            "model": model_key,
            "model_name": model_cfg["model"],
            "lead_hour": lead_hour,
            "forecast_type": forecast_type,
            "reason": str(e),
        }

        return None, missing


# ============================================================
# 10. Worker task wrapper
# ============================================================

def make_task_cache_dir(
    base_raw_dir: Path,
    model_key: str,
    valid_time: pd.Timestamp,
    lead_hour: int,
) -> Path:
    """
    Make a task-specific cache folder.

    This avoids different parallel workers writing cfgrib index/cache files
    into the same exact folder at the same time.
    """
    stamp = valid_time.strftime("%Y%m%d_%H%M")
    return base_raw_dir / f"{model_key}_{stamp}_f{lead_hour:03d}"


def run_extract_task(task: dict[str, Any]) -> dict[str, Any]:
    """
    Worker function.

    This runs in a separate process when ProcessPoolExecutor is used.
    It returns either a DataFrame or a missing-log dictionary.
    """
    model_key = task["model_key"]
    model_cfg = dict(task["model_cfg"])
    valid_time = pd.Timestamp(task["valid_time"])
    lead_hour = int(task["lead_hour"])
    forecast_type = task["forecast_type"]

    base_raw_dir = Path(model_cfg["raw_dir"])
    task_raw_dir = make_task_cache_dir(
        base_raw_dir=base_raw_dir,
        model_key=model_key,
        valid_time=valid_time,
        lead_hour=lead_hour,
    )

    model_cfg["raw_dir"] = task_raw_dir
    task_raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        df, missing = extract_one_model_lead(
            model_key=model_key,
            model_cfg=model_cfg,
            valid_time=valid_time,
            lead_hour=lead_hour,
            forecast_type=forecast_type,
        )

        return {
            "df": df,
            "missing": missing,
            "model_key": model_key,
            "valid_time": valid_time.isoformat(),
            "lead_hour": lead_hour,
            "forecast_type": forecast_type,
        }

    finally:
        if getattr(cfg, "CLEAN_RAW_CACHE_AFTER_TASK", True):
            shutil.rmtree(task_raw_dir, ignore_errors=True)

        gc.collect()


def make_tasks_for_model_year(
    model_key: str,
    model_cfg: dict[str, Any],
    year: int,
) -> list[dict[str, Any]]:
    """
    Create extraction tasks for one model-year.

    One task = one valid_time and one forecast lead.
    """
    dates = get_dates_for_year(year)

    if len(dates) == 0:
        return []

    valid_times = make_valid_times_for_model(dates, model_cfg)

    tasks = []

    for valid_time in valid_times:
        for lead_hour, forecast_type in model_cfg["lead_hours"].items():
            tasks.append(
                {
                    "model_key": model_key,
                    "model_cfg": model_cfg,
                    "year": year,
                    "valid_time": valid_time.isoformat(),
                    "lead_hour": int(lead_hour),
                    "forecast_type": forecast_type,
                }
            )

    return tasks


def iter_task_results(tasks: list[dict[str, Any]], max_workers: int):
    """
    Yield task results with a small bounded queue.

    This avoids submitting hundreds of tasks and letting completed DataFrames
    pile up in memory.
    """
    if not tasks:
        return

    max_pending = get_max_pending_tasks(max_workers)
    max_tasks_per_child = get_max_tasks_per_child()

    executor_kwargs: dict[str, Any] = {
        "max_workers": max_workers,
    }

    if max_tasks_per_child is not None:
        executor_kwargs["max_tasks_per_child"] = max_tasks_per_child

    task_iter = iter(tasks)

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        pending = set()

        def submit_next() -> bool:
            try:
                task = next(task_iter)
            except StopIteration:
                return False

            future = executor.submit(run_extract_task, task)
            pending.add(future)
            return True

        for _ in range(min(max_pending, len(tasks))):
            submit_next()

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                result = future.result()
                yield result

                submit_next()


# ============================================================
# 11. Daily Iowa average
# ============================================================

def make_daily_iowa_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one daily Iowa-wide average per date/model/lead/forecast type.

    This is disabled unless SAVE_DAILY_IOWA_AVERAGE_YEARLY = True.
    """
    group_cols = [
        col for col in cfg.DAILY_AVG_GROUP_COLUMNS
        if col in df.columns
    ]

    value_cols = [
        col for col in cfg.PIXEL_OUTPUT_COLUMNS
        if col not in group_cols
        and col not in ["latitude", "longitude", "grid_id"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    daily = (
        df.groupby(group_cols, dropna=False)[value_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    return daily

def read_unique_grid_points_from_pixel_file(
    pixel_path: Path,
    model_key: str,
) -> pd.DataFrame:
    """
    Read only unique grid centroids from an existing pixel parquet file.

    This is memory safer than reading the whole yearly weather file.
    Used when RESUME_SKIP_EXISTING=True but geometry should still be rebuilt.
    """
    needed = [
        "model",
        "product",
        "model_resolution_note",
        "grid_id",
        "latitude",
        "longitude",
    ]

    available = pq.read_schema(pixel_path).names
    missing = [c for c in needed if c not in available]

    if missing:
        raise ValueError(
            f"Cannot rebuild geometry from {pixel_path}; missing columns: {missing}"
        )

    pf = pq.ParquetFile(pixel_path)
    parts = []

    for row_group_idx in range(pf.num_row_groups):
        chunk = pf.read_row_group(
            row_group_idx,
            columns=needed,
        ).to_pandas()

        chunk = chunk[
            chunk["model"].astype(str).str.lower() == str(model_key).lower()
        ].copy()

        if chunk.empty:
            del chunk
            continue

        chunk = (
            chunk[needed]
            .drop_duplicates(subset=["model", "product", "grid_id"])
            .dropna(subset=["latitude", "longitude"])
        )

        if not chunk.empty:
            parts.append(chunk)

        del chunk
        gc.collect()

    if not parts:
        return pd.DataFrame(columns=needed)

    out = pd.concat(parts, ignore_index=True)

    out = out.drop_duplicates(
        subset=["model", "product", "grid_id"],
        keep="last",
    )

    return out
# ============================================================
# 12. Chunked output writer
# ============================================================

class ChunkedOutputWriter:
    """
    Write model-year pixel files chunk by chunk.

    For Parquet:
        uses PyArrow ParquetWriter with stable schema.

    For CSV:
        appends chunks to one CSV file.

    Important:
        This writes to a temporary file first.
        The final file is created only after the model-year finishes.
    """

    def __init__(self, final_path: Path):
        self.final_path = final_path
        self.tmp_path = final_path.with_name(final_path.name + ".tmp")
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows_written = 0
        self.csv_header_written = False

        self.final_path.parent.mkdir(parents=True, exist_ok=True)

        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return

        if self.final_path.suffix == ".parquet":
            table = pa.Table.from_pandas(df, preserve_index=False)

            if self.writer is None:
                self.schema = table.schema
                self.writer = pq.ParquetWriter(self.tmp_path, self.schema)
            else:
                table = table.cast(self.schema)

            self.writer.write_table(table)

        elif self.final_path.suffix == ".csv":
            df.to_csv(
                self.tmp_path,
                mode="a",
                header=not self.csv_header_written,
                index=False,
            )
            self.csv_header_written = True

        else:
            raise ValueError(f"Unsupported output suffix: {self.final_path.suffix}")

        self.rows_written += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.schema = None

    def finalize(self) -> Path | None:
        """
        Close and rename temp file to final file.
        """
        self.close()

        if self.rows_written == 0:
            self.cleanup_tmp()
            return None

        self.tmp_path.replace(self.final_path)
        return self.final_path

    def cleanup_tmp(self) -> None:
        self.close()

        if self.tmp_path.exists():
            self.tmp_path.unlink()




# ============================================================
# 13. Yearly processing
# ============================================================
def process_model_year(
    model_key: str,
    model_cfg: dict[str, Any],
    year: int,
) -> tuple[Path | None, Path | None, gpd.GeoDataFrame | None, list[dict[str, Any]]]:
    """
    Process one model for one year.

    Memory-safe behavior:
    - creates one task per date/lead
    - runs tasks through a bounded worker pool
    - writes pixel chunks immediately
    - builds geometry only once per model/product
    - logs missing chunks instead of creating full NA grids
    - if pixel file already exists, optionally rebuild geometry from it
    """
    if cfg.FILL_MISSING_WITH_NA:
        raise ValueError(
            "This efficient version does not support FILL_MISSING_WITH_NA=True. "
            "Set FILL_MISSING_WITH_NA=False to avoid huge NA grids."
        )

    suffix = output_suffix()

    pixel_out = model_cfg["pixel_dir"] / f"{model_key}_weather_iowa_pixels_{year}{suffix}"
    daily_out = model_cfg["daily_dir"] / f"{model_key}_weather_iowa_daily_avg_{year}{suffix}"

    # ------------------------------------------------------------
    # If pixel file already exists, skip extraction but optionally
    # rebuild the geometry from the existing pixel file.
    # ------------------------------------------------------------
    if cfg.RESUME_SKIP_EXISTING and pixel_out.exists():
        print(f"Skipping existing pixel file: {pixel_out}")

        grid_gdf = None

        if (
            cfg.SAVE_COMBINED_GRID_GEOMETRY_YEARLY
            and getattr(cfg, "REBUILD_GEOMETRY_WHEN_PIXEL_EXISTS", True)
        ):
            try:
                print("Rebuilding grid geometry from existing pixel file...")

                unique_grid_df = read_unique_grid_points_from_pixel_file(
                    pixel_path=pixel_out,
                    model_key=model_key,
                )

                if not unique_grid_df.empty:
                    grid_gdf = build_grid_geometry_from_pixel_df(unique_grid_df)
                    print(f"Rebuilt geometry cells: {len(grid_gdf):,}")
                else:
                    print("WARNING: No unique grid points found for geometry rebuild.")

            except Exception as e:
                print("WARNING: Could not rebuild geometry from existing pixel file.")
                print(f"Reason: {e}")

        return pixel_out, daily_out if daily_out.exists() else None, grid_gdf, []

    # ------------------------------------------------------------
    # If pixel file does NOT exist, process the weather model-year.
    # ------------------------------------------------------------
    tasks = make_tasks_for_model_year(
        model_key=model_key,
        model_cfg=model_cfg,
        year=year,
    )

    if not tasks:
        print(f"No tasks created for {model_key} {year}.")
        return None, None, None, []

    max_workers = get_max_workers()

    print("=" * 70)
    print(f"Processing model-year: {model_key} {year}")
    print(f"Tasks: {len(tasks)}")
    print(f"Workers: {max_workers}")
    print(f"Max pending tasks: {get_max_pending_tasks(max_workers)}")
    print(f"Max tasks per child: {get_max_tasks_per_child()}")
    print("=" * 70)

    writer = ChunkedOutputWriter(pixel_out)

    missing_logs: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []

    grid_gdf_parts: list[gpd.GeoDataFrame] = []
    seen_geometry_keys: set[tuple[str, str]] = set()

    completed = 0
    successful = 0
    missing_count = 0

    try:
        for result in iter_task_results(tasks, max_workers=max_workers):
            completed += 1

            df = result["df"]
            missing = result["missing"]

            if df is not None and not df.empty:
                successful += 1

                if cfg.PRINT_PROGRESS:
                    print("-" * 70)
                    print(
                        f"[{completed}/{len(tasks)}] "
                        f"{model_key} valid={result['valid_time']} "
                        f"lead={result['lead_hour']} rows={len(df)}"
                    )

                    if cfg.PRINT_COORDINATE_RANGES:
                        print("Latitude:", df["latitude"].min(), "to", df["latitude"].max())
                        print("Longitude:", df["longitude"].min(), "to", df["longitude"].max())

                    if cfg.PRINT_COLUMNS:
                        print(list(df.columns))

                writer.write(df)

                if cfg.SAVE_DAILY_IOWA_AVERAGE_YEARLY:
                    daily_frames.append(make_daily_iowa_average(df))

                product_values = (
                    df[["model", "product"]]
                    .drop_duplicates()
                    .dropna()
                    .itertuples(index=False, name=None)
                )

                for model_value, product_value in product_values:
                    geom_key = (str(model_value), str(product_value))

                    if geom_key not in seen_geometry_keys:
                        sub = df[
                            (df["model"].astype(str) == str(model_value))
                            & (df["product"].astype(str) == str(product_value))
                        ].copy()

                        grid_gdf_parts.append(build_grid_geometry_from_pixel_df(sub))
                        seen_geometry_keys.add(geom_key)

                del df
                gc.collect()

            else:
                missing_count += 1

                if missing is not None:
                    missing_logs.append(missing)

                print("-" * 70)
                print(
                    f"[{completed}/{len(tasks)}] WARNING: missing "
                    f"{model_key} valid={result['valid_time']} "
                    f"lead={result['lead_hour']}"
                )

                if missing is not None:
                    print("Reason:", missing["reason"])

        final_pixel_path = writer.finalize()

    except Exception:
        writer.cleanup_tmp()
        raise

    if final_pixel_path is None:
        print(f"No pixel file produced for {model_key} {year}.")
        return None, None, None, missing_logs

    daily_path = None

    if cfg.SAVE_DAILY_IOWA_AVERAGE_YEARLY and daily_frames:
        daily = pd.concat(daily_frames, ignore_index=True, sort=False)
        daily_path = daily_out
        save_table(daily, daily_path)
        print(f"Saved daily Iowa-average yearly file: {daily_path}")
        print(f"Rows: {len(daily)}")

        del daily
        gc.collect()

    grid_gdf = None

    if grid_gdf_parts:
        grid_gdf = pd.concat(grid_gdf_parts, ignore_index=True)
        grid_gdf = gpd.GeoDataFrame(
            grid_gdf,
            geometry="geometry",
            crs=cfg.GRID_POLYGON_CRS,
        )

        grid_gdf = grid_gdf.drop_duplicates(
            subset=["model", "product", "grid_id"],
            keep="last",
        )

    print("=" * 70)
    print(f"Saved pixel-level yearly file: {final_pixel_path}")
    print(f"Successful chunks: {successful}")
    print(f"Missing chunks: {missing_count}")
    print(f"Rows written: {writer.rows_written}")

    if cfg.CLEAN_RAW_CACHE_AFTER_MODEL_YEAR:
        if model_cfg["raw_dir"].exists():
            shutil.rmtree(model_cfg["raw_dir"], ignore_errors=True)
            model_cfg["raw_dir"].mkdir(parents=True, exist_ok=True)
            print(f"Cleaned raw cache: {model_cfg['raw_dir']}")

    return final_pixel_path, daily_path, grid_gdf, missing_logs

# ============================================================
# 14. Logs and geometry
# ============================================================

def save_missing_log(missing_logs: list[dict[str, Any]]) -> None:
    """Save missing model/date/lead records."""
    if not missing_logs:
        print("No missing files logged.")
        return

    missing_df = pd.DataFrame(missing_logs)
    cfg.MISSING_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(cfg.MISSING_LOG_FILE, index=False)

    print("=" * 70)
    print("Saved missing log:")
    print(cfg.MISSING_LOG_FILE)


def save_manifest(records: list[dict[str, Any]]) -> None:
    """Save manifest of produced outputs."""
    manifest = pd.DataFrame(records)

    cfg.MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.MANIFEST_FILE, index=False)

    print("=" * 70)
    print("Saved manifest:")
    print(cfg.MANIFEST_FILE)


def save_or_merge_year_geometry(
    new_grid_gdfs: list[gpd.GeoDataFrame],
    geometry_path: Path,
) -> Path | None:
    """
    Save yearly grid geometry.

    If geometry_path already exists, merge old + new and drop duplicates.

    Important:
        New geometry is kept over old geometry for the same model/product/grid_id.
        This prevents old tiny polygons from surviving.
    """
    if not new_grid_gdfs:
        return None

    combined_parts = []

    if geometry_path.exists():
        try:
            old = gpd.read_parquet(geometry_path)
            combined_parts.append(old)
        except Exception as e:
            print(f"WARNING: Could not read existing geometry file: {geometry_path}")
            print(f"Reason: {e}")

    combined_parts.extend(new_grid_gdfs)

    combined_grid = pd.concat(combined_parts, ignore_index=True)
    combined_grid = gpd.GeoDataFrame(
        combined_grid,
        geometry="geometry",
        crs=cfg.GRID_POLYGON_CRS,
    )

    combined_grid = combined_grid.drop_duplicates(
        subset=["model", "product", "grid_id"],
        keep="last",
    )

    save_geoparquet(combined_grid, geometry_path)

    print("=" * 70)
    print("Saved or updated grid-geometry GeoParquet:")
    print(geometry_path)
    print(f"Grid cells: {len(combined_grid):,}")

    try:
        projected_crs = getattr(cfg, "GRID_PROJECTED_CRS", "EPSG:5070")
        area_km2 = combined_grid.to_crs(projected_crs).geometry.area / 1_000_000

        print("Combined grid-cell area km² summary:")
        print(area_km2.describe())
    except Exception as e:
        print(f"WARNING: Could not compute combined grid area summary. Reason: {e}")

    return geometry_path

# ============================================================
# 15. Main
# ============================================================

def main() -> None:
    print("=" * 70)
    print("Archived weather extraction started")
    print("Date range:", cfg.START_DATE, "to", cfg.END_DATE)
    print("Years to process:", year_list())
    print(
        "Models to process:",
        "all" if cfg.MODELS_TO_PROCESS is None else cfg.MODELS_TO_PROCESS,
    )
    print("Pixel yearly folder:", cfg.PIXEL_YEARLY_DIR)
    print("Daily Iowa-average yearly folder:", cfg.DAILY_IOWA_YEARLY_DIR)
    print("Combined grid geometry yearly folder:", cfg.GRID_GEOMETRY_YEARLY_DIR)
    print("=" * 70)

    all_missing_logs: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []

    for year in year_list():
        year_grid_gdfs: list[gpd.GeoDataFrame] = []

        for model_key, model_cfg in model_items_to_process():

            pixel_path, daily_path, grid_gdf, missing_logs = process_model_year(
                model_key=model_key,
                model_cfg=model_cfg,
                year=year,
            )

            all_missing_logs.extend(missing_logs)

            if grid_gdf is not None and not grid_gdf.empty:
                year_grid_gdfs.append(grid_gdf)

            manifest_records.append(
                {
                    "year": year,
                    "model": model_key,
                    "pixel_file": str(pixel_path) if pixel_path is not None else None,
                    "daily_iowa_average_file": str(daily_path) if daily_path is not None else None,
                }
            )

        geometry_path = None

        if cfg.SAVE_COMBINED_GRID_GEOMETRY_YEARLY and year_grid_gdfs:
            geometry_path = (
                cfg.GRID_GEOMETRY_YEARLY_DIR
                / f"weather_model_grids_iowa_{year}.geoparquet"
            )

            geometry_path = save_or_merge_year_geometry(
                new_grid_gdfs=year_grid_gdfs,
                geometry_path=geometry_path,
            )

        # Add geometry file path to manifest records for that year.
        for record in manifest_records:
            if record["year"] == year:
                record["combined_grid_geometry_file"] = (
                    str(geometry_path) if geometry_path is not None else None
                )

    save_manifest(manifest_records)

    if cfg.SAVE_MISSING_LOG:
        save_missing_log(all_missing_logs)

    print("=" * 70)
    print("Done.")
    print("Pixel-level yearly files:")
    print(cfg.PIXEL_YEARLY_DIR)
    print("Daily Iowa-average yearly files:")
    print(cfg.DAILY_IOWA_YEARLY_DIR)
    print("Combined grid geometry yearly files:")
    print(cfg.GRID_GEOMETRY_YEARLY_DIR)


if __name__ == "__main__":
    main()