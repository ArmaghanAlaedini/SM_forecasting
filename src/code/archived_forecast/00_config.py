from pathlib import Path
import sys
import tempfile


# ============================================================
# 0. Find project root
# ============================================================

def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    base = start if start.is_dir() else start.parent

    for path in [base, *base.parents]:
        if (path / ".git").exists():
            return path
        if (path / "environment.yml").exists() and (path / "src").exists():
            return path

    raise RuntimeError("Could not find project root.")


try:
    PROJECT_ROOT = _find_project_root(Path(__file__))
except NameError:
    PROJECT_ROOT = _find_project_root(Path.cwd())


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. Clean base folders
# ============================================================
# We intentionally do NOT call paths.ensure_dirs() here.
# That function creates extra archived_weather folders we do not need.

PROCESSED_ARCHIVED_WEATHER = (
    PROJECT_ROOT / "src" / "data" / "processed" / "archived_weather"
)

YEARLY_OUTPUT_DIR = PROCESSED_ARCHIVED_WEATHER / "yearly"
LOG_DIR = YEARLY_OUTPUT_DIR / "_logs"

YEARLY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Herbie needs a download directory.
# Put it outside the project so raw GRIB/cache files are not kept in src/data/raw in this project
HERBIE_CACHE_ROOT = Path(tempfile.gettempdir()) / "SM_forecasting_herbie_cache"
HERBIE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Date and run controls
# ============================================================

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
DATE_FREQ = "D"

TEST_N_DAYS = 5

# Run one year at a time for safety. Use None to process all years.
# YEARS_TO_PROCESS = [2020, 2021, 2022, 2023, 2024, 2025]
YEARS_TO_PROCESS = [2020]

# None means all models: hrrr, gfs, rap, nam. One model can be written like this: ["hrrr"]
MODELS_TO_PROCESS = ["gfs"]


# ============================================================
# 3. Spatial domain
# ============================================================
# This is Iowa boundaries and the same window I used for SMAP data
BBOX = {
    "west": -97.0,
    "south": 40.0,
    "east": -89.0,
    "north": 44.0,
}


# ============================================================
# 4. Lead-time design
# ============================================================

HRRR_LEAD_HOURS = {
    0: "analysis_f00", # This is not original weather data it is just model nowcast, Can be used for comparison with IEM data
    24: "forecast_1_day",
    # 48: "forecast_2_day", #HRRR doesn't have 48 hours at all so we skipped it
}

FORECAST_ONLY_LEAD_HOURS = {
    24: "forecast_1_day",
    48: "forecast_2_day",
}


# ============================================================
# 5. Forecast variables
# ============================================================
# These notations are from RegEx and variables are from NOAA/Grib 

CORE_WEATHER_PATTERNS = [
    r":TMP:2 m above ground",
    r":DPT:2 m above ground",
    r":RH:2 m above ground",
    r":UGRD:10 m above ground",
    r":VGRD:10 m above ground",
    r":GUST:surface",
    r":PRATE:surface",
    r":APCP:surface",
    r":DSWRF:surface",
    r":LHTFL:surface",
    r":PRES:surface",
]

OPTIONAL_DAILY_STYLE_PATTERNS = [
    r":MAXT:2 m above ground",
    r":MINT:2 m above ground",
    r":TMAX:2 m above ground",
    r":TMIN:2 m above ground",
]

OPTIONAL_SOIL_PATTERNS = [
    r":SOILW:",
    r":SOILT:",
    r":TSOIL:",
]

HRRR_EXTRA_PATTERNS = [
    r":MSTAV:0 m underground",
    r":CNWAT:surface",
]


def make_search_string(patterns: list[str]) -> str:
    return "|".join(patterns)


BASE_SEARCH_STRING = make_search_string(
    CORE_WEATHER_PATTERNS
    + OPTIONAL_DAILY_STYLE_PATTERNS
    + OPTIONAL_SOIL_PATTERNS
)

HRRR_SEARCH_STRING = make_search_string(
    CORE_WEATHER_PATTERNS
    + OPTIONAL_DAILY_STYLE_PATTERNS
    + OPTIONAL_SOIL_PATTERNS
    + HRRR_EXTRA_PATTERNS
)


# ============================================================
# 6. Model metadata
# ============================================================
# Model description
MODEL_RESOLUTION_NOTES = {
    "hrrr": "HRRR; approx 3 km CONUS grid",
    "rap": "RAP; approx 13 km grid",
    "gfs": "GFS; 0.25/0.5/1.0 degree lat-lon products depending archive/date",
    "nam": "NAM CONUS nest; high-resolution nest output, commonly around 5 km for conusnest product",
}

# General model assumption
MODEL_CONFIGS = {
    "hrrr": {
        "model": "hrrr",
        "product_candidates": ["sfc"],
        "lead_hours": HRRR_LEAD_HOURS,
        "search_string": HRRR_SEARCH_STRING,
        "valid_hours_utc": [0],
    },

    "gfs": {
        "model": "gfs",
        "product_candidates": [
            "0.5-degree",
            "1.0-degree",
        ],
        "lead_hours": FORECAST_ONLY_LEAD_HOURS,
        "search_string": BASE_SEARCH_STRING,
        "valid_hours_utc": [0],
    },

    "rap": {
        "model": "rap",
        "product_candidates": [
            "awp130pgrb",
            "sfc",
        ],
        "lead_hours": FORECAST_ONLY_LEAD_HOURS,
        "search_string": BASE_SEARCH_STRING,
        "valid_hours_utc": [3],
    },

    "nam": {
        "model": "nam",
        "product_candidates": [
            "conusnest.hiresf",
            "awphys",
            "awip12",
        ],
        "lead_hours": FORECAST_ONLY_LEAD_HOURS,
        "search_string": BASE_SEARCH_STRING,
        "valid_hours_utc": [0],
    },
}


# ============================================================
# 7. Output folders
# ============================================================
# All final files go into:
# src/data/processed/archived_weather/yearly/
#
# For each year, expected final files:
#   hrrr_weather_iowa_pixels_2020.parquet
#   gfs_weather_iowa_pixels_2020.parquet
#   rap_weather_iowa_pixels_2020.parquet
#   nam_weather_iowa_pixels_2020.parquet
#   weather_model_grids_iowa_2020.geoparquet

PIXEL_YEARLY_DIR = YEARLY_OUTPUT_DIR
DAILY_IOWA_YEARLY_DIR = YEARLY_OUTPUT_DIR
GRID_GEOMETRY_YEARLY_DIR = YEARLY_OUTPUT_DIR

for model_key, model_cfg in MODEL_CONFIGS.items():
    model_cfg["raw_dir"] = HERBIE_CACHE_ROOT / model_key
    model_cfg["pixel_dir"] = YEARLY_OUTPUT_DIR
    model_cfg["daily_dir"] = YEARLY_OUTPUT_DIR

    model_cfg["raw_dir"].mkdir(parents=True, exist_ok=True)


# ============================================================
# 8. Grid geometry output
# ============================================================
# polygons boundaries
SAVE_COMBINED_GRID_GEOMETRY_YEARLY = True

GRID_POLYGON_CRS = "EPSG:4326"

# Model-specific approximate grid-cell polygons.
# GFS is handled as a regular lat/lon grid.
# HRRR/RAP/NAM are handled as projected meter-based approximate cells.
GRID_POLYGON_METHOD = "model_specific_approx_grid_cell_from_centroid"

GRID_PROJECTED_CRS = "EPSG:5070"   # USA Contiguous Albers, meters

CLIP_GRID_POLYGONS_TO_IOWA = True

# If pixel parquet already exists and RESUME_SKIP_EXISTING=True,
# still rebuild/update the geometry file from that existing parquet.
REBUILD_GEOMETRY_WHEN_PIXEL_EXISTS = True

TOWNSHIP_FILE = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "raw"
    / "townships"
    / "civil_townships_a_ia.shp"
)

# Models whose products are regular latitude/longitude grids.
REGULAR_LATLON_MODELS = ["gfs"]

# Approximate model grid-cell width in meters for non-regular-lat/lon grids.
# These are practical support polygons, not official native cell boundaries.
MODEL_GRID_CELL_SIZE_M = {
    "hrrr": 3000,
    "rap": 13000,

    "nam": {
        "conusnest.hiresf": 5000,
        "awphys": 12000,
        "awip12": 12000,
    },
}

DEFAULT_GRID_CELL_SIZE_M = 9000

# Just a heads up for ML models later
SOIL_WATER_MODEL_NOTE = (
    "soil_water_model is a forecast-model land-surface soil water variable "
    "when available. It is not the same as SMAP volumetric soil moisture "
    "or IEM station volumetric water content."
)


# ============================================================
# 9. Output behavior
# ============================================================

OUTPUT_FORMAT = "parquet" # This is a mode efficient way of storing data compared to csv

SAVE_PIXEL_YEARLY = True

# Computationally intensive. You can compute daily/monthly/yearly averages later from the pixel files.
SAVE_DAILY_IOWA_AVERAGE_YEARLY = False

RESUME_SKIP_EXISTING = True

# Delete Herbie cache after each model-year. 
CLEAN_RAW_CACHE_AFTER_MODEL_YEAR = True
# Manually delete if the process is killed gfrom the terminal.
# Codes for Linux Ubuntu OS for manually deleting
# rm -f src/data/processed/archived_weather/yearly/hrrr_weather_iowa_pixels_2020.parquet
# rm -f src/data/processed/archived_weather/yearly/weather_model_grids_iowa_2020.geoparquet
# rm -rf /tmp/SM_forecasting_herbie_cache


FILL_MISSING_WITH_NA = False

SAVE_MISSING_LOG = True
MISSING_LOG_FILE = LOG_DIR / "missing_weather_model_files.csv"
MANIFEST_FILE = LOG_DIR / "weather_yearly_manifest.csv"


# ============================================================
# 10. Final pixel-level columns
# ============================================================

PIXEL_OUTPUT_COLUMNS = [
    "valid_date",
    "valid_time",
    "init_time",
    "valid_hour_utc",
    "model",
    "product",
    "model_resolution_note",
    "lead_hour",
    "forecast_type",
    "latitude",
    "longitude",
    "grid_id",

    "grid_polygon_method",
    "grid_polygon_crs",
    "soil_water_model_note",

    "temperature_c",
    "temperature_f",
    "high_temperature_c",
    "high_temperature_f",
    "low_temperature_c",
    "low_temperature_f",
    "dewpoint_c",
    "dewpoint_f",
    "relative_humidity_percent",
    "wind_u_10m_mps",
    "wind_v_10m_mps",
    "wind_speed_10m_mps",
    "wind_speed_10m_mph",
    "wind_gust_mps",
    "wind_gust_mph",
    "precip_rate",
    "precip_accum_mm",
    "precip_accum_in",
    "downward_shortwave_radiation_wm2",
    "latent_heat_flux_wm2",
    "surface_pressure_hpa",

    "moisture_availability_percent",
    "canopy_water",
    "soil_water_model",
    "soil_temperature_c",
    "soil_temperature_f",
]


DAILY_AVG_GROUP_COLUMNS = [
    "valid_date",
    "valid_time",
    "init_time",
    "valid_hour_utc",
    "model",
    "product",
    "model_resolution_note",
    "lead_hour",
    "forecast_type",
]


# ============================================================
# 11. Verbosity
# ============================================================

PRINT_PROGRESS = True
PRINT_COORDINATE_RANGES = True
PRINT_COLUMNS = False

# ============================================================
# 12. Performance
# ============================================================
# Local laptop: keep this 1.
# HPC: try 2 first, then 4 if memory is fine.
MAX_WORKERS = 1

# Keep only a few tasks queued at once.
MAX_PENDING_TASKS = 2

# Restart worker process after this many tasks to reduce memory growth from cfgrib/xarray.
MAX_TASKS_PER_CHILD = 8