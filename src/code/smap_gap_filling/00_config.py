from pathlib import Path
import os


# ============================================================
# 0. Project root
# ============================================================

def find_project_root() -> Path:
    """
    Find the project root.

    Priority:
    1. Use SMAP_PROJECT_ROOT if set.
    2. Otherwise, walk upward until a project marker is found.
    """
    env_root = os.environ.get("SMAP_PROJECT_ROOT")

    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.exists():
            return root
        raise FileNotFoundError(f"SMAP_PROJECT_ROOT is set but does not exist: {root}")

    start = Path(__file__).resolve().parent

    for folder in [start, *start.parents]:
        has_src = (folder / "src").is_dir()
        has_git = (folder / ".git").exists()
        has_renv = (folder / "renv").is_dir()
        has_env = (folder / "environment.yml").exists()

        if has_src and (has_git or has_renv or has_env):
            return folder

    raise FileNotFoundError(
        "Could not find project root. Run from inside the project folder "
        "or set SMAP_PROJECT_ROOT."
    )


PROJECT_ROOT = find_project_root()
SRC_DIR = PROJECT_ROOT / "src"


# ============================================================
# 1. Data roots
# ============================================================

def get_data_root() -> Path:
    """
    Local default:
        PROJECT_ROOT / src / data

    HPC override:
        export SMAP_DATA_ROOT=/path/to/data
    """
    env_data = os.environ.get("SMAP_DATA_ROOT")

    if env_data:
        data_root = Path(env_data).expanduser().resolve()
    else:
        data_root = PROJECT_ROOT / "src" / "data"

    data_root.mkdir(parents=True, exist_ok=True)
    return data_root


DATA_ROOT = get_data_root()
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"


# ============================================================
# 2. Raw/support inputs
# ============================================================

RAW_SMAP_NC_DIR = Path(
    os.environ.get("SMAP_RAW_NC_DIR", RAW_DIR / "smap_observations")
).expanduser().resolve()

TOWNSHIP_SHP_PATH = RAW_DIR / "townships" / "civil_townships_a_ia.shp"


# ============================================================
# 3. Existing processed SMAP inputs
# ============================================================

SMAP_DETRENDED_DIR = PROCESSED_DIR / "smap_detrended"

SMAP_DETRENDED_AM_RDS_DIR = SMAP_DETRENDED_DIR / "am" / "rds"
SMAP_DETRENDED_PM_RDS_DIR = SMAP_DETRENDED_DIR / "pm" / "rds"

SMAP_DETRENDED_AM_CSV_DIR = SMAP_DETRENDED_DIR / "am" / "csv"
SMAP_DETRENDED_PM_CSV_DIR = SMAP_DETRENDED_DIR / "pm" / "csv"


# ============================================================
# 4. IEM station input
# ============================================================

IEM_STATION_DIR = PROCESSED_DIR / "isu_stations"

# Main expected file:
IEM_STATIONS_FULL_PATH = IEM_STATION_DIR / "stations_full.csv"

# Fallback name, in case your file is named this way:
IEM_STATIONS_FULL_FALLBACK_PATH = IEM_STATION_DIR / "full_stations.csv"


# ============================================================
# 5. Gap-filling output folders
# ============================================================

GAP_FILLING_DIR = PROCESSED_DIR / "smap_gap_filling"
SUPPORT_DIR = GAP_FILLING_DIR / "support"

SMAP_LATTICE_DIR = SUPPORT_DIR / "smap_lattice"

# IEM point-to-area kriged daily outputs.
# Daily CSV files will be saved directly here.
IEM_PTA_DIR = GAP_FILLING_DIR / "iem_point_to_area"

# NOTE: the old "01_preprocess_smap" and "02_preprocess_final" folders were
# never written to by any script (only defined here), so they have been
# removed. The real pipeline outputs start at "03_full_smap_iem_data".


# ============================================================
# 6. Study settings
# ============================================================

PASSES = ["am", "pm"]

TRAIN_YEARS = [2020, 2021, 2022, 2023]
VALIDATION_YEARS = [2024]   # held-out year used to train the stacking meta-model
TEST_YEAR = 2025
# 2024 MUST be included here, otherwise 03/05 never build the validation year
# and 10a/10b find "0 files in the validation split".
ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + [TEST_YEAR]

CRS_WGS84 = 4326
CRS_EASE = 6933

SMAP_CELLSIZE_M = 9024.31

IOWA_BBOX = (-97.0, 40.0, -89.0, 44.0)


# ============================================================
# 7. Runtime limits
# ============================================================

def get_file_limit(env_name: str, default: int | None = None) -> int | None:
    """
    Read optional integer limit from environment variable.

    Examples:
        export SMAP_MAX_DAYS=3
        export SMAP_LATTICE_SCAN_FILES=50

    Full run:
        unset SMAP_MAX_DAYS
    """
    value = os.environ.get(env_name)

    if value is None or value.strip() == "":
        return default

    value = value.strip().lower()

    if value in {"all", "none"}:
        return None

    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be an integer, 'all', 'none', or unset."
        ) from exc

    if n <= 0:
        raise ValueError(f"{env_name} must be positive.")

    return n


MAX_FILES = get_file_limit("SMAP_MAX_FILES", default=None)
MAX_DAYS = get_file_limit("SMAP_MAX_DAYS", default=None)
LATTICE_SCAN_FILES = get_file_limit("SMAP_LATTICE_SCAN_FILES", default=100) # in 100 files the lattice cover Iowa for sure.


# ============================================================
# 8. IEM point-to-area kriging settings
# ============================================================

IEM_PTA_VARIABLES = [
    "precip",
    "rh",
    "speed",
    "gust",
    "et",

    "soil04tn",
    "soil04t",
    "soil04tx",

    "soil12tn",
    "soil12t",
    "soil12tx",
    "soil12vwc",

    "soil24tn",
    "soil24t",
    "soil24tx",
    "soil24vwc",

    "soil50tn",
    "soil50t",
    "soil50tx",
    "soil50vwc",
]

IEM_MISSING_VALUE = -99.0
MIN_STATIONS_FOR_KRIGING = 8

# Number of sample points inside each SMAP polygon for point-to-area averaging.
# 1 = centroid only
# 5 = center/east/west/north/south
IEM_PTA_SAMPLE_MODE = "five_point"

# Kriging variogram model used by PyKrige.
IEM_VARIIOGRAM_MODEL = "spherical"

# ============================================================
# Full SMAP + IEM PTA daily data
# ============================================================

# Complete daily SMAP lattice files with:
# observed SMAP + original NA rows + IEM PTA auxiliary variables.
FULL_SMAP_IEM_DIR = GAP_FILLING_DIR / "03_full_smap_iem_data"

FULL_SMAP_IEM_AM_DIR = FULL_SMAP_IEM_DIR / "am"
FULL_SMAP_IEM_PM_DIR = FULL_SMAP_IEM_DIR / "pm"

FULL_SMAP_IEM_AM_COMPLETE_DIR = FULL_SMAP_IEM_AM_DIR / "complete"
FULL_SMAP_IEM_AM_OBSERVED_DIR = FULL_SMAP_IEM_AM_DIR / "observed"
FULL_SMAP_IEM_AM_MISSING_DIR = FULL_SMAP_IEM_AM_DIR / "missing"

FULL_SMAP_IEM_PM_COMPLETE_DIR = FULL_SMAP_IEM_PM_DIR / "complete"
FULL_SMAP_IEM_PM_OBSERVED_DIR = FULL_SMAP_IEM_PM_DIR / "observed"
FULL_SMAP_IEM_PM_MISSING_DIR = FULL_SMAP_IEM_PM_DIR / "missing"

FULL_SMAP_IEM_SUMMARY_PATH = FULL_SMAP_IEM_DIR / "full_smap_iem_build_summary.csv"

# ============================================================
# 9. SMAP helper functions
# ============================================================

def _limit_files(files: list[Path], max_files: int | None) -> list[Path]:
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files


def _list_files(folder: Path, suffix: str) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob(f"*{suffix}"))


def get_smap_dirs(pass_name: str) -> dict[str, Path]:
    pass_name = pass_name.lower()

    if pass_name == "am":
        return {
            "csv": SMAP_DETRENDED_AM_CSV_DIR,
            "rds": SMAP_DETRENDED_AM_RDS_DIR,
        }

    if pass_name == "pm":
        return {
            "csv": SMAP_DETRENDED_PM_CSV_DIR,
            "rds": SMAP_DETRENDED_PM_RDS_DIR,
        }

    raise ValueError("pass_name must be 'am' or 'pm'.")


def list_smap_files(
    pass_name: str,
    file_mode: str = "auto",
    max_files: int | None = MAX_FILES,
) -> list[Path]:
    file_mode = file_mode.lower()
    dirs = get_smap_dirs(pass_name)

    csv_files = _list_files(dirs["csv"], ".csv")
    rds_files = _list_files(dirs["rds"], ".rds")

    if file_mode == "auto":
        files = csv_files if len(csv_files) > 0 else rds_files
    elif file_mode == "csv":
        files = csv_files
    elif file_mode == "rds":
        files = rds_files
    else:
        raise ValueError("file_mode must be 'auto', 'csv', or 'rds'.")

    return _limit_files(files, max_files)


def get_preprocess_smap_dir(pass_name: str) -> Path:  # deprecated, kept as no-op stub
    raise NotImplementedError(
        "01_preprocess_smap was removed; nothing uses this folder anymore."
    )


def get_preprocess_final_dir(pass_name: str) -> Path:  # deprecated, kept as no-op stub
    raise NotImplementedError(
        "02_preprocess_final was removed; nothing uses this folder anymore."
    )


# ============================================================
# 10. Output helpers
# ============================================================

def get_iem_pta_daily_csv_path(date_yyyymmdd: str) -> Path:
    return IEM_PTA_DIR / f"iem_pta_smap_lattice_{date_yyyymmdd}.csv"


def ensure_output_dirs() -> None:
    folders = [
        GAP_FILLING_DIR,
        SUPPORT_DIR,
        SMAP_LATTICE_DIR,
        IEM_PTA_DIR,
        FULL_SMAP_IEM_DIR,
        FULL_SMAP_IEM_AM_DIR,
        FULL_SMAP_IEM_PM_DIR,
        FULL_SMAP_IEM_AM_COMPLETE_DIR,
        FULL_SMAP_IEM_AM_OBSERVED_DIR,
        FULL_SMAP_IEM_AM_MISSING_DIR,
        FULL_SMAP_IEM_PM_COMPLETE_DIR,
        FULL_SMAP_IEM_PM_OBSERVED_DIR,
        FULL_SMAP_IEM_PM_MISSING_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


ensure_output_dirs()


# ============================================================
# 11. Config summary
# ============================================================

def print_config_summary() -> None:
    print("\nSMAP gap-filling configuration")
    print("-" * 60)

    print(f"Project root:             {PROJECT_ROOT}")
    print(f"Data root:                {DATA_ROOT}")
    print(f"Raw dir:                  {RAW_DIR}")
    print(f"Processed dir:            {PROCESSED_DIR}")
    print()

    print("Inputs")
    print("-" * 60)
    print(f"Raw SMAP NC4 dir:         {RAW_SMAP_NC_DIR}")
    print(f"Township shapefile:       {TOWNSHIP_SHP_PATH}")
    print(f"IEM stations full:        {IEM_STATIONS_FULL_PATH}")
    print(f"IEM fallback path:        {IEM_STATIONS_FULL_FALLBACK_PATH}")
    print(f"SMAP lattice dir:         {SMAP_LATTICE_DIR}")
    print()

    print("Outputs")
    print("-" * 60)
    print(f"IEM PTA output:           {IEM_PTA_DIR}")
    print(f"Full SMAP+IEM output:     {FULL_SMAP_IEM_DIR}")
    print()

    print("Study settings")
    print("-" * 60)
    print(f"Train years:              {TRAIN_YEARS}")
    print(f"Validation years:         {VALIDATION_YEARS}")
    print(f"Test year:                {TEST_YEAR}")
    print(f"All years:                {ALL_YEARS}")
    print(f"CRS WGS84:                EPSG:{CRS_WGS84}")
    print(f"CRS EASE:                 EPSG:{CRS_EASE}")
    print(f"SMAP cell size m:         {SMAP_CELLSIZE_M}")
    print()

    print("Runtime limits")
    print("-" * 60)
    print(f"SMAP_MAX_FILES:           {MAX_FILES if MAX_FILES is not None else 'all'}")
    print(f"SMAP_MAX_DAYS:            {MAX_DAYS if MAX_DAYS is not None else 'all'}")
    print(f"SMAP_LATTICE_SCAN_FILES:  {LATTICE_SCAN_FILES if LATTICE_SCAN_FILES is not None else 'all'}")
    print()

    print("IEM PTA settings")
    print("-" * 60)
    print(f"IEM variables:            {IEM_PTA_VARIABLES}")
    print(f"IEM missing value:        {IEM_MISSING_VALUE}")
    print(f"Min stations:             {MIN_STATIONS_FOR_KRIGING}")
    print(f"Sample mode:              {IEM_PTA_SAMPLE_MODE}")
    print(f"Variogram model:          {IEM_VARIIOGRAM_MODEL}")
    print("-" * 60)

# ============================================================
# 12. Helper SMAP IEM full map
# ============================================================

def get_full_smap_iem_dirs(pass_name: str) -> dict[str, Path]:
    """Return complete/observed/missing output folders for AM or PM."""
    pass_name = pass_name.lower()

    if pass_name == "am":
        return {
            "complete": FULL_SMAP_IEM_AM_COMPLETE_DIR,
            "observed": FULL_SMAP_IEM_AM_OBSERVED_DIR,
            "missing": FULL_SMAP_IEM_AM_MISSING_DIR,
        }

    if pass_name == "pm":
        return {
            "complete": FULL_SMAP_IEM_PM_COMPLETE_DIR,
            "observed": FULL_SMAP_IEM_PM_OBSERVED_DIR,
            "missing": FULL_SMAP_IEM_PM_MISSING_DIR,
        }

    raise ValueError("pass_name must be 'am' or 'pm'.")

if __name__ == "__main__":
    print_config_summary()