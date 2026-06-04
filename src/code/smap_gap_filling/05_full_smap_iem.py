from pathlib import Path
import importlib.util
import json
import re
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree


# ============================================================
# 0. Load config
# ============================================================

def load_config():
    """
    Load 00_config.py from the same folder as this script.
    """
    config_path = Path(__file__).resolve().with_name("00_config.py")

    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# 1. Settings
# ============================================================

# This is the actual SMAP soil moisture column in your detrended RDS files.
SMAP_TARGET_COLUMN = "soil_moisture"

# Main SMAP soil moisture column candidates, in case capitalization changes.
SMAP_VALUE_CANDIDATES = [
    "soil_moisture",
    "Soil_Moisture",
    "SMAP",
    "smap",
    "sm",
]

# Projected coordinate columns used only for matching RDS observations to the lattice.
X_CANDIDATES = ["x", "X", "centroid_x", "x_m"]
Y_CANDIDATES = ["y", "Y", "centroid_y", "y_m"]

PASSES = getattr(cfg, "PASSES", ["am", "pm"])

# SMAP EASE-grid cell size is about 9024 m.
# Observed RDS rows farther than about half a cell from the nearest lattice
# centroid are treated as outside the final Iowa lattice and dropped.
MAX_NEAREST_DISTANCE_M = getattr(cfg, "SMAP_CELL_SIZE_M", 9024.31) * 0.55

# RDS columns to drop from the output because the complete file already has
# authoritative date/pass/lattice coordinates/geometry.
RDS_DROP_COLUMNS_FOR_OUTPUT = {
    "date",
    "date_tag",
    "pass",
    "lon",
    "lat",
    "x",
    "y",
    "geometry",
    "geometry_wkt",
    "grid_row",
    "grid_col",
    "cell_area_m2",
    "smap_pixel_key",
    "nearest_lattice_distance_m",
}

# Lattice columns to keep in the final complete/observed/missing outputs.
LATTICE_KEEP_COLUMNS = [
    "smap_pixel_key",
    "grid_row",
    "grid_col",
    "lon",
    "lat",
    "x",
    "y",
    "cell_area_m2",
    "geometry_wkt",
]


# ============================================================
# 2. Output paths
# ============================================================

def get_gap_filling_dir() -> Path:
    if hasattr(cfg, "GAP_FILLING_DIR"):
        return cfg.GAP_FILLING_DIR
    return cfg.PROCESSED_DIR / "smap_gap_filling"


def get_full_smap_iem_root() -> Path:
    """
    Output folder for full daily SMAP + IEM files.
    """
    if hasattr(cfg, "FULL_SMAP_IEM_DIR"):
        return cfg.FULL_SMAP_IEM_DIR

    return get_gap_filling_dir() / "03_full_smap_iem_data"


def get_iem_pta_dir() -> Path:
    """
    Folder containing daily IEM point-to-area kriged files.
    """
    for name in ["IEM_PTA_DIR", "IEM_POINT_TO_AREA_DIR", "IEM_PTA_DAILY_DIR"]:
        if hasattr(cfg, name):
            return getattr(cfg, name)

    return get_gap_filling_dir() / "iem_point_to_area"


FULL_SMAP_IEM_DIR = get_full_smap_iem_root()
IEM_PTA_DIR = get_iem_pta_dir()
SUMMARY_PATH = FULL_SMAP_IEM_DIR / "full_smap_iem_build_summary.csv"


def get_output_dirs(pass_name: str) -> dict[str, Path]:
    """
    Return complete/observed/missing folders for one pass.
    """
    pass_name = pass_name.lower()
    base = FULL_SMAP_IEM_DIR / pass_name

    return {
        "complete": base / "complete",
        "observed": base / "observed",
        "missing": base / "missing",
    }


def ensure_dirs() -> None:
    """
    Create output folders.
    """
    FULL_SMAP_IEM_DIR.mkdir(parents=True, exist_ok=True)

    for pass_name in PASSES:
        for folder in get_output_dirs(pass_name).values():
            folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Helper functions
# ============================================================

def find_column(columns, candidates) -> str | None:
    """
    Find a column using exact or case-insensitive matching.
    """
    columns = list(columns)
    lower_map = {str(c).lower(): c for c in columns}

    for c in candidates:
        if c in columns:
            return c

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def extract_date_from_filename(path: Path) -> str:
    """
    Extract YYYYMMDD from filename and return YYYY-MM-DD.
    """
    match = re.search(r"(20\d{6})", path.name)

    if match is None:
        raise ValueError(f"Could not extract date from filename: {path.name}")

    return pd.to_datetime(match.group(1), format="%Y%m%d").strftime("%Y-%m-%d")


def date_to_yyyymmdd(date_string: str) -> str:
    """
    Convert YYYY-MM-DD to YYYYMMDD.
    """
    return pd.to_datetime(date_string).strftime("%Y%m%d")


def retained_rds_columns(columns) -> list[str]:
    """
    Keep actual useful RDS variables with their original names.

    Dropped from RDS output:
    - date/pass duplicates
    - RDS lon/lat/x/y because lattice coordinates are authoritative
    - RDS geometry because lattice polygon is authoritative
    """
    out = []

    for col in columns:
        col_str = str(col)

        if col_str in RDS_DROP_COLUMNS_FOR_OUTPUT:
            continue

        if col_str not in out:
            out.append(col_str)

    return out


def make_empty_smap_out(raw_columns) -> pd.DataFrame:
    """
    Create an empty observed-SMAP table with the same retained RDS columns.
    This allows empty PM/date retrievals to still produce a full lattice file.
    """
    data = {
        "smap_pixel_key": pd.Series(dtype="object"),
    }

    for col in retained_rds_columns(raw_columns):
        dtype = "float64" if col == SMAP_TARGET_COLUMN else "object"
        data[col] = pd.Series(dtype=dtype)

    if SMAP_TARGET_COLUMN not in data:
        data[SMAP_TARGET_COLUMN] = pd.Series(dtype="float64")

    return pd.DataFrame(data)


def to_numeric_if_possible(series: pd.Series) -> pd.Series:
    """
    Convert a column to numeric only if conversion is useful.
    """
    converted = pd.to_numeric(series, errors="coerce")
    n_good = converted.notna().sum()

    if n_good == 0:
        return series

    return converted


# ============================================================
# 4. Read SMAP RDS / CSV files
# ============================================================

def read_rds_with_rscript(path: Path) -> pd.DataFrame:
    """
    Read one RDS file using Rscript.

    pyreadr could not read your RDS files reliably, so this uses R's readRDS().
    List-like columns are flattened to text. Geometry is later dropped from
    RDS output because the full lattice geometry is used instead.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_csv = Path(tmpdir) / "rds_export.csv"

        r_code = f"""
        x <- readRDS({json.dumps(str(path))})

        if (!is.data.frame(x)) {{
          stop("RDS object is not a data.frame or sf-like data.frame")
        }}

        x <- as.data.frame(x)

        flatten_one <- function(z) {{
          if (length(z) == 0) return(NA_character_)
          paste(as.character(unlist(z)), collapse = ";")
        }}

        for (nm in names(x)) {{
          if (is.list(x[[nm]])) {{
            x[[nm]] <- vapply(x[[nm]], flatten_one, character(1))
          }}
        }}

        write.csv(x, {json.dumps(str(tmp_csv))}, row.names = FALSE, na = "")
        """

        result = subprocess.run(
            ["Rscript", "--vanilla", "-e", r_code],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Rscript failed reading {path}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        return pd.read_csv(tmp_csv)


def read_smap_file(path: Path) -> pd.DataFrame:
    """
    Read one daily SMAP detrended file.
    """
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".rds":
        return read_rds_with_rscript(path)

    raise ValueError(f"Unsupported SMAP file type: {path}")


# ============================================================
# 5. Load full SMAP lattice
# ============================================================

def load_lattice() -> pd.DataFrame:
    """
    Load the full SMAP lattice.

    This is the base table. Every complete daily file will have exactly
    these lattice pixels.
    """
    lattice_path = cfg.SMAP_LATTICE_DIR / "smap_lattice_iowa.parquet"

    if not lattice_path.exists():
        raise FileNotFoundError(f"SMAP lattice not found:\n{lattice_path}")

    lattice = gpd.read_parquet(lattice_path)

    required = ["smap_pixel_key", "lon", "lat", "x", "y", "geometry"]
    missing = [c for c in required if c not in lattice.columns]

    if missing:
        raise ValueError(f"Lattice is missing required columns: {missing}")

    lattice = lattice.copy()

    if "geometry_wkt" not in lattice.columns:
        lattice["geometry_wkt"] = lattice.geometry.to_wkt()

    keep_cols = [c for c in LATTICE_KEEP_COLUMNS if c in lattice.columns]
    out = pd.DataFrame(lattice[keep_cols]).drop_duplicates("smap_pixel_key")

    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    if out["x"].isna().any() or out["y"].isna().any():
        raise ValueError("Lattice has missing x/y coordinates.")

    return out.reset_index(drop=True)


def build_lattice_tree(lattice: pd.DataFrame):
    """
    Build nearest-neighbor search tree from lattice centroid x/y coordinates.
    """
    coords = lattice[["x", "y"]].to_numpy()
    return cKDTree(coords)


# ============================================================
# 6. Standardize one daily SMAP observed subset
# ============================================================

def standardize_smap_daily(
    raw: pd.DataFrame,
    path: Path,
    lattice: pd.DataFrame,
    lattice_tree,
) -> tuple[pd.DataFrame, dict]:
    """
    Convert one daily SMAP RDS/CSV file into an observed-SMAP table.

    Important:
    - The RDS file is an observed subset, not the full lattice.
    - The full lattice is added later.
    - Missing lattice pixels become NA after the left join.
    - RDS variables are kept with their original names.
    - RDS coordinate/geometry duplicates are not kept in the final output.
    """
    df = raw.copy()

    smap_col = find_column(df.columns, SMAP_VALUE_CANDIDATES)
    x_col = find_column(df.columns, X_CANDIDATES)
    y_col = find_column(df.columns, Y_CANDIDATES)

    if smap_col is None:
        raise ValueError(
            f"Could not find SMAP soil moisture column in {path.name}. "
            f"Columns: {list(df.columns)}"
        )

    if smap_col != SMAP_TARGET_COLUMN:
        df[SMAP_TARGET_COLUMN] = pd.to_numeric(df[smap_col], errors="coerce")
        smap_col = SMAP_TARGET_COLUMN

    empty_out = make_empty_smap_out(df.columns)

    if x_col is None or y_col is None:
        info = {
            "file_name": path.name,
            "raw_rows": len(raw),
            "valid_observed_rows": 0,
            "matched_rows": 0,
            "unmatched_rows": 0,
            "smap_value_column": smap_col,
            "max_distance_all_observed_m": pd.NA,
            "max_distance_matched_m": pd.NA,
            "mean_distance_matched_m": pd.NA,
            "note": "No x/y columns found; complete file saved with soil_moisture=NA.",
        }
        return empty_out, info

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[smap_col] = pd.to_numeric(df[smap_col], errors="coerce")

    observed = df.loc[
        df[x_col].notna()
        & df[y_col].notna()
        & df[smap_col].notna()
    ].copy()

    if observed.empty:
        info = {
            "file_name": path.name,
            "raw_rows": len(raw),
            "valid_observed_rows": 0,
            "matched_rows": 0,
            "unmatched_rows": 0,
            "smap_value_column": smap_col,
            "max_distance_all_observed_m": pd.NA,
            "max_distance_matched_m": pd.NA,
            "mean_distance_matched_m": pd.NA,
            "note": "No valid observed SMAP rows; complete file saved with soil_moisture=NA.",
        }
        return empty_out, info

    coords = observed[[x_col, y_col]].to_numpy()
    distances, indices = lattice_tree.query(coords, k=1)

    observed["__nearest_distance_m"] = distances
    observed["smap_pixel_key"] = lattice.iloc[indices]["smap_pixel_key"].to_numpy()

    matched = observed.loc[
        observed["__nearest_distance_m"] <= MAX_NEAREST_DISTANCE_M
    ].copy()

    unmatched_rows = len(observed) - len(matched)

    if matched.empty:
        info = {
            "file_name": path.name,
            "raw_rows": len(raw),
            "valid_observed_rows": len(observed),
            "matched_rows": 0,
            "unmatched_rows": unmatched_rows,
            "smap_value_column": smap_col,
            "max_distance_all_observed_m": float(observed["__nearest_distance_m"].max()),
            "max_distance_matched_m": pd.NA,
            "mean_distance_matched_m": pd.NA,
            "note": "Observed rows existed, but none matched the final Iowa lattice.",
        }
        return empty_out, info

    # If multiple observed rows map to the same lattice pixel, keep the closest.
    matched = (
        matched.sort_values("__nearest_distance_m")
        .drop_duplicates(subset=["smap_pixel_key"], keep="first")
        .reset_index(drop=True)
    )

    keep_rds_cols = retained_rds_columns(matched.columns)

    out = matched[["smap_pixel_key"] + keep_rds_cols].copy()

    # Convert numeric-looking columns back to numeric where appropriate.
    for col in out.columns:
        if col == "smap_pixel_key":
            continue
        out[col] = to_numeric_if_possible(out[col])

    info = {
        "file_name": path.name,
        "raw_rows": len(raw),
        "valid_observed_rows": len(observed),
        "matched_rows": len(out),
        "unmatched_rows": unmatched_rows,
        "smap_value_column": smap_col,
        "max_distance_all_observed_m": float(observed["__nearest_distance_m"].max()),
        "max_distance_matched_m": float(matched["__nearest_distance_m"].max()),
        "mean_distance_matched_m": float(matched["__nearest_distance_m"].mean()),
        "note": f"RDS treated as observed subset. Dropped {unmatched_rows} rows outside final lattice.",
    }

    return out, info


# ============================================================
# 7. Load daily IEM PTA file
# ============================================================

def load_pta_for_date(date_string: str) -> pd.DataFrame:
    """
    Load one daily IEM point-to-area file.

    Keeps only actual PTA output columns:
    - *_pta
    - *_pta_var
    - *_n_samples

    It does not keep PTA duplicate date/coordinate/geometry columns.
    """
    yyyymmdd = date_to_yyyymmdd(date_string)
    pta_path = IEM_PTA_DIR / f"iem_pta_smap_lattice_{yyyymmdd}.csv"

    if not pta_path.exists():
        raise FileNotFoundError(f"IEM PTA file not found:\n{pta_path}")

    pta = pd.read_csv(pta_path)

    if "smap_pixel_key" not in pta.columns:
        raise ValueError(f"PTA file has no smap_pixel_key column:\n{pta_path}")

    keep_cols = [
        c for c in pta.columns
        if c == "smap_pixel_key"
        or c.endswith("_pta")
        or c.endswith("_pta_var")
        or c.endswith("_n_samples")
    ]

    if len(keep_cols) == 1:
        raise ValueError(
            f"No PTA variable columns found in {pta_path}. "
            "Expected columns ending in _pta, _pta_var, or _n_samples."
        )

    out = pta[keep_cols].drop_duplicates("smap_pixel_key").reset_index(drop=True)

    return out


# ============================================================
# 8. Build complete / observed / missing files
# ============================================================

def build_complete(
    lattice: pd.DataFrame,
    smap_daily: pd.DataFrame,
    pta_daily: pd.DataFrame,
    date_string: str,
    pass_name: str,
) -> pd.DataFrame:
    """
    Build one complete daily file.

    Base:
        full SMAP lattice

    Attached:
        actual IEM PTA columns
        actual retained RDS variables

    No duplicate prefixed SMAP columns are created.
    """
    full = lattice.copy()

    full = full.merge(
        pta_daily,
        on="smap_pixel_key",
        how="left",
    )

    full = full.merge(
        smap_daily,
        on="smap_pixel_key",
        how="left",
    )

    if SMAP_TARGET_COLUMN not in full.columns:
        full[SMAP_TARGET_COLUMN] = np.nan

    full.insert(0, "date", date_string)
    full.insert(1, "pass", pass_name.lower())

    full["smap_status"] = np.where(
        full[SMAP_TARGET_COLUMN].notna(),
        "observed",
        "original_NA",
    )

    first_cols = [
        "date",
        "pass",
        "smap_pixel_key",
        "smap_status",
        SMAP_TARGET_COLUMN,
        "source_file",
        "lon",
        "lat",
        "x",
        "y",
        "grid_row",
        "grid_col",
        "cell_area_m2",
        "geometry_wkt",
    ]

    first_cols = [c for c in first_cols if c in full.columns]
    other_cols = [c for c in full.columns if c not in first_cols]

    return full[first_cols + other_cols]


def save_outputs(full: pd.DataFrame, date_string: str, pass_name: str) -> dict:
    """
    Save complete, observed, and missing files for one date-pass.
    """
    pass_name = pass_name.lower()
    yyyymmdd = date_to_yyyymmdd(date_string)

    dirs = get_output_dirs(pass_name)

    for folder in dirs.values():
        folder.mkdir(parents=True, exist_ok=True)

    complete_path = dirs["complete"] / f"smap_iem_{pass_name}_complete_{yyyymmdd}.csv"
    observed_path = dirs["observed"] / f"smap_iem_{pass_name}_observed_{yyyymmdd}.csv"
    missing_path = dirs["missing"] / f"smap_iem_{pass_name}_missing_{yyyymmdd}.csv"

    observed = full.loc[full["smap_status"] == "observed"].copy()
    missing = full.loc[full["smap_status"] == "original_NA"].copy()

    full.to_csv(complete_path, index=False)
    observed.to_csv(observed_path, index=False)
    missing.to_csv(missing_path, index=False)

    pta_cols = [
        c for c in full.columns
        if c.endswith("_pta") or c.endswith("_pta_var") or c.endswith("_n_samples")
    ]

    rds_cols = [
        c for c in full.columns
        if c not in {
            "date",
            "pass",
            "smap_pixel_key",
            "smap_status",
            "lon",
            "lat",
            "x",
            "y",
            "grid_row",
            "grid_col",
            "cell_area_m2",
            "geometry_wkt",
        }
        and c not in pta_cols
    ]

    return {
        "complete_file": str(complete_path),
        "observed_file": str(observed_path),
        "missing_file": str(missing_path),
        "n_lattice_pixels": len(full),
        "n_observed": len(observed),
        "n_missing": len(missing),
        "n_rds_columns_kept": len(rds_cols),
        "n_pta_columns_kept": len(pta_cols),
    }


# ============================================================
# 9. File listing
# ============================================================

def list_smap_files(pass_name: str) -> list[Path]:
    """
    List SMAP detrended files for AM or PM using config helper.
    """
    if not hasattr(cfg, "list_smap_files"):
        raise AttributeError("00_config.py must define list_smap_files().")

    files = cfg.list_smap_files(
        pass_name=pass_name,
        file_mode="auto",
        max_files=getattr(cfg, "MAX_FILES", None),
    )

    files = sorted(files)

    if not files:
        raise FileNotFoundError(f"No SMAP files found for {pass_name}")

    return files


# ============================================================
# 10. Process one pass
# ============================================================

def process_pass(lattice: pd.DataFrame, lattice_tree, pass_name: str) -> list[dict]:
    """
    Process all daily files for one pass: AM or PM.
    """
    pass_name = pass_name.lower()
    files = list_smap_files(pass_name)

    summaries = []

    print(f"\nProcessing {pass_name.upper()} files: {len(files)}")
    print("-" * 80)

    for i, path in enumerate(files, start=1):
        date_string = pd.NA

        try:
            date_string = extract_date_from_filename(path)

            print(f"[{i}/{len(files)}] {pass_name.upper()} {date_string} | {path.name}")

            raw = read_smap_file(path)

            smap_daily, smap_info = standardize_smap_daily(
                raw=raw,
                path=path,
                lattice=lattice,
                lattice_tree=lattice_tree,
            )

            pta_daily = load_pta_for_date(date_string)

            full = build_complete(
                lattice=lattice,
                smap_daily=smap_daily,
                pta_daily=pta_daily,
                date_string=date_string,
                pass_name=pass_name,
            )

            save_info = save_outputs(full, date_string, pass_name)

            summary = {
                "date": date_string,
                "pass": pass_name,
                "status": "ok",
                "message": "",
                **smap_info,
                **save_info,
            }

            summaries.append(summary)

            max_dist = smap_info.get("max_distance_matched_m", pd.NA)
            max_dist_text = "NA" if pd.isna(max_dist) else f"{float(max_dist):.2f} m"

            print(
                f"    observed={save_info['n_observed']} | "
                f"missing={save_info['n_missing']} | "
                f"unmatched={smap_info.get('unmatched_rows', 0)} | "
                f"max_matched_dist={max_dist_text}"
            )

        except Exception as exc:
            print(f"    FAILED: {path.name} | {exc}")

            summaries.append({
                "date": date_string,
                "pass": pass_name,
                "status": "failed",
                "message": f"{path.name}: {exc}",
                "file_name": path.name,
                "raw_rows": pd.NA,
                "valid_observed_rows": pd.NA,
                "matched_rows": pd.NA,
                "unmatched_rows": pd.NA,
                "smap_value_column": pd.NA,
                "max_distance_all_observed_m": pd.NA,
                "max_distance_matched_m": pd.NA,
                "mean_distance_matched_m": pd.NA,
                "note": "",
                "complete_file": "",
                "observed_file": "",
                "missing_file": "",
                "n_lattice_pixels": len(lattice),
                "n_observed": pd.NA,
                "n_missing": pd.NA,
                "n_rds_columns_kept": pd.NA,
                "n_pta_columns_kept": pd.NA,
            })

    return summaries


# ============================================================
# 11. Main
# ============================================================

def main() -> None:
    """
    Build daily complete/observed/missing SMAP + IEM files.

    Output structure:
        03_full_smap_iem_data/
            am/
                complete/
                observed/
                missing/
            pm/
                complete/
                observed/
                missing/

    Target column:
        soil_moisture

    Status column:
        smap_status = observed or original_NA
    """
    if hasattr(cfg, "print_config_summary"):
        cfg.print_config_summary()

    ensure_dirs()

    print("\nBuilding full SMAP + IEM PTA daily files")
    print("=" * 80)

    lattice = load_lattice()
    lattice_tree = build_lattice_tree(lattice)

    print(f"Lattice pixels: {len(lattice)}")
    print(f"IEM PTA folder: {IEM_PTA_DIR}")
    print(f"Output folder:  {FULL_SMAP_IEM_DIR}")
    print(f"Summary path:   {SUMMARY_PATH}")

    all_summaries = []

    for pass_name in PASSES:
        all_summaries.extend(process_pass(lattice, lattice_tree, pass_name))

    summary = pd.DataFrame(all_summaries)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nDone.")
    print(f"Summary saved to: {SUMMARY_PATH}")

    print("\nStatus counts:")
    print(summary["status"].value_counts(dropna=False))

    if "n_observed" in summary.columns:
        print("\nObserved count summary:")
        print(summary.groupby("pass")["n_observed"].describe())


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()