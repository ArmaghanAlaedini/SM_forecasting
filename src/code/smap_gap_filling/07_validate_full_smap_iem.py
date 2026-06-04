#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timedelta
import importlib.util
import json
import re
import subprocess
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ============================================================
# 0. USER SETTINGS
# ============================================================

START_DATE = "2020-01-01"
N_DAYS = 10

PASSES_TO_CHECK = ["am", "pm"]

TARGET_COLUMN = "soil_moisture"

VALUE_TOL = 1e-12

WRITE_MISMATCH_FILES = True


# ============================================================
# 1. LOAD CONFIG
# ============================================================

def load_config():
    config_path = Path(__file__).resolve().with_name("00_config.py")

    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# 2. PATHS
# ============================================================

def get_gap_filling_dir() -> Path:
    if hasattr(cfg, "GAP_FILLING_DIR"):
        return Path(cfg.GAP_FILLING_DIR)
    return Path(cfg.PROCESSED_DIR) / "smap_gap_filling"


def get_full_smap_iem_dir() -> Path:
    if hasattr(cfg, "FULL_SMAP_IEM_DIR"):
        return Path(cfg.FULL_SMAP_IEM_DIR)
    return get_gap_filling_dir() / "03_full_smap_iem_data"


FULL_DIR = get_full_smap_iem_dir()
SUMMARY_PATH = FULL_DIR / "full_smap_iem_build_summary.csv"

VALIDATION_DIR = FULL_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_SUMMARY_PATH = VALIDATION_DIR / "validation_summary.csv"
MISMATCH_PATH = VALIDATION_DIR / "value_mismatches.csv"


# ============================================================
# 3. DATE HELPERS
# ============================================================

def make_dates(start_date: str, n_days: int) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_days)]


def date_to_yyyymmdd(date_string: str) -> str:
    return pd.to_datetime(date_string).strftime("%Y%m%d")


def extract_date_from_filename(path: Path) -> str | None:
    match = re.search(r"(20\d{6})", path.name)
    if match is None:
        return None
    return pd.to_datetime(match.group(1), format="%Y%m%d").strftime("%Y-%m-%d")


def completed_file(date_string: str, pass_name: str) -> Path:
    yyyymmdd = date_to_yyyymmdd(date_string)
    return FULL_DIR / pass_name / "complete" / f"smap_iem_{pass_name}_complete_{yyyymmdd}.csv"


def observed_file(date_string: str, pass_name: str) -> Path:
    yyyymmdd = date_to_yyyymmdd(date_string)
    return FULL_DIR / pass_name / "observed" / f"smap_iem_{pass_name}_observed_{yyyymmdd}.csv"


def missing_file(date_string: str, pass_name: str) -> Path:
    yyyymmdd = date_to_yyyymmdd(date_string)
    return FULL_DIR / pass_name / "missing" / f"smap_iem_{pass_name}_missing_{yyyymmdd}.csv"


# ============================================================
# 4. READ RDS THROUGH R
# ============================================================

def read_rds_with_rscript(path: Path) -> pd.DataFrame:
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

        return pd.read_csv(tmp_csv, low_memory=False)


# ============================================================
# 5. FIND RDS FILES
# ============================================================

def list_rds_files(pass_name: str) -> list[Path]:
    if hasattr(cfg, "list_smap_files"):
        try:
            files = cfg.list_smap_files(
                pass_name=pass_name,
                file_mode="auto",
                max_files=None,
            )
        except TypeError:
            files = cfg.list_smap_files(pass_name=pass_name)

        files = [Path(p) for p in files if str(p).endswith(".rds")]
        if files:
            return sorted(files)

    fallback = Path(cfg.PROCESSED_DIR) / "smap_detrended" / pass_name / "rds"
    return sorted(fallback.glob("*.rds"))


def find_rds_for_date(pass_name: str, date_string: str, summary: pd.DataFrame) -> Path:
    rows = summary[
        (summary["date"].astype(str) == date_string)
        & (summary["pass"].astype(str).str.lower() == pass_name.lower())
    ]

    rds_files = list_rds_files(pass_name)

    if rows.shape[0] > 0 and "file_name" in rows.columns:
        file_name = str(rows.iloc[0]["file_name"])
        exact = [p for p in rds_files if p.name == file_name]
        if exact:
            return exact[0]

    matches = [p for p in rds_files if extract_date_from_filename(p) == date_string]

    if not matches:
        raise FileNotFoundError(f"No RDS file found for {pass_name} {date_string}")

    return sorted(matches)[-1]


# ============================================================
# 6. MATCH RDS OBSERVATIONS TO COMPLETE LATTICE
# ============================================================

def find_column(columns, candidates) -> str | None:
    columns = list(columns)
    lower_map = {str(c).lower(): c for c in columns}

    for c in candidates:
        if c in columns:
            return c

    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    return None


def match_rds_to_complete_lattice(rds: pd.DataFrame, complete: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x_col = find_column(rds.columns, ["x", "X", "centroid_x", "x_m"])
    y_col = find_column(rds.columns, ["y", "Y", "centroid_y", "y_m"])
    target_col = find_column(rds.columns, [TARGET_COLUMN, "Soil_Moisture", "smap", "SMAP", "sm"])

    if x_col is None or y_col is None:
        raise ValueError("RDS file does not have x/y columns.")

    if target_col is None:
        raise ValueError(f"RDS file does not have target column {TARGET_COLUMN}.")

    rds = rds.copy()
    rds[x_col] = pd.to_numeric(rds[x_col], errors="coerce")
    rds[y_col] = pd.to_numeric(rds[y_col], errors="coerce")
    rds[target_col] = pd.to_numeric(rds[target_col], errors="coerce")

    valid = rds[
        rds[x_col].notna()
        & rds[y_col].notna()
        & rds[target_col].notna()
    ].copy()

    if valid.empty:
        return pd.DataFrame(columns=["smap_pixel_key", TARGET_COLUMN, "__distance_m"]), {
            "rds_target_column": target_col,
            "rds_raw_rows": len(rds),
            "rds_valid_observed_rows": 0,
            "rds_matched_rows": 0,
            "rds_unmatched_rows": 0,
            "max_distance_all_valid_m": np.nan,
            "max_distance_matched_m": np.nan,
        }

    lattice = complete[["smap_pixel_key", "x", "y"]].copy()
    lattice["x"] = pd.to_numeric(lattice["x"], errors="coerce")
    lattice["y"] = pd.to_numeric(lattice["y"], errors="coerce")

    tree = cKDTree(lattice[["x", "y"]].to_numpy())

    distances, indices = tree.query(valid[[x_col, y_col]].to_numpy(), k=1)

    valid["__distance_m"] = distances
    valid["smap_pixel_key"] = lattice.iloc[indices]["smap_pixel_key"].to_numpy()

    max_allowed = getattr(cfg, "SMAP_CELL_SIZE_M", 9024.31) * 0.55

    matched = valid[valid["__distance_m"] <= max_allowed].copy()
    unmatched_rows = len(valid) - len(matched)

    if not matched.empty:
        matched = (
            matched.sort_values("__distance_m")
            .drop_duplicates("smap_pixel_key", keep="first")
            .reset_index(drop=True)
        )

    out = matched[["smap_pixel_key", target_col, "__distance_m"]].copy()
    out = out.rename(columns={target_col: TARGET_COLUMN})

    info = {
        "rds_target_column": target_col,
        "rds_raw_rows": len(rds),
        "rds_valid_observed_rows": len(valid),
        "rds_matched_rows": len(out),
        "rds_unmatched_rows": unmatched_rows,
        "max_distance_all_valid_m": float(valid["__distance_m"].max()) if len(valid) else np.nan,
        "max_distance_matched_m": float(out["__distance_m"].max()) if len(out) else np.nan,
    }

    return out, info


# ============================================================
# 7. VALIDATE ONE DATE/PASS
# ============================================================

def validate_one(date_string: str, pass_name: str, summary: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    c_path = completed_file(date_string, pass_name)
    o_path = observed_file(date_string, pass_name)
    m_path = missing_file(date_string, pass_name)

    if not c_path.exists():
        raise FileNotFoundError(c_path)

    complete = pd.read_csv(c_path, low_memory=False)

    required = ["smap_pixel_key", "smap_status", TARGET_COLUMN, "x", "y"]
    missing_required = [c for c in required if c not in complete.columns]

    if missing_required:
        raise ValueError(f"Complete file missing required columns: {missing_required}")

    rds_path = find_rds_for_date(pass_name, date_string, summary)
    rds = read_rds_with_rscript(rds_path)

    rds_matched, rds_info = match_rds_to_complete_lattice(rds, complete)

    complete_obs = complete[complete[TARGET_COLUMN].notna()].copy()
    complete_miss = complete[complete[TARGET_COLUMN].isna()].copy()

    merged = rds_matched.merge(
        complete[["smap_pixel_key", TARGET_COLUMN, "smap_status"]],
        on="smap_pixel_key",
        how="outer",
        suffixes=("_rds", "_complete"),
        indicator=True,
    )

    both = merged[merged["_merge"] == "both"].copy()

    if len(both) > 0:
        both["abs_diff"] = (
            pd.to_numeric(both[f"{TARGET_COLUMN}_rds"], errors="coerce")
            - pd.to_numeric(both[f"{TARGET_COLUMN}_complete"], errors="coerce")
        ).abs()
    else:
        both["abs_diff"] = pd.Series(dtype=float)

    n_value_mismatch = int((both["abs_diff"] > VALUE_TOL).sum()) if len(both) else 0
    max_abs_diff = float(both["abs_diff"].max()) if len(both) else np.nan

    n_rds_not_in_complete = int((merged["_merge"] == "left_only").sum())
    n_complete_not_in_rds = int((merged["_merge"] == "right_only").sum())

    status_counts = complete["smap_status"].value_counts(dropna=False).to_dict()

    partition_ok = (
        len(complete) == complete["smap_pixel_key"].nunique()
        and len(complete_obs) + len(complete_miss) == len(complete)
        and status_counts.get("observed", 0) == len(complete_obs)
        and status_counts.get("original_NA", 0) == len(complete_miss)
    )

    observed_subset_ok = np.nan
    missing_subset_ok = np.nan

    if o_path.exists():
        obs_file = pd.read_csv(o_path, low_memory=False)
        observed_subset_ok = (
            set(obs_file["smap_pixel_key"]) == set(complete_obs["smap_pixel_key"])
            and len(obs_file) == len(complete_obs)
        )

    if m_path.exists():
        miss_file = pd.read_csv(m_path, low_memory=False)
        missing_subset_ok = (
            set(miss_file["smap_pixel_key"]) == set(complete_miss["smap_pixel_key"])
            and len(miss_file) == len(complete_miss)
        )

    pta_cols = [
        c for c in complete.columns
        if c.endswith("_pta") and not c.endswith("_pta_var")
    ]

    pta_min_nonmissing = (
        int(min(complete[c].notna().sum() for c in pta_cols))
        if pta_cols else 0
    )

    row = {
        "date": date_string,
        "pass": pass_name,
        "complete_file": str(c_path),
        "rds_file": str(rds_path),
        "n_complete_rows": len(complete),
        "n_unique_smap_pixel_key": complete["smap_pixel_key"].nunique(),
        "n_complete_observed": len(complete_obs),
        "n_complete_missing": len(complete_miss),
        "partition_ok": partition_ok,
        "observed_subset_ok": observed_subset_ok,
        "missing_subset_ok": missing_subset_ok,
        "n_rds_matched_rows": rds_info["rds_matched_rows"],
        "n_rds_unmatched_rows": rds_info["rds_unmatched_rows"],
        "n_rds_not_in_complete": n_rds_not_in_complete,
        "n_complete_not_in_rds": n_complete_not_in_rds,
        "n_value_mismatch": n_value_mismatch,
        "max_abs_diff": max_abs_diff,
        "n_pta_value_columns": len(pta_cols),
        "pta_min_nonmissing": pta_min_nonmissing,
        **rds_info,
    }

    mismatches = merged[
        (merged["_merge"] != "both")
        | (merged.get("abs_diff", pd.Series(dtype=float)) > VALUE_TOL)
    ].copy()

    mismatches.insert(0, "date", date_string)
    mismatches.insert(1, "pass", pass_name)

    return row, mismatches


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print("\nValidating full SMAP + IEM completed files")
    print("-" * 80)
    print(f"Start date: {START_DATE}")
    print(f"N days:     {N_DAYS}")
    print(f"Passes:     {PASSES_TO_CHECK}")
    print(f"Full dir:   {FULL_DIR}")
    print("-" * 80)

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary file not found:\n{SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH, low_memory=False)

    rows = []
    all_mismatches = []

    for date_string in make_dates(START_DATE, N_DAYS):
        for pass_name in PASSES_TO_CHECK:
            print(f"Checking {date_string} {pass_name.upper()} ...")

            try:
                row, mismatches = validate_one(date_string, pass_name, summary)
                row["validation_status"] = "ok"

                if len(mismatches) > 0:
                    row["validation_status"] = "problem"

                rows.append(row)

                if WRITE_MISMATCH_FILES and len(mismatches) > 0:
                    all_mismatches.append(mismatches)

            except Exception as exc:
                rows.append({
                    "date": date_string,
                    "pass": pass_name,
                    "validation_status": "failed",
                    "error": str(exc),
                })
                print(f"  FAILED: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(VALIDATION_SUMMARY_PATH, index=False)

    if WRITE_MISMATCH_FILES and all_mismatches:
        pd.concat(all_mismatches, ignore_index=True).to_csv(MISMATCH_PATH, index=False)

    print("\nDone.")
    print(f"Validation summary saved to:\n{VALIDATION_SUMMARY_PATH}")

    if WRITE_MISMATCH_FILES and all_mismatches:
        print(f"Mismatches saved to:\n{MISMATCH_PATH}")
    else:
        print("No mismatch file written.")

    print("\nValidation status counts:")
    print(out["validation_status"].value_counts(dropna=False))

    print("\nKey columns to inspect:")
    keep = [
        "date",
        "pass",
        "validation_status",
        "n_complete_rows",
        "n_complete_observed",
        "n_rds_matched_rows",
        "n_rds_unmatched_rows",
        "n_complete_not_in_rds",
        "n_rds_not_in_complete",
        "n_value_mismatch",
        "max_abs_diff",
        "partition_ok",
        "observed_subset_ok",
        "missing_subset_ok",
    ]
    keep = [c for c in keep if c in out.columns]
    print(out[keep].head(30))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()