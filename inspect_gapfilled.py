#!/usr/bin/env python3
"""
Inspect ONE gap-filled / complete SMAP daily file and report:
  - total pixels (rows), whether observed or originally NA / gap-filled
  - lon/lat range (the full Iowa lattice extent, before township clipping)
  - observed vs originally-missing breakdown, if a flag column exists

Usage:
    python inspect_gapfilled.py path/to/one_file.csv
    python inspect_gapfilled.py                 # uses the DEFAULT_GLOB below

Works on .csv or .parquet. Column names are auto-detected; the full column
list is printed so you can see exactly what is in the file.
"""
from pathlib import Path
import sys
import glob
import pandas as pd

# Default: first "complete" AM file (has all pixels, observed + originally NA).
DEFAULT_GLOB = (
    "src/data/processed/smap_gap_filling/"
    "03_full_smap_iem_data/am/complete/*.csv"
)

LAT_NAMES = ["lat", "latitude", "center_lat", "y_lat"]
LON_NAMES = ["lon", "long", "longitude", "center_lon", "x_lon"]
KEY_NAMES = ["smap_pixel_key", "pixel_key", "cell_id"]
FLAG_NAMES = ["status", "origin", "source", "is_observed", "observed",
              "is_missing", "missing", "na_flag", "filled", "gapfilled",
              "sm_origin", "obs_flag"]
SM_NAMES = ["soil_moisture", "sm", "sm_observed", "smap_sm", "soil_moisture_obs"]


def pick(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in low:
            return low[n]
    return None


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        hits = sorted(glob.glob(DEFAULT_GLOB))
        if not hits:
            print("No file given and nothing matched the default location:")
            print(f"  {DEFAULT_GLOB}")
            print("Pass a file path explicitly, e.g.:")
            print("  python inspect_gapfilled.py src/.../one_file.csv")
            return
        path = hits[0]

    print(f"File: {path}")
    df = load(path)
    print(f"Rows (total SMAP pixels, NA or observed): {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print("-" * 60)

    key = pick(df.columns, KEY_NAMES)
    if key:
        print(f"Unique pixels ({key}): {df[key].nunique()}")

    lat = pick(df.columns, LAT_NAMES)
    lon = pick(df.columns, LON_NAMES)
    if lat and lon:
        print(f"Lat range: {df[lat].min():.4f} to {df[lat].max():.4f}")
        print(f"Lon range: {df[lon].min():.4f} to {df[lon].max():.4f}")
    else:
        print("Could not auto-detect lat/lon columns; check the column list above.")

    flag = pick(df.columns, FLAG_NAMES)
    if flag:
        print("-" * 60)
        print(f"Observed/missing breakdown by '{flag}':")
        print(df[flag].value_counts(dropna=False).to_string())
    else:
        sm = pick(df.columns, SM_NAMES)
        if sm:
            n_na = int(df[sm].isna().sum())
            print("-" * 60)
            print(f"Using NaN in '{sm}' as an 'originally missing' proxy:")
            print(f"  observed:           {len(df) - n_na}")
            print(f"  originally missing: {n_na}")
        else:
            print()
            print("No observed/missing flag found. Paste the column list above "
                  "and I'll point the script at the exact column.")


if __name__ == "__main__":
    main()
