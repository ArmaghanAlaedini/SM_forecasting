
from pathlib import Path
import sys
import pandas as pd


def _find_project_root(start: Path) -> Path:
    """
    Walk upward until we find the project root.
    This makes the script work both:
    - line by line in the REPL
    - as a script from the terminal
    """
    start = start.resolve()
    base = start if start.is_dir() else start.parent

    for path in [base, *base.parents]:
        if (path / ".git").exists():
            return path
        if (path / "environment.yml").exists() and (path / "src").exists():
            return path

    raise RuntimeError(
        "Could not find project root. Start the REPL from the project root "
        "or run the script from inside the repository."
    )


try:
    _START = Path(__file__).resolve()
except NameError:
    _START = Path.cwd().resolve()

PROJECT_ROOT = _find_project_root(_START)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import RAW_ISU, PROC_ISU_STATIONS_FULL, ensure_dirs

ensure_dirs()

stations = pd.read_csv(RAW_ISU / "stations.csv")
stations_meta = pd.read_csv(RAW_ISU / "stations_meta.csv")

print("stations path:", RAW_ISU / "stations.csv")
print("metadata path:", RAW_ISU / "stations_meta.csv")

print(stations.head())
print(stations.columns.tolist())
print(stations_meta.head())
print(stations_meta.columns.tolist())

meta_small = stations_meta[["stid", "station_name", "lat", "lon", "elev", "iem_network"]]
print(meta_small.head())

print(stations["station"].head(10).tolist())
print(stations_meta["stid"].head(10).tolist())

main_ids = set(stations["station"])
meta_ids = set(stations_meta["stid"])

print("Stations in main data:", len(main_ids))
print("Stations in metadata:", len(meta_ids))
print("Matched IDs:", len(main_ids & meta_ids))
print("In main but not metadata:", main_ids - meta_ids)
print("In metadata but not main:", meta_ids - main_ids)

stations_full = stations.merge(
    meta_small,
    left_on="station",
    right_on="stid",
    how="left"
)

stations_full = stations_full.drop(columns="stid")

print(stations_full[["station", "station_name", "lat", "lon", "elev", "iem_network"]].head())
print(stations_full[["lat", "lon"]].isna().sum())
print("Duplicate stid in metadata:", stations_meta["stid"].duplicated().sum())

stations_full.to_csv(PROC_ISU_STATIONS_FULL, index=False)
print("Saved to:", PROC_ISU_STATIONS_FULL)
