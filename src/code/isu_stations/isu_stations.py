from pathlib import Path
import pandas as pd
from src.config.paths import RAW_ISU_STATIONS
print(RAW_ISU_STATIONS)
import src.config.paths as paths
print(paths.RAW_ISU)
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()
ROOT
DATA_DIR = ROOT /"src"/"ISU_stations"/ "data"
DATA_DIR
stations = pd.read_csv(DATA_DIR / "stations.csv")
stations_meta = pd.read_csv(DATA_DIR / "stations_meta.csv")

stations.head()
print(stations.columns)
stations_meta.head()
print(stations_meta.columns)
meta_small = stations_meta[['stid', 'station_name', 'lat', 'lon', 'elev', 'iem_network']]
print(meta_small.head())

print(stations['station'].head(10).tolist())
print(stations_meta['stid'].head(10).tolist())

main_ids = set(stations['station'])
meta_ids = set(stations_meta['stid'])

print("Stations in main data:", len(main_ids))
print("Stations in metadata:", len(meta_ids))
print("Matched IDs:", len(main_ids & meta_ids))
print("In main but not metadata:", main_ids - meta_ids)
print("In metadata but not main:", meta_ids - main_ids)

stations_full = stations.merge(
    meta_small,
    left_on='station',
    right_on='stid',
    how='left'
)

stations_full = stations_full.drop(columns='stid')

print(stations_full[['station', 'station_name', 'lat', 'lon', 'elev', 'iem_network']].head())
print(stations_full[['lat', 'lon']].isna().sum()) # check for missing lat/lon values after merge

print(stations_meta['stid'].duplicated().sum()) # no duplicates in metadata 
stations_full.columns
stations_full.to_csv(DATA_DIR / "stations_full.csv", index=False)