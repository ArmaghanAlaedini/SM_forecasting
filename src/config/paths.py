from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
CONFIG = _THIS_FILE.parent
SRC = CONFIG.parent
ROOT = SRC.parent

CODE = SRC / "code"
DATA = SRC / "data"

RAW = DATA / "raw"
PROCESSED = DATA / "processed"

RAW_DEP = RAW / "dep"
RAW_GFS = RAW / "gfs"
RAW_ISU = RAW / "isu_stations"
RAW_SMAP_OBS = RAW / "smap_observations"
RAW_TOWNSHIPS = RAW / "townships"

PROC_DEP = PROCESSED / "dep"
PROC_GFS = PROCESSED / "gfs"
PROC_ISU = PROCESSED / "isu_stations"

PROC_KRIGED = PROCESSED / "kriged_predictions"
PROC_KRIGED_AM = PROC_KRIGED / "am"
PROC_KRIGED_PM = PROC_KRIGED / "pm"

PROC_SMAP = PROCESSED / "smap_processed"
PROC_SMAP_AM = PROC_SMAP / "am"
PROC_SMAP_PM = PROC_SMAP / "pm"

RAW_ISU_STATIONS = RAW_ISU / "stations.csv"
RAW_ISU_META = RAW_ISU / "stations_meta.csv"
PROC_ISU_STATIONS_FULL = PROC_ISU / "stations_full.csv"


def ensure_dirs() -> None:
    for path in [
        PROC_DEP,
        PROC_GFS,
        PROC_ISU,
        PROC_KRIGED,
        PROC_KRIGED_AM,
        PROC_KRIGED_PM,
        PROC_SMAP,
        PROC_SMAP_AM,
        PROC_SMAP_PM,
    ]:
        path.mkdir(parents=True, exist_ok=True)