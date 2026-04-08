from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """
    Walk upward until we find the project root.
    Expected markers: .git or environment.yml plus src/.
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

ROOT = _find_project_root(_START)

SRC = ROOT / "src"
CODE = SRC / "code"
CONFIG = SRC / "config"
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

RAW_DEP_CSV = RAW_DEP / "DEP_20260405.csv"
RAW_GFS_CSV = RAW_GFS / "gfs.csv"
RAW_ISU_STATIONS = RAW_ISU / "stations.csv"
RAW_ISU_META = RAW_ISU / "stations_meta.csv"
RAW_TOWNSHIPS_SHP = RAW_TOWNSHIPS / "civil_townships_a_ia.shp"

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