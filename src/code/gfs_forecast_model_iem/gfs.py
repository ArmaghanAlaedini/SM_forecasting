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

from src.config.paths import PROC_ISU_STATIONS_FULL, RAW_GFS_CSV, PROC_GFS, ensure_dirs

ensure_dirs()

df_gfs = pd.read_csv(RAW_GFS_CSV)

print("gfs path:", RAW_GFS_CSV / "gfs.csv")

print(df_gfs.head())
print(df_gfs.columns.tolist())