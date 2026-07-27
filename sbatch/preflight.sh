#!/usr/bin/env bash
# Light checks only; run on the Nova login node before submitting.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

echo "Project root: ${SMAP_PROJECT_ROOT}"
echo "Code folder:  ${CODE_DIR}"
echo "Data root:    ${SMAP_DATA_ROOT}"

required=(
  00_config.py
  gapfill_workflow_common.py
  gapfill_geostat_common.R
  01_smap_lattice.py
  03_iem_pta_kriging.py
  05_full_smap_iem.py
  07_validate_full_smap_iem.py
  09_feature_selection.py
  10_generate_holdout_manifests.py
  10a_ML_validation.py
  10b_interpolation_validation.R
  10c_compare_validation_results.py
  10d_selected_methods_test.py
  10e_selected_interpolation_test.R
  10f_generate_stacking_meta_features.py
  10g_train_stacking_meta_model.py
  10h_evaluate_stacking_test.py
  11a_generate_ml_gapfill_predictions.py
  11b_generate_interpolation_gapfill_predictions.R
  11c_stack_and_finalize_gapfills.py
)

for file in "${required[@]}"; do
    [[ -f "${CODE_DIR}/${file}" ]] || {
        echo "[MISSING] ${CODE_DIR}/${file}"
        exit 1
    }
done
echo "[OK] Corrected code files are present."

activate_python
python - <<'PY'
import importlib
packages = [
    "numpy", "pandas", "scipy", "sklearn", "xgboost", "matplotlib",
    "joblib", "geopandas", "shapely", "pyproj", "statsmodels",
    "pykrige", "pyarrow",
]
failed = []
for name in packages:
    try:
        module = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        failed.append(name)
        print(f"[FAIL] {name}: {exc}")
if failed:
    raise SystemExit("Missing/broken Python packages: " + ", ".join(failed))
PY

python - <<'PY'
from pathlib import Path
import importlib.util
path = Path("src/code/smap_gap_filling/00_config.py")
spec = importlib.util.spec_from_file_location("cfg", path)
cfg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)
print("Configured project root:", cfg.PROJECT_ROOT)
print("Configured data root:", cfg.DATA_ROOT)
print("Training years:", cfg.TRAIN_YEARS)
print("Validation years:", cfg.VALIDATION_YEARS)
print("Test years:", cfg.TEST_YEARS)
print("Random seed:", cfg.RANDOM_SEED)
assert cfg.RANDOM_SEED == 1234
PY

activate_r
Rscript --vanilla - <<'RS'
packages <- c("data.table", "sp", "gstat")
bad <- character()
for (p in packages) {
  if (!requireNamespace(p, quietly = TRUE)) {
    cat("[FAIL]", p, "\n")
    bad <- c(bad, p)
  } else {
    cat("[OK]", p, as.character(packageVersion(p)), "\n")
  }
}
if (length(bad) > 0) stop("Missing R packages: ", paste(bad, collapse=", "))
RS

complete="${SMAP_DATA_ROOT}/processed/smap_gap_filling/03_full_smap_iem_data"
if [[ -d "${complete}/am/complete" && -d "${complete}/pm/complete" ]]; then
    am_count=$(find "${complete}/am/complete" -maxdepth 1 -type f -name '*.csv' | wc -l)
    pm_count=$(find "${complete}/pm/complete" -maxdepth 1 -type f -name '*.csv' | wc -l)
    echo "[OK] Existing complete data: AM=${am_count}, PM=${pm_count}"
else
    echo "[INFO] Complete data are absent; use submit_full_pipeline.sh."
fi

echo "PRE-FLIGHT PASSED."
