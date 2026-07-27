#!/usr/bin/env bash
# Archive old downstream outputs without touching the completed input data.
# Preview is the default; use --apply to move the folders.
set -euo pipefail

PROJECT_ROOT="${SMAP_PROJECT_ROOT:-/work/estherjo/alaedini/projects/gap-filling}"
BASE="${PROJECT_ROOT}/src/data/processed/smap_gap_filling"
STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="${BASE}/_previous_runs/${STAMP}"

folders=(
  04_feature_screening
  05_gapfill_model_validation
  06_selected_methods_test
  07_gapfill_predictions
  08_gapfilled_final
  09_final_visualization
)

apply=0
[[ "${1:-}" == "--apply" ]] && apply=1

for name in "${folders[@]}"; do
    src="${BASE}/${name}"
    if [[ -e "${src}" ]]; then
        if [[ "${apply}" -eq 1 ]]; then
            mkdir -p "${BACKUP}"
            echo "Moving ${src} -> ${BACKUP}/"
            mv "${src}" "${BACKUP}/"
        else
            echo "[preview] would move ${src}"
        fi
    fi
done

if [[ "${apply}" -eq 0 ]]; then
    echo "Nothing changed. Run '$0 --apply' after reviewing the paths."
else
    echo "Old outputs archived under ${BACKUP}"
fi
