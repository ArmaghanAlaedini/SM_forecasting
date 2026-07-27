#!/usr/bin/env bash
# Run on the laptop after the Nova workflow completes.
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/home/armaghan/projects/SM_forecasting}"
REMOTE_HOST="${REMOTE_HOST:-alaedini@novadtn.its.iastate.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/estherjo/alaedini/projects/gap-filling}"

REMOTE_BASE="${REMOTE_ROOT}/src/data/processed/smap_gap_filling"
LOCAL_BASE="${LOCAL_ROOT}/src/data/processed/smap_gap_filling"
mkdir -p "${LOCAL_BASE}"

folders=(
  04_feature_screening
  05_gapfill_model_validation
  06_selected_methods_test
  08_gapfilled_final
  09_final_visualization
)

for folder in "${folders[@]}"; do
    rsync -avh --progress \
      "${REMOTE_HOST}:${REMOTE_BASE}/${folder}/" \
      "${LOCAL_BASE}/${folder}/"
done

echo "Results copied. The very large 07_gapfill_predictions folder was skipped."
