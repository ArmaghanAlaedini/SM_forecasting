#!/usr/bin/env bash
# Run on the laptop. Preview is the default; use --apply after reviewing it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCAL_CODE="${LOCAL_ROOT}/src/code/smap_gap_filling"
LOCAL_SBATCH="${LOCAL_ROOT}/sbatch"

REMOTE_HOST="${REMOTE_HOST:-alaedini@novadtn.its.iastate.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/estherjo/alaedini/projects/gap-filling}"

extra=(--dry-run)
[[ "${1:-}" == "--apply" ]] && extra=()

echo "Synchronizing code..."
rsync -avh --progress --delete \
  --exclude='__pycache__/' --exclude='*.pyc' \
  "${extra[@]}" \
  "${LOCAL_CODE}/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/src/code/smap_gap_filling/"

echo
echo "Synchronizing the single clean sbatch folder..."
rsync -avh --progress --delete \
  "${extra[@]}" \
  "${LOCAL_SBATCH}/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/sbatch/"

if [[ "${#extra[@]}" -gt 0 ]]; then
    echo
    echo "Dry run only. Apply with:"
    echo "  bash sbatch/sync_to_nova.sh --apply"
fi
