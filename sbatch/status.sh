#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${SMAP_PROJECT_ROOT:-/work/estherjo/alaedini/projects/gap-filling}"
cd "${PROJECT_ROOT}"

echo "Current queue"
squeue -u "${USER}" -o "%.18i %.22j %.9T %.10M %.10l %.6D %R"

echo
echo "Today's SMAP jobs"
sacct -S today -u "${USER}" -X \
  --format=JobID,JobName%22,State,ExitCode,Elapsed,Start,End \
  | grep -E 'JobID|smap' || true

echo
echo "Nonempty error logs"
find logs/hpc_pipeline -maxdepth 1 -type f -name '*.err' -size +0c -print 2>/dev/null || true
