#!/usr/bin/env bash
# Submit preprocessing and modeling STRICTLY SEQUENTIALLY.
#
# Use only when lattice, IEM PTA, and complete SMAP/IEM products must be rebuilt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SMAP_PROJECT_ROOT:-/work/estherjo/alaedini/projects/gap-filling}"

cd "${PROJECT_ROOT}"
mkdir -p logs/hpc_pipeline

submit_job() {
    local dependency="$1"
    local filename="$2"

    if [[ -n "${dependency}" ]]; then
        sbatch --parsable --dependency="${dependency}" "${SCRIPT_DIR}/${filename}"
    else
        sbatch --parsable "${SCRIPT_DIR}/${filename}"
    fi
}


J01=$(submit_job "" "01_smap_lattice.sbatch")
J03=$(submit_job "afterok:${J01}" "03_iem_kriging.sbatch")
J05=$(submit_job "afterok:${J03}" "05_full_smap_iem.sbatch")
J07=$(submit_job "afterok:${J05}" "07_validate_full_data.sbatch")
J09=$(submit_job "afterok:${J07}" "09_prepare_modeling.sbatch")

J10A=$(submit_job "afterok:${J09}" "10a_ml_validation.sbatch")
J10B=$(submit_job "afterok:${J10A}" "10b_gi_validation.sbatch")

J10C=$(submit_job "afterok:${J10B}" "10c_build_stack.sbatch")

J10D=$(submit_job "afterok:${J10C}" "10d_ml_test.sbatch")
J10E=$(submit_job "afterok:${J10D}" "10e_gi_test.sbatch")

J10H=$(submit_job "afterok:${J10E}" "10h_evaluate_stack.sbatch")

J11A=$(submit_job "afterok:${J10H}" "11a_ml_gapfill.sbatch")
J11B=$(submit_job "afterok:${J11A}" "11b_gi_gapfill.sbatch")

J11C=$(submit_job "afterok:${J11B}" "11c_finalize.sbatch")

J12=""
if [[ "${SUBMIT_VISUALIZATIONS:-0}" == "1" ]]; then
    J12=$(submit_job "afterok:${J11C}" "12_visualize.sbatch")
fi

record="logs/hpc_pipeline/full_jobs_sequential_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "STRICTLY SEQUENTIAL FULL SMAP PIPELINE"
    echo "Submitted: $(date --iso-8601=seconds)"
    echo
    echo "01 lattice:                ${J01}"
    echo "03 IEM kriging:            ${J03}"
    echo "05 full data:              ${J05}"
    echo "07 validate complete data: ${J07}"
    echo "09 prepare modeling:       ${J09}"
    echo "10a ML validation:         ${J10A}"
    echo "10b GI validation:         ${J10B}"
    echo "10c build stack:           ${J10C}"
    echo "10d ML test:               ${J10D}"
    echo "10e GI test:               ${J10E}"
    echo "10h evaluate stack:        ${J10H}"
    echo "11a ML gap predictions:    ${J11A}"
    echo "11b GI gap predictions:    ${J11B}"
    echo "11c finalize:              ${J11C}"
    [[ -n "${J12}" ]] && echo "12 visualizations:         ${J12}"
} | tee "${record}"

echo
echo "Every job depends on the immediately preceding job."
echo "Only one job from this pipeline can run at a time."
echo
echo "Monitor with:"
echo "  bash sbatch/status.sh"
