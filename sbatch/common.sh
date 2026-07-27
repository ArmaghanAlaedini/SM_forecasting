#!/usr/bin/env bash
# Common Nova setup for all SMAP Slurm jobs.
set -euo pipefail

export SMAP_PROJECT_ROOT="${SMAP_PROJECT_ROOT:-/work/estherjo/alaedini/projects/gap-filling}"
export SMAP_DATA_ROOT="${SMAP_DATA_ROOT:-${SMAP_PROJECT_ROOT}/src/data}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/work/estherjo/alaedini/micromamba}"

PYTHON_ENV="${PYTHON_ENV:-smap_pta}"
R_ENV="${R_ENV:-smap_r}"

CODE_DIR="${SMAP_PROJECT_ROOT}/src/code/smap_gap_filling"
VIZ_DIR="${CODE_DIR}/Visualization"
LOG_DIR="${SMAP_PROJECT_ROOT}/logs/hpc_pipeline"

if ! type module >/dev/null 2>&1; then
    source /etc/profile 2>/dev/null || true
fi

module purge
module load micromamba
eval "$(micromamba shell hook --shell=bash)"

mkdir -p "${LOG_DIR}"
cd "${SMAP_PROJECT_ROOT}"

# Do not let numerical libraries exceed the Slurm CPU allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export RENV_CONFIG_AUTOLOADER_ENABLED=FALSE

activate_python() {
    micromamba activate "${PYTHON_ENV}"
    echo "Python environment: ${CONDA_PREFIX:-unknown}"
    which python
    python --version
}

activate_r() {
    micromamba activate "${R_ENV}"
    echo "R environment: ${CONDA_PREFIX:-unknown}"
    which Rscript
    Rscript --version
}

run_python() {
    local script="$1"
    echo
    echo "======================================================================"
    echo "START: ${script}"
    echo "TIME:  $(date --iso-8601=seconds)"
    echo "HOST:  $(hostname)"
    echo "CPUS:  ${SLURM_CPUS_PER_TASK:-unknown}"
    echo "======================================================================"
    python "${CODE_DIR}/${script}"
    echo "FINISH: ${script} at $(date --iso-8601=seconds)"
}

run_r() {
    local script="$1"
    echo
    echo "======================================================================"
    echo "START: ${script}"
    echo "TIME:  $(date --iso-8601=seconds)"
    echo "HOST:  $(hostname)"
    echo "CPUS:  ${SLURM_CPUS_PER_TASK:-unknown}"
    echo "======================================================================"
    Rscript --vanilla "${CODE_DIR}/${script}"
    echo "FINISH: ${script} at $(date --iso-8601=seconds)"
}
