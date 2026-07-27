#!/usr/bin/env bash
#SBATCH --job-name=smap12_viz
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hpc_pipeline/%x_%j.out
#SBATCH --error=logs/hpc_pipeline/%x_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_python

if [[ ! -f "${VIZ_DIR}/run_visualizations.sh" ]]; then
    echo "Visualization runner not found: ${VIZ_DIR}/run_visualizations.sh"
    exit 1
fi

bash "${VIZ_DIR}/run_visualizations.sh"
