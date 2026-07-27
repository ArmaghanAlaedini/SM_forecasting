#!/usr/bin/env bash
#SBATCH --job-name=smap09_prepare
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/hpc_pipeline/%x_%j.out
#SBATCH --error=logs/hpc_pipeline/%x_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_python
run_python "09_feature_selection.py"
run_python "10_generate_holdout_manifests.py"
