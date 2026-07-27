#!/usr/bin/env bash
#SBATCH --job-name=smap10c_stack
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
run_python "10c_compare_validation_results.py"
run_python "10f_generate_stacking_meta_features.py"
run_python "10g_train_stacking_meta_model.py"
