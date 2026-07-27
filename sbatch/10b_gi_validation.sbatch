#!/usr/bin/env bash
#SBATCH --job-name=smap10b_gival
#SBATCH --partition=nova
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/hpc_pipeline/%x_%j.out
#SBATCH --error=logs/hpc_pipeline/%x_%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_r
run_r "10b_interpolation_validation.R"
