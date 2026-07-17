#!/bin/bash
#SBATCH --job-name=interp_gapfill
#SBATCH --output=logs/11b_interp_predictions_%j.out
#SBATCH --error=logs/11b_interp_predictions_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_r
export RENV_CONFIG_AUTOLOADER_ENABLED=FALSE

echo "Starting 11b at $(date)"
Rscript src/code/smap_gap_filling/11b_generate_interpolation_gapfill_predictions.R
echo "Done at $(date)"
