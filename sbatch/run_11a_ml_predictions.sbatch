#!/bin/bash
#SBATCH --job-name=ml_gapfill
#SBATCH --output=logs/11a_ml_predictions_%j.out
#SBATCH --error=logs/11a_ml_predictions_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_pta

echo "Starting 11a at $(date)"
python src/code/smap_gap_filling/11a_generate_ml_gapfill_predictions.py
echo "Done at $(date)"
