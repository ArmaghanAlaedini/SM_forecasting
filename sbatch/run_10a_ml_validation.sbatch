#!/bin/bash
#SBATCH --job-name=ml_validation
#SBATCH --output=logs/10a_ml_validation_%j.out
#SBATCH --error=logs/10a_ml_validation_%j.err
#SBATCH --time=08:00:00
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

echo "Starting 10a_ML_validation.py at $(date)"
python src/code/smap_gap_filling/10a_ML_validation.py
echo "Done at $(date)"
