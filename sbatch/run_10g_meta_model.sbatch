#!/bin/bash
#SBATCH --job-name=meta_model
#SBATCH --output=logs/10g_meta_model_%j.out
#SBATCH --error=logs/10g_meta_model_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/iem_pta_kriging
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_pta

echo "Starting 10g at $(date)"
python src/code/smap_gap_filling/10g_train_stacking_meta_model.py
echo "Done at $(date)"
