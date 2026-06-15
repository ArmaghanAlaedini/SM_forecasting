#!/bin/bash
#SBATCH --job-name=meta_features
#SBATCH --output=logs/10f_meta_features_%j.out
#SBATCH --error=logs/10f_meta_features_%j.err
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

echo "Starting 10f at $(date)"
python src/code/smap_gap_filling/10f_generate_stacking_meta_features.py
echo "Done at $(date)"
