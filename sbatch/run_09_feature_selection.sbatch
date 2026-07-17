#!/bin/bash
#SBATCH --job-name=feat_select
#SBATCH --output=logs/09_feature_selection_%j.out
#SBATCH --error=logs/09_feature_selection_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_pta

echo "Starting 09_feature_selection.py at $(date)"
python src/code/smap_gap_filling/09_feature_selection.py
echo "Done at $(date)"
