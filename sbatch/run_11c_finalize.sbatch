#!/bin/bash
#SBATCH --job-name=finalize_gaps
#SBATCH --output=logs/11c_finalize_%j.out
#SBATCH --error=logs/11c_finalize_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/iem_pta_kriging
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_pta

echo "Starting 11c at $(date)"
python src/code/smap_gap_filling/11c_stack_and_finalize_gapfills.py
echo "Done at $(date)"
