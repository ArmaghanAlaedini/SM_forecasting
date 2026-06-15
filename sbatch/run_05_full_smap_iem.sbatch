#!/bin/bash
#SBATCH --job-name=full_smap_iem
#SBATCH --output=logs/05_full_smap_iem_%j.out
#SBATCH --error=logs/05_full_smap_iem_%j.err
#SBATCH --time=06:00:00
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

echo "Starting 05_full_smap_iem.py at $(date)"
python src/code/smap_gap_filling/05_full_smap_iem.py
echo "Done at $(date)"
