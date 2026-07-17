#!/bin/bash
#SBATCH --job-name=iem_kriging
#SBATCH --output=logs/03_iem_kriging_%j.out
#SBATCH --error=logs/03_iem_kriging_%j.err
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
micromamba activate smap_pta

echo "Starting 03_iem_pta_kriging.py at $(date)"
python src/code/smap_gap_filling/03_iem_pta_kriging.py
echo "Done at $(date)"
