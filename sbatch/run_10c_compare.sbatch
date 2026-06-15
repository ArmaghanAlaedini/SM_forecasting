#!/bin/bash
#SBATCH --job-name=compare_val
#SBATCH --output=logs/10c_compare_%j.out
#SBATCH --error=logs/10c_compare_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/iem_pta_kriging
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_pta

echo "Starting 10c_compare_validation_results.py at $(date)"
python src/code/smap_gap_filling/10c_compare_validation_results.py
echo "Done at $(date)"
