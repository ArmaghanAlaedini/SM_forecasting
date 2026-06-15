#!/bin/bash
#SBATCH --job-name=interp_val
#SBATCH --output=logs/10b_interp_validation_%j.out
#SBATCH --error=logs/10b_interp_validation_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=nova

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/iem_pta_kriging
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load r/4.5.1-py311-ggqalmr

echo "Starting 10b_interpolation_validation.R at $(date)"
Rscript src/code/smap_gap_filling/10b_interpolation_validation.R
echo "Done at $(date)"
