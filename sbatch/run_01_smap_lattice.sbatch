#!/bin/bash
#SBATCH --job-name=smap_lattice
#SBATCH --output=logs/01_smap_lattice_%j.out
#SBATCH --error=logs/01_smap_lattice_%j.err
#SBATCH --time=02:00:00
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

echo "Starting 01_smap_lattice.py at $(date)"
python src/code/smap_gap_filling/01_smap_lattice.py
echo "Done at $(date)"
