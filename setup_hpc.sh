#!/bin/bash
# ==============================================================
# setup_hpc.sh
#
# Run this ONCE from your LAPTOP to:
#   1. Create all individual .sbatch files locally
#   2. rsync all updated scripts to HPC
#
# Usage (from your laptop, inside SM_forecasting project root):
#   bash src/code/smap_gap_filling/setup_hpc.sh
# ==============================================================

set -e

LAPTOP_PROJECT=/home/armaghan/projects/SM_forecasting
HPC_USER=alaedini
HPC_DTN=novadtn.its.iastate.edu
HPC_PROJECT=/work/estherjo/alaedini/projects/gap-filling
SCRIPTS_DIR=$LAPTOP_PROJECT/src/code/smap_gap_filling
SBATCH_DIR=$LAPTOP_PROJECT/sbatch

mkdir -p "$SBATCH_DIR"
mkdir -p "$LAPTOP_PROJECT/logs"

echo "=== Creating individual sbatch files in $SBATCH_DIR ==="

# ---- 01 ----
cat > "$SBATCH_DIR/run_01_smap_lattice.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=smap_lattice
#SBATCH --output=logs/01_smap_lattice_%j.out
#SBATCH --error=logs/01_smap_lattice_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
EOF

# ---- 01 ----
cat > "$SBATCH_DIR/run_01_smap_lattice.sbatch" << 'EOF'
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
EOF

# ---- 03 ----
cat > "$SBATCH_DIR/run_03_iem_kriging.sbatch" << 'EOF'
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
EOF

# ---- 05 ----
cat > "$SBATCH_DIR/run_05_full_smap_iem.sbatch" << 'EOF'
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

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
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
EOF

# ---- 09 ----
cat > "$SBATCH_DIR/run_09_feature_selection.sbatch" << 'EOF'
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
EOF

# ---- 10a ----
cat > "$SBATCH_DIR/run_10a_ml_validation.sbatch" << 'EOF'
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
EOF

# ---- 10b ----
cat > "$SBATCH_DIR/run_10b_interp_validation.sbatch" << 'EOF'
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

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
cd $SMAP_PROJECT_ROOT
mkdir -p logs

module purge
module load micromamba
export MAMBA_ROOT_PREFIX=/work/estherjo/alaedini/micromamba
eval "$(micromamba shell hook --shell=bash)"
micromamba activate smap_r
export RENV_CONFIG_AUTOLOADER_ENABLED=FALSE

echo "Starting 10b_interpolation_validation.R at $(date)"
Rscript src/code/smap_gap_filling/10b_interpolation_validation.R
echo "Done at $(date)"
EOF

# ---- 10c ----
cat > "$SBATCH_DIR/run_10c_compare.sbatch" << 'EOF'
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

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
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
EOF

# ---- 10f ----
cat > "$SBATCH_DIR/run_10f_meta_features.sbatch" << 'EOF'
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

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
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
EOF

# ---- 10g ----
cat > "$SBATCH_DIR/run_10g_meta_model.sbatch" << 'EOF'
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

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
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
EOF

# ---- 11a ----
cat > "$SBATCH_DIR/run_11a_ml_predictions.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=ml_gapfill
#SBATCH --output=logs/11a_ml_predictions_%j.out
#SBATCH --error=logs/11a_ml_predictions_%j.err
#SBATCH --time=12:00:00
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

echo "Starting 11a at $(date)"
python src/code/smap_gap_filling/11a_generate_ml_gapfill_predictions.py
echo "Done at $(date)"
EOF

# ---- 11b ----
cat > "$SBATCH_DIR/run_11b_interp_predictions.sbatch" << 'EOF'
#!/bin/bash
#SBATCH --job-name=interp_gapfill
#SBATCH --output=logs/11b_interp_predictions_%j.out
#SBATCH --error=logs/11b_interp_predictions_%j.err
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
micromamba activate smap_r
export RENV_CONFIG_AUTOLOADER_ENABLED=FALSE

echo "Starting 11b at $(date)"
Rscript src/code/smap_gap_filling/11b_generate_interpolation_gapfill_predictions.R
echo "Done at $(date)"
EOF

# ---- 11c ----
cat > "$SBATCH_DIR/run_11c_finalize.sbatch" << 'EOF'
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

export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/gap-filling
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
EOF

echo ""
echo "=== All sbatch files created in $SBATCH_DIR ==="
ls -lh "$SBATCH_DIR/"
echo ""

echo "=== Syncing scripts to HPC ==="
rsync -avh --progress \
  "$SCRIPTS_DIR/" \
  "$HPC_USER@$HPC_DTN:$HPC_PROJECT/src/code/smap_gap_filling/"

echo ""
echo "=== Syncing sbatch files to HPC ==="
rsync -avh --progress \
  "$SBATCH_DIR/" \
  "$HPC_USER@$HPC_DTN:$HPC_PROJECT/sbatch/"

echo ""
echo "=== Creating logs folder on HPC ==="
ssh "$HPC_USER@$HPC_DTN" "mkdir -p $HPC_PROJECT/logs"

echo ""
echo "=== Done! ==="
echo ""
echo "Now SSH into HPC and submit jobs in order:"
echo "  ssh alaedini@nova.its.iastate.edu"
echo "  cd /work/estherjo/alaedini/projects/gap-filling"
echo ""
echo "  # If 03 and 05 already ran, skip to:"
echo "  sbatch sbatch/run_10a_ml_validation.sbatch"
echo "  sbatch sbatch/run_10b_interp_validation.sbatch"
echo "  # Wait for both, then:"
echo "  sbatch sbatch/run_10c_compare.sbatch"
echo "  sbatch sbatch/run_10f_meta_features.sbatch"
echo "  sbatch sbatch/run_10g_meta_model.sbatch"
echo "  # Wait for 10g, then run 11a and 11b in parallel:"
echo "  sbatch sbatch/run_11a_ml_predictions.sbatch"
echo "  sbatch sbatch/run_11b_interp_predictions.sbatch"
echo "  # Wait for both, then:"
echo "  sbatch sbatch/run_11c_finalize.sbatch"