#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/09_feature_selection.py"
python "$SCRIPT_DIR/10_generate_holdout_manifests.py"
python "$SCRIPT_DIR/10a_ML_validation.py"
Rscript "$SCRIPT_DIR/10b_interpolation_validation.R"
python "$SCRIPT_DIR/10c_compare_validation_results.py"
python "$SCRIPT_DIR/10f_generate_stacking_meta_features.py"
python "$SCRIPT_DIR/10g_train_stacking_meta_model.py"
python "$SCRIPT_DIR/10d_selected_methods_test.py"
Rscript "$SCRIPT_DIR/10e_selected_interpolation_test.R"
python "$SCRIPT_DIR/10h_evaluate_stacking_test.py"
python "$SCRIPT_DIR/11a_generate_ml_gapfill_predictions.py"
Rscript "$SCRIPT_DIR/11b_generate_interpolation_gapfill_predictions.R"
python "$SCRIPT_DIR/11c_stack_and_finalize_gapfills.py"
