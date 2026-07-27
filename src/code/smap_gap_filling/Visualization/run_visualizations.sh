#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model evaluation and final-product visualizations can be run after the full
# corrected modeling pipeline has completed.
python "$SCRIPT_DIR/12a_visualize_validation_and_test_results.py"
python "$SCRIPT_DIR/12b_visualize_final_gapfill_results.py" --pass-name am
python "$SCRIPT_DIR/12b_visualize_final_gapfill_results.py" --pass-name pm
