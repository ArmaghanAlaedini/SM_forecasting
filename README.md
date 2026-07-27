# 🛰️ SMAP Soil Moisture Gap Filling and Forecasting over Iowa

This repository contains a Python/R workflow for:

1. **Reconstructing missing SMAP soil moisture (SM) retrievals** over Iowa using machine learning (ML), geostatistical interpolation (GI), standard interpolation, and ridge-regression stacking
2. **Forecasting future SM** from the reconstructed SMAP record *(planned)*
3. **Translating pixel-level products to Iowa civil townships** through area-to-area kriging and change of support *(planned)*

> **Data availability:** Raw SMAP files and Iowa Environmental Mesonet (IEM) station observations are not distributed with this repository.
>
> - SMAP data: [NASA Earthdata / NSIDC](https://nsidc.org/data/smap)
> - IEM data: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu)

---

## 🧠 Overview

SMAP retrieval fields contain missing pixels because of orbital swath gaps, retrieval-quality screening, frozen-ground conditions, radio-frequency interference, and other retrieval failures. This project reconstructs those missing pixels while preserving every originally observed SMAP value unchanged.

### Stage 1 — SMAP gap filling *(implemented)*

The gap-filling workflow:

1. Builds a fixed 9-km SMAP lattice over Iowa and a surrounding spatial buffer.
2. Translates daily IEM station variables to SMAP pixel support using point-to-area (PTA) kriging.
3. Evaluates five candidate ML models and three spatial interpolation methods on shared artificial gaps.
4. Retains three ML models and three interpolation methods as six stacking base learners.
5. Trains a ridge-regression stacking meta-model using aligned 2024 spatial-block predictions.
6. Evaluates the frozen stack on an independent 2025 test set.
7. Applies the tested workflow to original missing SMAP pixels from 2020–2025.

### Stage 2 — SM forecasting *(planned)*

The reconstructed daily SMAP grids will be used to train models that predict future SM at the SMAP pixel level.

### Stage 3 — Township-scale translation *(planned)*

Pixel-level SM products will be translated to Iowa civil-township support using area-to-area kriging, producing township-level estimates and uncertainty layers.

### Temporal design

| Period | Role |
|---|---|
| 2020–2023 | Train the ML base models |
| 2024 | Feature/model validation and ridge meta-model development |
| 2025 | Independent test of the frozen base models and ridge stack |
| Original gaps, 2020–2025 | Final reconstruction after testing |

All workflow components that use randomness use the project seed **1234**.

---

## 📂 Repository Structure

```text
SM_forecasting/
├── src/
│   └── code/
│       ├── smap_gap_filling/              # Stage 1
│       │   ├── 00_config.py
│       │   ├── 01_smap_lattice.py
│       │   ├── 03_iem_pta_kriging.py
│       │   ├── 05_full_smap_iem.py
│       │   ├── 07_validate_full_smap_iem.py
│       │   ├── 09_feature_selection.py
│       │   ├── 10_generate_holdout_manifests.py
│       │   ├── 10a_ML_validation.py
│       │   ├── 10b_interpolation_validation.R
│       │   ├── 10c_compare_validation_results.py
│       │   ├── 10d_selected_methods_test.py
│       │   ├── 10e_selected_interpolation_test.R
│       │   ├── 10f_generate_stacking_meta_features.py
│       │   ├── 10g_train_stacking_meta_model.py
│       │   ├── 10h_evaluate_stacking_test.py
│       │   ├── 11_gapfilling_setting.py
│       │   ├── 11a_generate_ml_gapfill_predictions.py
│       │   ├── 11b_generate_interpolation_gapfill_predictions.R
│       │   ├── 11c_stack_and_finalize_gapfills.py
│       │   ├── gapfill_workflow_common.py
│       │   ├── gapfill_geostat_common.R
│       │   ├── run_gapfill_modeling_pipeline.sh
│       │   ├── README_FIXED_WORKFLOW.md
│       │   ├── CHANGELOG_AND_TESTS.md
│       │   └── Visualization/
│       │       ├── 02_visualize_lattice.py
│       │       ├── 04_visualize_iem_pta.py
│       │       ├── 06_visualize_full_smap_iem_one_day.py
│       │       ├── 08_visualize_complete_on_iowa_boundaries.py
│       │       ├── 12a_visualize_validation_results.py
│       │       └── 12b_visualize_gapfill_results.py
│       │
│       ├── smap_forecasting/              # Stage 2: planned
│       │   └── (coming soon)
│       │
│       └── township_kriging/              # Stage 3: planned
│           └── (coming soon)
│
├── sbatch/                                # Sequential Nova/Slurm workflow
│   ├── common.sh
│   ├── preflight.sh
│   ├── archive_old_outputs.sh
│   ├── submit_modeling_pipeline.sh
│   ├── submit_full_pipeline.sh
│   ├── status.sh
│   ├── sync_to_nova.sh
│   ├── copy_results_back.sh
│   ├── 01_smap_lattice.sbatch
│   ├── 03_iem_kriging.sbatch
│   ├── 05_full_smap_iem.sbatch
│   ├── 07_validate_full_data.sbatch
│   ├── 09_prepare_modeling.sbatch
│   ├── 10a_ml_validation.sbatch
│   ├── 10b_gi_validation.sbatch
│   ├── 10c_build_stack.sbatch
│   ├── 10d_ml_test.sbatch
│   ├── 10e_gi_test.sbatch
│   ├── 10h_evaluate_stack.sbatch
│   ├── 11a_ml_gapfill.sbatch
│   ├── 11b_gi_gapfill.sbatch
│   ├── 11c_finalize.sbatch
│   └── 12_visualize.sbatch
│
├── logs/
├── environment.yml
├── renv.lock
└── setup_hpc.sh
```

`11_gapfilling_setting.py` is retained as a compatibility wrapper. The active workflow settings are defined centrally in `00_config.py`.

---

## ✅ Stage 1 — Gap Filling

### Base methods

Five candidate ML models are evaluated during 2024 validation:

- Random Forest
- Extra Trees
- Histogram Gradient Boosting
- XGBoost
- Feed-forward neural network

The retained ML base learners are:

- **XGBoost**
- **Histogram Gradient Boosting**
- **Random Forest**

Three spatial interpolation methods are evaluated and retained:

- **Detrended centroid ordinary kriging**
- **Regression kriging**
- **Same-day nearest-neighbor interpolation**

The final stack therefore contains **six base learners**.

### Shared artificial gaps

Artificial gaps are selected once by `10_generate_holdout_manifests.py`. It creates:

- a shared 2024 validation manifest;
- a shared 2025 independent-test manifest.

Both Python and R scripts read the exact same date–pass–pixel target keys.

Two holdout designs are used:

- **Random-cell:** approximately 25% of observed pixels are withheld within each date–pass retrieval.
- **Spatial-block:** a contiguous \(2 \times 2\) block from a \(4 \times 4\) spatial quantile grid is withheld, representing approximately 25% of observed pixels.

Spatial-block validation is the primary model-selection design because it better represents spatially clustered SMAP gaps.

### ML predictors

The selected ML feature set contains:

- 20 IEM variables translated to SMAP pixel support;
- projected pixel coordinates \(x\) and \(y\);
- \(\sin(\mathrm{DOY})\) and \(\cos(\mathrm{DOY})\);
- a binary AM/PM pass indicator.

AM and PM rows are pooled for ML training, with satellite pass included as a predictor. Missing ML predictor values are imputed using medians estimated from the corresponding training data.

### GI methods

GI and nearest-neighbor methods are fitted independently for each date–pass retrieval using the observed donor pixels remaining after artificial targets are removed.

- **Same-day nearest neighbor** assigns each target the SM value of the closest observed pixel in the same retrieval.
- **Detrended centroid ordinary kriging** fits an optional centered quadratic spatial trend, kriges the residuals using a spherical variogram, and adds the trend back.
- **Regression kriging** fits a retrieval-specific trend using usable IEM-derived covariates, kriges the residuals, and adds the trend prediction back.

The shared `gapfill_geostat_common.R` implementation is used during validation, independent testing, and final original-gap prediction to prevent the GI settings from drifting across scripts.

### Ridge-regression stacking

The ridge meta-training table is constructed from the selected 2024 spatial-block predictions.

The workflow:

1. Aligns predictions by date, pass, and SMAP pixel identifier.
2. Retains only rows with finite predictions from all six base learners.
3. Adds \(x\), \(y\), \(\sin(\mathrm{DOY})\), \(\cos(\mathrm{DOY})\), and the pass indicator as additive context variables.
4. Standardizes the meta-features.
5. Selects the ridge regularization parameter using grouped cross-validation, where complete date–pass retrievals define the groups.
6. Refits the final ridge pipeline using all eligible 2024 meta-training rows.
7. Freezes the fitted pipeline before any 2025 evaluation.

Base-model predictions are **not median-imputed** in the corrected stack. Ridge coefficients are standardized regression coefficients; they are not constrained convex weights and do not vary by season, location, or pass unless interaction terms are explicitly added.

### Independent 2025 test

The selected ML models are retrained using observed 2020–2023 rows and applied to the shared 2025 target pixels. The three spatial methods predict the exact same targets.

`10h_evaluate_stacking_test.py` inner-joins all six base predictions, applies the frozen 2024 ridge model, and compares every base learner and the stack on identical 2025 common-support rows.

### Final original-gap reconstruction

After independent testing:

- the selected ML models predict original missing pixels from 2020–2025;
- the three spatial methods predict the same original missing pixels;
- the ridge model is applied only where all six base predictions and required context variables are finite;
- originally observed SMAP values remain unchanged.

When stacking is unavailable for a missing pixel, the explicit fallback order is:

1. Same-day nearest-neighbor interpolation
2. Detrended centroid ordinary kriging
3. Regression kriging
4. XGBoost
5. Histogram Gradient Boosting
6. Random Forest

Filled values are not clipped unless clipping is enabled in `00_config.py`.

---

## 📜 Script Descriptions

### Preprocessing and data construction

`00_config.py`  
Central configuration for paths, years, project seed, feature sets, model settings, holdout rules, GI settings, stacking settings, and output folders.

`01_smap_lattice.py`  
Builds the fixed Iowa SMAP lattice and stable pixel identifiers from raw SMAP files.

`03_iem_pta_kriging.py`  
Translates daily IEM station variables to SMAP pixel support using PTA sampling and kriging.

`05_full_smap_iem.py`  
Combines the fixed lattice, observed/originally missing SMAP values, and IEM PTA predictors into complete daily AM and PM files.

`07_validate_full_smap_iem.py`  
Checks key, row-count, subset, and value consistency in the completed SMAP–IEM files before modeling.

### Feature and base-model development

`09_feature_selection.py`  
Audits predictor availability and computes descriptive training-period association statistics. Formal feature-set and model comparison occurs in `10a`.

`10_generate_holdout_manifests.py`  
Creates the shared 2024 validation and 2025 test manifests. This is the only script that selects artificial gaps.

`10a_ML_validation.py`  
Trains the five candidate ML models on 2020–2023 and predicts all shared 2024 targets for all candidate feature sets.

`10b_interpolation_validation.R`  
Predicts the same shared 2024 targets with detrended centroid ordinary kriging, regression kriging, and same-day nearest-neighbor interpolation.

`10c_compare_validation_results.py`  
Recomputes prediction-level ML and spatial-method metrics, reports method-specific and common-support results, and verifies that the configured retained ML models agree with the validation ranking.

### Stacking development

`10f_generate_stacking_meta_features.py`  
Filters to selected 2024 spatial-block predictions, pivots the six retained base learners, inner-joins them by date/pass/pixel, and keeps only complete finite prediction rows.

`10g_train_stacking_meta_model.py`  
Trains and saves the standardized ridge pipeline using grouped date–pass cross-validation and an exact feature contract.

### Independent 2025 testing

`10d_selected_methods_test.py`  
Retrains the three retained ML models on 2020–2023 and predicts the shared 2025 targets.

`10e_selected_interpolation_test.R`  
Predicts the same shared 2025 targets with the three spatial methods using the same GI implementation as validation.

`10h_evaluate_stacking_test.py`  
Applies the frozen 2024 ridge model to aligned 2025 common-support rows and compares the stack with all six base learners.

### Original-gap prediction and finalization

`11_gapfilling_setting.py`  
Compatibility wrapper around `00_config.py`.

`11a_generate_ml_gapfill_predictions.py`  
Trains the retained ML models on observed 2020–2023 rows and predicts original missing pixels from 2020–2025.

`11b_generate_interpolation_gapfill_predictions.R`  
Predicts original missing pixels from same-day observed donors using all three spatial methods. It does not silently replace a failed kriging prediction with nearest neighbor.

`11c_stack_and_finalize_gapfills.py`  
Applies ridge stacking where eligible, otherwise applies the explicit fallback waterfall, preserves observed values, and writes final gap-filled files and summary tables.

---

## ▶️ How to Run Stage 1

### Local modeling workflow

Assuming `03_full_smap_iem_data` already exists and has passed validation:

```bash
conda activate py312
cd ~/projects/SM_forecasting

bash src/code/smap_gap_filling/run_gapfill_modeling_pipeline.sh
```

Equivalent explicit order:

```bash
python src/code/smap_gap_filling/07_validate_full_smap_iem.py
python src/code/smap_gap_filling/09_feature_selection.py
python src/code/smap_gap_filling/10_generate_holdout_manifests.py
python src/code/smap_gap_filling/10a_ML_validation.py
Rscript src/code/smap_gap_filling/10b_interpolation_validation.R
python src/code/smap_gap_filling/10c_compare_validation_results.py
python src/code/smap_gap_filling/10f_generate_stacking_meta_features.py
python src/code/smap_gap_filling/10g_train_stacking_meta_model.py
python src/code/smap_gap_filling/10d_selected_methods_test.py
Rscript src/code/smap_gap_filling/10e_selected_interpolation_test.R
python src/code/smap_gap_filling/10h_evaluate_stacking_test.py
python src/code/smap_gap_filling/11a_generate_ml_gapfill_predictions.py
Rscript src/code/smap_gap_filling/11b_generate_interpolation_gapfill_predictions.R
python src/code/smap_gap_filling/11c_stack_and_finalize_gapfills.py
```

To rebuild the preprocessing products too, run these first:

```bash
python src/code/smap_gap_filling/01_smap_lattice.py
python src/code/smap_gap_filling/03_iem_pta_kriging.py
python src/code/smap_gap_filling/05_full_smap_iem.py
```

### HPC workflow: ISU Nova

The clean Slurm workflow is **strictly sequential**. At most one pipeline stage runs at a time; later jobs remain pending because of dependencies. This avoids simultaneous high-memory stages.

#### 1. Transfer the current code from the laptop

```bash
cd ~/projects/SM_forecasting

bash sbatch/sync_to_nova.sh          # dry run
bash sbatch/sync_to_nova.sh --apply  # transfer
```

#### 2. Run preflight checks on Nova

```bash
ssh alaedini@nova.its.iastate.edu

cd /work/estherjo/alaedini/projects/gap-filling
chmod +x sbatch/*.sh

bash sbatch/preflight.sh
```

Continue only after the script reports:

```text
PRE-FLIGHT PASSED.
```

#### 3. Archive old downstream outputs before a clean rerun

```bash
bash sbatch/archive_old_outputs.sh          # preview
bash sbatch/archive_old_outputs.sh --apply  # archive
```

The source folder `03_full_smap_iem_data` is preserved.

#### 4. Submit the modeling workflow

Use this when the completed SMAP–IEM files already exist:

```bash
bash sbatch/submit_modeling_pipeline.sh
```

Use this only when preprocessing scripts `01`, `03`, and `05` must also be rerun:

```bash
bash sbatch/submit_full_pipeline.sh
```

#### 5. Monitor the sequential jobs

```bash
bash sbatch/status.sh
```

One job may be `RUNNING`; all later jobs should normally be `PENDING (Dependency)`.

A compact current-pipeline status table can also be generated with:

```bash
LATEST=$(ls -t logs/hpc_pipeline/modeling_jobs_sequential_*.txt | head -1)
JOBIDS=$(awk '$NF ~ /^[0-9]+$/ {print $NF}' "$LATEST" | paste -sd, -)

sacct -j "$JOBIDS" -X \
  --format=JobID,JobName%24,State%20,Elapsed,ExitCode
```

### Sync results back to the laptop

Run from the laptop:

```bash
rsync -avh --progress \
  alaedini@novadtn.its.iastate.edu:/work/estherjo/alaedini/projects/gap-filling/src/data/processed/smap_gap_filling/05_gapfill_model_validation/ \
  ~/projects/SM_forecasting/src/data/processed/smap_gap_filling/05_gapfill_model_validation/

rsync -avh --progress \
  alaedini@novadtn.its.iastate.edu:/work/estherjo/alaedini/projects/gap-filling/src/data/processed/smap_gap_filling/06_selected_methods_test/ \
  ~/projects/SM_forecasting/src/data/processed/smap_gap_filling/06_selected_methods_test/

rsync -avh --progress \
  alaedini@novadtn.its.iastate.edu:/work/estherjo/alaedini/projects/gap-filling/src/data/processed/smap_gap_filling/08_gapfilled_final/ \
  ~/projects/SM_forecasting/src/data/processed/smap_gap_filling/08_gapfilled_final/
```

---

## 📊 Visualizations

Run locally after downloading the outputs:

```bash
python src/code/smap_gap_filling/Visualization/02_visualize_lattice.py
python src/code/smap_gap_filling/Visualization/04_visualize_iem_pta.py
python src/code/smap_gap_filling/Visualization/06_visualize_full_smap_iem_one_day.py
python src/code/smap_gap_filling/Visualization/08_visualize_complete_on_iowa_boundaries.py
python src/code/smap_gap_filling/Visualization/12a_visualize_validation_results.py
python src/code/smap_gap_filling/Visualization/12b_visualize_gapfill_results.py
```

For `12b_visualize_gapfill_results.py`, edit the controls at the top of the script:

```python
SELECTED_DATE = "2025-08-19"
PASS_NAME = "am"  # "am" or "pm"
```

---

## 📦 Stage 1 Outputs

```text
src/data/processed/smap_gap_filling/
├── iem_point_to_area/
├── 03_full_smap_iem_data/
├── 04_feature_screening/
├── 05_gapfill_model_validation/
│   ├── holdouts/
│   ├── ml/
│   ├── interpolation/
│   ├── comparison/
│   └── stacking/
├── 06_selected_methods_test/
│   ├── holdouts/
│   ├── ml/
│   ├── interpolation/
│   └── stacking/
├── 07_gapfill_predictions/
│   ├── ml/
│   └── interpolation/
└── 08_gapfilled_final/
    ├── am/
    ├── pm/
    ├── gapfill_summary_by_file.csv
    └── gapfill_overall_summary.csv
```

Each final daily file contains one row per SMAP pixel, including:

- `soil_moisture`: original SMAP value; missing for an originally unavailable pixel
- `soil_moisture_filled`: observed value or reconstructed value
- `fill_status`: `observed`, `filled`, or `unfilled`
- `fill_method`: `observed`, `stacking`, or the fallback method used

Possible fallback labels are:

- `nearest_neighbor_same_day`
- `centroid_ordinary_kriging`
- `regression_kriging`
- `xgboost`
- `hist_gbdt`
- `random_forest`

---

## 🔮 Stage 2 — SM Forecasting *(planned)*

> **Status:** Not yet implemented. Future scripts will be placed in `src/code/smap_forecasting/`.

Planned components include:

- input: reconstructed daily SMAP grids from `08_gapfilled_final/`;
- target: SM forecasts 1–7 days ahead at SMAP pixel support;
- predictors: lagged SM, meteorological predictors, and cyclic temporal variables;
- baselines and candidate models: to be selected during implementation;
- temporally ordered training, validation, and testing.

---

## 🗺️ Stage 3 — Township-Scale Translation *(planned)*

> **Status:** Not yet implemented. Future scripts will be placed in `src/code/township_kriging/`.

Planned components include:

- input: reconstructed or forecast SMAP pixel products;
- target support: Iowa civil-township polygons;
- method: area-to-area kriging and change of support;
- outputs: daily township-level SM estimates and uncertainty layers.

---

## 📦 Environment

### Python

Create and activate the local conda environment:

```bash
conda env create -f environment.yml
conda activate py312
```

Core Python packages include:

- pandas
- NumPy
- SciPy
- scikit-learn
- XGBoost
- joblib
- GeoPandas
- Shapely
- PyProj
- PyKrige
- Matplotlib
- PyArrow
- netCDF4

### R

Restore the R environment using the project lockfile where applicable:

```bash
Rscript -e 'renv::restore()'
```

Core R packages include:

- data.table
- sp
- gstat
- sf
- ggplot2

The `atakrig` package is planned for the township change-of-support stage and is not required for the current pixel-level gap-filling workflow.

---

## 📩 Contact

**Maintainer:** Armaghan Alaedini  
**Email:** alaedini@iastate.edu