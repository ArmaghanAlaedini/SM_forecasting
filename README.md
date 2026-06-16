# 🛰️ Soil Moisture Forecasting Using Reconstructed SMAP Data (Iowa)


This repository contains a Python/R pipeline for:
1. **Gap-filling** daily NASA SMAP L3 soil moisture observations over Iowa
2. **Forecasting** future soil moisture from gap-filled history *(TBC)*
3. **Downscaling** forecasts to Iowa township level via area-to-area kriging *(TBC)*

> **Note:** Raw SMAP NC4 files and IEM station data are **not included** in this repository.
> SMAP data: [NASA Earthdata / NSIDC](https://nsidc.org/data/smap).
> IEM station data: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu).

---

## 🧠 Overview

SMAP satellite overpasses leave many pixels missing on any given day due to orbital gaps,
frozen soil masking, and radio frequency interference. This pipeline addresses this in
three stages:

**Stage 1 — Gap-Filling (complete)**
Missing SMAP pixels are filled using a layered stacking approach combining
point-to-area kriging of IEM weather station data with machine learning models
(XGBoost, HistGBDT, Random Forest), regression kriging, and a Ridge stacking meta-model
that learns the optimal blend of all base predictions from 2024 validation data.

**Stage 2 — Forecasting (TBC)**
The complete gap-filled daily SM grids will be used to train a forecasting model
that predicts future soil moisture at the SMAP pixel level.

**Stage 3 — Township Downscaling (TBC)**
Pixel-level forecasts will be downscaled to Iowa civil township boundaries using
area-to-area kriging, producing township-level soil moisture forecast products.

**Data split:**
- Train: 2020–2023
- Validation: 2024
- Test: 2025

> **HPC note:** Paths below are specific to the ISU Nova cluster. Adjust
> `HPC_PROJECT`, `MAMBA_ROOT_PREFIX`, and partition names for other systems.

---

## 📂 Repository Structure

```text
SM_forecasting/
├── src/
│   └── code/
│       ├── smap_gap_filling/          ← Stage 1: complete
│       │   ├── 00_config.py
│       │   ├── 01_smap_lattice.py
│       │   ├── 03_iem_pta_kriging.py
│       │   ├── 05_full_smap_iem.py
│       │   ├── 07_validate_full_smap_iem.py
│       │   ├── 09_feature_selection.py
│       │   ├── 10a_ML_validation.py
│       │   ├── 10b_interpolation_validation.R
│       │   ├── 10c_compare_validation_results.py
│       │   ├── 10d_selected_methods_test.py
│       │   ├── 10e_selected_interpolation_test.R
│       │   ├── 10f_generate_stacking_meta_features.py
│       │   ├── 10g_train_stacking_meta_model.py
│       │   ├── 11_gapfilling_setting.py
│       │   ├── 11a_generate_ml_gapfill_predictions.py
│       │   ├── 11b_generate_interpolation_gapfill_predictions.R
│       │   ├── 11c_stack_and_finalize_gapfills.py
│       │   └── Visualization/
│       │       ├── 02_visualize_lattice.py
│       │       ├── 04_visualize_iem_pta.py
│       │       ├── 06_visualize_full_smap_iem_one_day.py
│       │       ├── 08_visualize_complete_on_iowa_boundaries.py
│       │       ├── 12a_visualize_validation_results.py
│       │       └── 12b_visualize_gapfill_results.py
│       │
│       ├── smap_forecasting/          ← Stage 2: planned
│       │   └── (coming soon)
│       │
│       └── township_kriging/          ← Stage 3: planned
│           └── (coming soon)
│
├── sbatch/                            # HPC job scripts (Nova/ISU)
├── logs/                              # HPC job output logs
├── environment.yml                    # Python conda environment
├── renv.lock                          # R package lockfile
└── setup_hpc.sh                       # One-shot HPC setup script
```

---

## ✅ Stage 1 — Gap-Filling

### How It Works

Missing SMAP pixels are filled through a stacking ensemble. Five base models each produce
a prediction for every missing pixel:

- centroid ordinary kriging — spatial polynomial detrend → krige residuals → add trend back
- regression kriging — fits an OLS trend on IEM weather covariates, then kriges residuals
- nearest-neighbor same-day — assigns the value of the closest observed pixel
- XGBoost, HistGBDT, Random Forest — trained on 2020–2023 observed pixels with IEM features

All three kriging processes in this pipeline follow the same detrend → krige
residuals → add trend back structure, differing only in the trend model:

- **IEM point-to-area kriging** (`03`) — spatial quadratic trend on station coordinates
- **centroid ordinary kriging** — spatial quadratic trend on SMAP pixel coordinates
- **regression kriging** — OLS trend on IEM weather covariates

In every case the trend is used only if it is statistically meaningful
(F-test p < 0.05 and R² > 0.01); otherwise the method falls back to plain
ordinary kriging, so detrending never reduces accuracy.

A Ridge regression meta-model (trained on 2024 spatial-block holdout predictions)
learns the optimal weighted combination of all five base predictions. Where the
meta-model cannot predict (all base predictions are NaN), a waterfall fallback
applies the base methods in priority order.

### Script Descriptions

`00_config.py` — Central configuration. All file paths, year settings, CRS, kriging
parameters, and runtime limits. All other scripts load this.

`01_smap_lattice.py` — Scans raw SMAP NC4 files to build the fixed Iowa pixel grid
(EASE-2 projection, ~9 km cells). Saves as GeoParquet + CSV.

`03_iem_pta_kriging.py` — For each day in 2020–2025, kriges 20 IEM weather variables
from station points to SMAP polygon centroids. Uses detrended ordinary kriging:
quadratic spatial trend → krige residuals → add trend back.

`05_full_smap_iem.py` — Merges observed SMAP soil moisture with kriged IEM values
into complete daily files. Saves complete/observed/missing splits.

`07_validate_full_smap_iem.py` — Spot-checks merged files to verify SMAP values
were preserved correctly.

`09_feature_selection.py` — Screens all IEM variables as predictors. Computes
correlations, feature importances, and model scores across feature groups.

`10a_ML_validation.py` — Trains and validates ML models on 2024 observed pixels
using artificial holdouts (random-cell and spatial-block).

`10b_interpolation_validation.R` — Validates centroid ordinary kriging,
regression kriging, and nearest-neighbor on 2024 using the same holdout design.
Both kriging methods detrend before kriging (centroid OK uses a spatial x/y
polynomial trend; regression kriging uses IEM covariates), krige the residuals,
then add the trend back.

`10c_compare_validation_results.py` — Reads outputs from 10a and 10b, ranks all
methods by RMSE under spatial-block validation, writes recommendation table.

`10d_selected_methods_test.py` — Tests selected ML models on 2025 held-out data.

`10e_selected_interpolation_test.R` — Tests interpolation methods on 2025.

`10f_generate_stacking_meta_features.py` — Joins per-pixel predictions from all
base models into a single meta-training table using 2024 spatial-block holdout rows.

`10g_train_stacking_meta_model.py` — Trains a Ridge regression meta-model that
learns the optimal weights for combining all base predictions. Saves
`meta_model.joblib` and a coefficient table showing how much each method contributes.

`11_gapfilling_setting.py` — Manual controls: which models to use, which years
to fill, stacking meta-model path, clipping options.

`11a_generate_ml_gapfill_predictions.py` — Trains ML models on 2020–2023 and
predicts all real missing pixels across 2020–2025.

`11b_generate_interpolation_gapfill_predictions.R` — Runs centroid ordinary kriging,
regression kriging, and nearest-neighbor on real missing pixels across all years.
Both kriging methods detrend before kriging then add the trend back.

`11c_stack_and_finalize_gapfills.py` — Applies stacking meta-model to combine all
base predictions and writes final gap-filled CSV files. Falls back to a priority
waterfall (centroid OK → regression kriging → nearest-neighbor → ML models) for
any pixel where the meta-model returns NaN.

### How to Run (Stage 1)

**Local:**
```bash
conda activate py312
cd /home/armaghan/projects/SM_forecasting

python src/code/smap_gap_filling/03_iem_pta_kriging.py
python src/code/smap_gap_filling/05_full_smap_iem.py
python src/code/smap_gap_filling/09_feature_selection.py
python src/code/smap_gap_filling/10a_ML_validation.py
Rscript src/code/smap_gap_filling/10b_interpolation_validation.R
python src/code/smap_gap_filling/10c_compare_validation_results.py
python src/code/smap_gap_filling/10f_generate_stacking_meta_features.py
python src/code/smap_gap_filling/10g_train_stacking_meta_model.py
python src/code/smap_gap_filling/11a_generate_ml_gapfill_predictions.py
Rscript src/code/smap_gap_filling/11b_generate_interpolation_gapfill_predictions.R
python src/code/smap_gap_filling/11c_stack_and_finalize_gapfills.py
```

**HPC (Nova — Iowa State University):**
```bash
bash setup_hpc.sh  # one-time laptop setup

cd /work/estherjo/alaedini/projects/gap-filling
J03=$(sbatch sbatch/run_03_iem_kriging.sbatch | awk '{print $4}')
J05=$(sbatch --dependency=afterok:$J03 sbatch/run_05_full_smap_iem.sbatch | awk '{print $4}')
J09=$(sbatch --dependency=afterok:$J05 sbatch/run_09_feature_selection.sbatch | awk '{print $4}')
J10A=$(sbatch --dependency=afterok:$J09 sbatch/run_10a_ml_validation.sbatch | awk '{print $4}')
J10B=$(sbatch --dependency=afterok:$J09 sbatch/run_10b_interp_validation.sbatch | awk '{print $4}')
J10F=$(sbatch --dependency=afterok:$J10A:$J10B sbatch/run_10f_meta_features.sbatch | awk '{print $4}')
J10G=$(sbatch --dependency=afterok:$J10F sbatch/run_10g_meta_model.sbatch | awk '{print $4}')
J11A=$(sbatch --dependency=afterok:$J10G sbatch/run_11a_ml_predictions.sbatch | awk '{print $4}')
J11B=$(sbatch --dependency=afterok:$J10G sbatch/run_11b_interp_predictions.sbatch | awk '{print $4}')
sbatch --dependency=afterok:$J11A:$J11B sbatch/run_11c_finalize.sbatch
```

**Sync results back to laptop:**
```bash
rsync -avh --progress \
  alaedini@novadtn.its.iastate.edu:/work/estherjo/alaedini/projects/gap-filling/src/data/processed/smap_gap_filling/08_gapfilled_final/ \
  /home/armaghan/projects/SM_forecasting/src/data/processed/smap_gap_filling/08_gapfilled_final/

rsync -avh --progress \
  alaedini@novadtn.its.iastate.edu:/work/estherjo/alaedini/projects/gap-filling/src/data/processed/smap_gap_filling/05_gapfill_model_validation/ \
  /home/armaghan/projects/SM_forecasting/src/data/processed/smap_gap_filling/05_gapfill_model_validation/
```

**Visualizations (run locally after downloading results):**
```bash
python src/code/smap_gap_filling/Visualization/02_visualize_lattice.py
python src/code/smap_gap_filling/Visualization/04_visualize_iem_pta.py
python src/code/smap_gap_filling/Visualization/06_visualize_full_smap_iem_one_day.py
python src/code/smap_gap_filling/Visualization/08_visualize_complete_on_iowa_boundaries.py
python src/code/smap_gap_filling/Visualization/12a_visualize_validation_results.py
python src/code/smap_gap_filling/Visualization/12b_visualize_gapfill_results.py
```

For `12b`, edit the user controls at the top:
```python
SELECTED_DATE = "2025-08-19"    # any date you want to visualize
PASS_NAME     = "am"            # "am" or "pm"
```

### Stage 1 Outputs

```text
src/data/processed/smap_gap_filling/
├── iem_point_to_area/           # Daily kriged IEM variables per SMAP pixel
├── 03_full_smap_iem_data/       # Complete daily SMAP + IEM files
├── 04_feature_screening/        # Feature importance tables and figures
├── 05_gapfill_model_validation/ # Validation metrics, predictions, stacking model
│   └── stacking/
│       ├── meta_model.joblib          # Ridge stacking meta-model
│       └── meta_model_coefficients.csv  # Learned weights per base model
├── 07_gapfill_predictions/      # Base model predictions for real missing pixels
└── 08_gapfilled_final/          # ← Main output: complete daily SM grids
    ├── am/                      # AM pass files (2020–2025)
    ├── pm/                      # PM pass files (2020–2025)
    ├── gapfill_summary_by_file.csv
    └── gapfill_overall_summary.csv
```

Each final file contains one row per SMAP pixel with:
- `soil_moisture` — original observed value (NaN if missing)
- `soil_moisture_filled` — gap-filled value
- `fill_status` — `observed`, `filled`, or `unfilled`
- `fill_method` — which method filled each pixel: `observed`, `stacking`,
  `centroid_ordinary_kriging`, `regression_kriging`, `nearest_neighbor_same_day`,
  `xgboost`, `hist_gbdt`, or `random_forest`

---

## 🔮 Stage 2 — Forecasting *(TBC)*

> **Status: not yet implemented.**
> Scripts will live in `src/code/smap_forecasting/`.

**Planned approach:**
- Input: complete gap-filled daily SM grids from Stage 1 (`08_gapfilled_final/`)
- Target: predict soil moisture 1–7 days ahead at SMAP pixel level
- Methods under consideration: LSTM, temporal CNN, linear AR baseline
- Features: lagged SM values, IEM weather variables, day-of-year encoding
- Validation: train 2020–2023, validate 2024, test 2025

---

## 🗺️ Stage 3 — Township Downscaling *(TBC)*

> **Status: not yet implemented.**
> Scripts will live in `src/code/township_kriging/`.

**Planned approach:**
- Input: pixel-level soil moisture forecasts from Stage 2
- Target: produce soil moisture estimates for Iowa civil townships (~1,700 units)
- Method: area-to-area kriging (atakrig) from SMAP pixels to township polygons
- Output: daily township-level SM forecast with uncertainty estimates

---

## 📦 Environment

### Python, R, and core dependencies

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate py312
```
The `atakrig` R package is installed separately from R/CRAN.
```bash
Rscript -e 'install.packages("atakrig", repos = "https://cloud.r-project.org")'
```
**Key Python packages:** geopandas, pykrige, scikit-learn, xgboost, statsmodels,
pandas, numpy, joblib, matplotlib, pyproj, netCDF4, pyarrow

**Key R packages:** data.table, gstat, sf, ggplot2

---

## 📩 Contact

Maintainer: Armaghan Alaedini

Email: alaedini@iastate.edu