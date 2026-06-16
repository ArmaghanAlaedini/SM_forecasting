#!/usr/bin/env Rscript

# 10b_interpolation_validation.R
#
# Interpolation / geostatistical validation for SMAP gap filling.
#
# Methods (all run by default):
#   nearest_neighbor_same_day
#   centroid_ordinary_kriging   -- spatial detrend (x/y polynomial) + krige residuals + add trend
#   regression_kriging          -- IEM covariate trend + krige residuals + add trend
#
# Both kriging methods detrend before kriging. They differ in the trend model:
#   centroid OK uses a spatial x/y polynomial trend.
#   regression kriging uses IEM weather covariates as the trend.
#
# Input:
#   03_full_smap_iem_data/{am,pm}/complete/*.csv
#
# Output:
#   05_gapfill_model_validation/interpolation/

suppressPackageStartupMessages({
  library(data.table)
})

# ============================================================
# USER SETTINGS
# ============================================================

TARGET <- "soil_moisture"
PASSES_TO_USE <- c("am", "pm")
VALIDATION_YEARS <- c(2024)
TEST_YEARS <- c(2025)
RUN_TEST <- FALSE
MAX_FILES_PER_SPLIT_PER_PASS <- NULL
HOLDOUT_MODES <- c("random_cell", "spatial_block")
MAX_OBSERVED_ROWS_PER_FILE <- 800
MAX_EVAL_TARGET_ROWS_PER_MODE <- 120000
EVAL_HOLDOUT_FRACTION <- 0.25
MIN_DONOR_ROWS <- 5
BLOCK_ATTEMPTS_PER_GROUP <- 60
RANDOM_STATE <- 42

# Centroid OK / Regression Kriging settings
CENTROID_OK_NMAX <- 30
CENTROID_OK_CRS  <- 3857

# Spatial detrend control for centroid OK.
# Trend is used only if the OLS fit is statistically meaningful
# (F-test p < threshold AND R2 > threshold), otherwise plain OK is used.
# This guarantees detrending never hurts: worst case falls back to plain OK.
CENTROID_OK_DETREND      <- TRUE
CENTROID_OK_TREND_PVALUE <- 0.05
CENTROID_OK_TREND_R2     <- 0.01

# IEM covariate columns to use in regression kriging trend model
RK_IEM_COVARIATES <- c(
  "soil04t_pta", "soil12vwc_pta", "soil24vwc_pta",
  "soil50vwc_pta", "precip_pta", "et_pta", "rh_pta"
)

# ATA settings (keep FALSE unless explicitly testing)
RUN_ATA_OK        <- FALSE
RUN_ATA_COKRIGING <- FALSE
ATA_COKRIGING_AUX <- c("soil12vwc_pta", "soil24vwc_pta", "soil50vwc_pta")

SAVE_PREDICTION_SAMPLE_ROWS <- 250000


# ============================================================
# PATHS
# ============================================================

find_project_root <- function() {
  env_root <- Sys.getenv("SMAP_PROJECT_ROOT", unset = "")
  if (nzchar(env_root)) return(normalizePath(env_root, mustWork = TRUE))
  p <- normalizePath(getwd(), mustWork = TRUE)
  for (i in 1:6) {
    if (dir.exists(file.path(p, "src")) &&
        (file.exists(file.path(p, ".git")) ||
         dir.exists(file.path(p, "renv")) ||
         file.exists(file.path(p, "environment.yml")))) {
      return(normalizePath(p, mustWork = TRUE))
    }
    p <- dirname(p)
  }
  stop("Could not find project root. Set SMAP_PROJECT_ROOT.")
}

PROJECT_ROOT <- find_project_root()
message("Project root: ", PROJECT_ROOT)

FULL_DIR <- file.path(PROJECT_ROOT, "src/data/processed/smap_gap_filling/03_full_smap_iem_data")
OUT_DIR  <- file.path(PROJECT_ROOT, "src/data/processed/smap_gap_filling/05_gapfill_model_validation/interpolation")
FIG_DIR  <- file.path(OUT_DIR, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)


# ============================================================
# FILE HELPERS
# ============================================================

parse_date_from_filename <- function(path) {
  m <- regmatches(basename(path), regexpr("[0-9]{8}", basename(path)))
  if (length(m) == 0 || m == "") stop("Could not parse date from: ", path)
  as.IDate(m, format = "%Y%m%d")
}

year_to_split <- function(year) {
  if (year %in% VALIDATION_YEARS) return("validation")
  if (year %in% TEST_YEARS)       return("test")
  return("unused")
}

build_manifest <- function() {
  rows <- list()
  for (pass_name in PASSES_TO_USE) {
    d <- file.path(FULL_DIR, pass_name, "complete")
    if (!dir.exists(d)) next
    files <- sort(list.files(d, pattern = "\\.csv$", full.names = TRUE))
    for (path in files) {
      date  <- parse_date_from_filename(path)
      year  <- as.integer(format(date, "%Y"))
      split <- year_to_split(year)
      if (split == "unused") next
      rows[[length(rows) + 1]] <- list(
        path = path, date = as.character(date),
        year = year, pass = pass_name, split = split
      )
    }
  }
  dt <- rbindlist(rows)
  if (nrow(dt) == 0) stop("No files found in ", FULL_DIR)
  if (!is.null(MAX_FILES_PER_SPLIT_PER_PASS))
    dt <- dt[, .SD[seq_len(min(.N, MAX_FILES_PER_SPLIT_PER_PASS))], by = .(split, pass)]
  dt
}

add_basic_columns <- function(dt, pass_name, path) {
  setDT(dt)
  dt <- dt[, !duplicated(names(dt)), with = FALSE]
  if ("date" %in% names(dt)) dt[, date := as.IDate(date)] else
    dt[, date := parse_date_from_filename(path)]
  dt[, year        := as.integer(format(date, "%Y"))]
  dt[, month       := as.integer(format(date, "%m"))]
  dt[, day_of_year := as.integer(format(date, "%j"))]
  dt[, pass        := pass_name]
  if (!("smap_pixel_key" %in% names(dt))) {
    if (all(c("grid_row", "grid_col") %in% names(dt)))
      dt[, smap_pixel_key := paste0(grid_row, "_", grid_col)]
    else
      dt[, smap_pixel_key := as.character(seq_len(.N))]
  }
  dt
}


# ============================================================
# HOLDOUT MARKING
# ============================================================

mark_random_cell_holdouts <- function(dt) {
  obs <- dt[!is.na(get(TARGET))]
  n_target <- max(1L, round(nrow(obs) * EVAL_HOLDOUT_FRACTION))
  obs[, eval_role := "donor"]
  obs[sample(nrow(obs), n_target), eval_role := "target"]
  obs
}

mark_spatial_block_holdouts <- function(dt) {
  obs <- dt[!is.na(get(TARGET)) & is.finite(x) & is.finite(y)]
  if (nrow(obs) < (MIN_DONOR_ROWS + 1L)) return(obs[, eval_role := "donor"])
  obs[, eval_role := "donor"]
  n_target <- max(1L, round(nrow(obs) * EVAL_HOLDOUT_FRACTION))
  chosen <- integer(0)
  for (attempt in seq_len(BLOCK_ATTEMPTS_PER_GROUP)) {
    if (length(chosen) >= n_target) break
    cx <- sample(obs$x, 1); cy <- sample(obs$y, 1)
    radius <- sample(seq(50000, 200000, by = 25000), 1)
    dist2  <- (obs$x - cx)^2 + (obs$y - cy)^2
    cands  <- which(dist2 <= radius^2 & obs$eval_role == "donor")
    if (length(cands) == 0) next
    obs[cands, eval_role := "target"]
    chosen <- c(chosen, cands)
  }
  if (sum(obs$eval_role == "donor") < MIN_DONOR_ROWS)
    obs[eval_role == "target", eval_role := "donor"]
  obs
}

mark_holdouts <- function(dt, holdout_mode) {
  if (holdout_mode == "random_cell")   return(mark_random_cell_holdouts(dt))
  if (holdout_mode == "spatial_block") return(mark_spatial_block_holdouts(dt))
  stop("Unknown holdout_mode: ", holdout_mode)
}


# ============================================================
# SPATIAL TREND HELPER (shared by centroid OK detrending)
# ============================================================

# Build centered quadratic spatial features in kilometers.
make_spatial_trend_features <- function(x, y, x_mean_km, y_mean_km) {
  x0 <- (x / 1000.0) - x_mean_km
  y0 <- (y / 1000.0) - y_mean_km
  data.frame(x = x0, y = y0, x2 = x0^2, y2 = y0^2, xy = x0 * y0)
}

# Fit spatial trend; return NULL info if trend is not meaningful.
fit_spatial_trend <- function(x, y, z) {
  x_mean_km <- mean(x / 1000.0, na.rm = TRUE)
  y_mean_km <- mean(y / 1000.0, na.rm = TRUE)
  feats <- make_spatial_trend_features(x, y, x_mean_km, y_mean_km)
  df <- cbind(data.frame(z = z), feats)

  model <- tryCatch(lm(z ~ x + y + x2 + y2 + xy, data = df), error = function(e) NULL)
  if (is.null(model)) return(NULL)

  fstat   <- tryCatch(summary(model)$fstatistic, error = function(e) NULL)
  r2      <- tryCatch(summary(model)$r.squared,   error = function(e) NA_real_)
  pvalue  <- if (!is.null(fstat))
    stats::pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE) else NA_real_

  trend_used <- is.finite(pvalue) && is.finite(r2) &&
    pvalue < CENTROID_OK_TREND_PVALUE && r2 > CENTROID_OK_TREND_R2

  list(
    model = model, trend_used = trend_used,
    x_mean_km = x_mean_km, y_mean_km = y_mean_km,
    r2 = r2, pvalue = pvalue
  )
}

predict_spatial_trend <- function(info, x, y) {
  if (is.null(info) || !info$trend_used)
    return(rep(0.0, length(x)))  # zero trend → residuals == original values
  feats <- make_spatial_trend_features(x, y, info$x_mean_km, info$y_mean_km)
  as.numeric(predict(info$model, newdata = feats))
}


# ============================================================
# PREDICTION: NEAREST NEIGHBOR
# ============================================================

predict_nearest_neighbor <- function(eval_dt) {
  donors  <- eval_dt[eval_role == "donor"  & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  targets <- eval_dt[eval_role == "target" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  if (nrow(donors) < MIN_DONOR_ROWS || nrow(targets) == 0) return(NULL)
  dx <- outer(targets$x, donors$x, "-")
  dy <- outer(targets$y, donors$y, "-")
  d2 <- dx^2 + dy^2
  nearest <- max.col(-d2, ties.method = "first")
  data.table(
    date = targets$date, pass = targets$pass,
    smap_pixel_key = targets$smap_pixel_key,
    observed = targets[[TARGET]],
    prediction = donors[[TARGET]][nearest],
    method = "nearest_neighbor_same_day",
    nearest_distance_m = sqrt(d2[cbind(seq_len(nrow(targets)), nearest)])
  )
}


# ============================================================
# PREDICTION: CENTROID ORDINARY KRIGING (with spatial detrend)
# ============================================================

predict_centroid_ok <- function(eval_dt) {
  if (!requireNamespace("sf", quietly = TRUE) ||
      !requireNamespace("gstat", quietly = TRUE)) {
    message("[skip] centroid_ok needs sf, gstat")
    return(NULL)
  }
  donors  <- eval_dt[eval_role == "donor"  & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  targets <- eval_dt[eval_role == "target" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  if (nrow(donors) < 20 || nrow(targets) == 0) return(NULL)

  donors_df  <- as.data.frame(donors)
  targets_df <- as.data.frame(targets)

  donor_x <- donors_df$x; donor_y <- donors_df$y
  donor_z <- as.numeric(donors_df[[TARGET]])
  target_x <- targets_df$x; target_y <- targets_df$y

  # --- Spatial detrend step ---
  trend_info <- NULL
  if (CENTROID_OK_DETREND) {
    trend_info <- fit_spatial_trend(donor_x, donor_y, donor_z)
  }
  donor_trend  <- predict_spatial_trend(trend_info, donor_x,  donor_y)
  target_trend <- predict_spatial_trend(trend_info, target_x, target_y)

  donor_resid <- donor_z - donor_trend

  donors_sf <- sf::st_as_sf(
    data.frame(z = donor_resid, x = donor_x, y = donor_y),
    coords = c("x", "y"), crs = CENTROID_OK_CRS, remove = FALSE
  )
  targets_sf <- sf::st_as_sf(
    targets_df, coords = c("x", "y"), crs = CENTROID_OK_CRS, remove = FALSE
  )

  tryCatch({
    vg_emp <- gstat::variogram(z ~ 1, donors_sf)
    vg_fit <- tryCatch(
      gstat::fit.variogram(vg_emp, gstat::vgm(stats::var(donors_sf$z, na.rm = TRUE), "Sph", 50000, 0.001)),
      error = function(e) gstat::vgm(stats::var(donors_sf$z, na.rm = TRUE), "Sph", 50000, 0.001)
    )
    kriged <- gstat::krige(z ~ 1, donors_sf, targets_sf, model = vg_fit,
                           nmax = CENTROID_OK_NMAX, debug.level = 0)

    # Add the spatial trend back
    final_pred <- as.numeric(kriged$var1.pred) + target_trend

    data.table(
      date = targets$date, pass = targets$pass,
      smap_pixel_key = targets$smap_pixel_key,
      observed = targets[[TARGET]],
      prediction = final_pred,
      method = "centroid_ordinary_kriging",
      nearest_distance_m = NA_real_
    )
  }, error = function(e) {
    message("[centroid_ok failed] ", conditionMessage(e))
    NULL
  })
}


# ============================================================
# PREDICTION: REGRESSION KRIGING (IEM covariate trend)
# ============================================================

predict_regression_kriging <- function(eval_dt) {
  if (!requireNamespace("sf", quietly = TRUE) ||
      !requireNamespace("gstat", quietly = TRUE)) {
    message("[skip] regression_kriging needs sf, gstat")
    return(NULL)
  }
  donors  <- eval_dt[eval_role == "donor"  & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  targets <- eval_dt[eval_role == "target" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  if (nrow(donors) < 20 || nrow(targets) == 0) return(NULL)

  obs_df <- as.data.frame(donors)
  mis_df <- as.data.frame(targets)

  available_covs <- intersect(RK_IEM_COVARIATES, names(obs_df))
  available_covs <- available_covs[sapply(available_covs, function(v) {
    vals <- suppressWarnings(as.numeric(obs_df[[v]]))
    sum(is.finite(vals)) >= 10
  })]
  for (v in available_covs) {
    obs_df[[v]] <- suppressWarnings(as.numeric(obs_df[[v]]))
    mis_df[[v]] <- suppressWarnings(as.numeric(mis_df[[v]]))
  }
  obs_df[[TARGET]] <- suppressWarnings(as.numeric(obs_df[[TARGET]]))

  tryCatch({
    formula_str <- if (length(available_covs) > 0)
      paste(TARGET, "~", paste(available_covs, collapse = " + "))
    else paste(TARGET, "~ x + y")

    lm_fit <- tryCatch(lm(as.formula(formula_str), data = obs_df, na.action = na.omit),
                       error = function(e) NULL)
    if (is.null(lm_fit)) return(NULL)

    obs_df$rk_residual <- NA_real_
    fitted_idx <- as.integer(rownames(model.frame(lm_fit)))
    obs_df$rk_residual[fitted_idx] <- residuals(lm_fit)
    obs_ok <- obs_df[is.finite(obs_df$rk_residual) & is.finite(obs_df$x) & is.finite(obs_df$y), ]
    if (nrow(obs_ok) < 5L) return(NULL)

    obs_sp <- sf::st_as_sf(obs_ok, coords = c("x", "y"), crs = CENTROID_OK_CRS, remove = FALSE)
    mis_sp <- sf::st_as_sf(mis_df, coords = c("x", "y"), crs = CENTROID_OK_CRS, remove = FALSE)
    names(obs_sp)[names(obs_sp) == "rk_residual"] <- "z"

    vg_emp <- gstat::variogram(z ~ 1, obs_sp)
    vg_fit <- tryCatch(
      gstat::fit.variogram(vg_emp, gstat::vgm(stats::var(obs_sp$z, na.rm = TRUE), "Sph", 50000, 0.001)),
      error = function(e) gstat::vgm(stats::var(obs_sp$z, na.rm = TRUE), "Sph", 50000, 0.001)
    )
    kriged <- gstat::krige(z ~ 1, obs_sp, mis_sp, model = vg_fit, nmax = CENTROID_OK_NMAX, debug.level = 0)

    rk_trend <- tryCatch(predict(lm_fit, newdata = mis_df),
                         error = function(e) rep(mean(obs_df[[TARGET]], na.rm = TRUE), nrow(mis_df)))

    data.table(
      date = targets$date, pass = targets$pass,
      smap_pixel_key = targets$smap_pixel_key,
      observed = targets[[TARGET]],
      prediction = as.numeric(rk_trend) + as.numeric(kriged$var1.pred),
      method = "regression_kriging",
      nearest_distance_m = NA_real_
    )
  }, error = function(e) {
    message("[regression_kriging failed] ", conditionMessage(e))
    NULL
  })
}


# ============================================================
# DISPATCH
# ============================================================

predict_all_methods <- function(eval_dt) {
  parts <- list()
  parts[[length(parts) + 1]] <- predict_nearest_neighbor(eval_dt)
  parts[[length(parts) + 1]] <- predict_centroid_ok(eval_dt)
  parts[[length(parts) + 1]] <- predict_regression_kriging(eval_dt)
  if (RUN_ATA_OK)        message("[info] ATA OK not implemented in this version")
  if (RUN_ATA_COKRIGING) message("[info] ATA cokriging not implemented in this version")
  parts <- parts[!vapply(parts, is.null, logical(1))]
  if (length(parts) == 0) return(NULL)
  rbindlist(parts, fill = TRUE)
}


# ============================================================
# METRICS
# ============================================================

compute_metrics <- function(observed, predicted) {
  mask <- is.finite(observed) & is.finite(predicted)
  if (sum(mask) < 2) return(list(rmse = NA, mae = NA, bias = NA, r2 = NA, n = sum(mask)))
  obs <- observed[mask]; pred <- predicted[mask]
  ss_res <- sum((obs - pred)^2); ss_tot <- sum((obs - mean(obs))^2)
  list(rmse = sqrt(mean((obs - pred)^2)), mae = mean(abs(obs - pred)),
       bias = mean(pred - obs),
       r2 = ifelse(ss_tot == 0, NA_real_, 1 - ss_res / ss_tot),
       n = sum(mask))
}


# ============================================================
# MAIN
# ============================================================

main <- function() {
  message("10b: Interpolation validation")
  message(strrep("=", 70))
  message("Project root: ", PROJECT_ROOT)
  message("Methods: nearest_neighbor_same_day, centroid_ordinary_kriging, regression_kriging")
  message("Centroid OK spatial detrend: ", CENTROID_OK_DETREND)
  message(strrep("=", 70))

  manifest <- build_manifest()
  message("\nFiles by split/pass:")
  print(manifest[, .N, by = .(split, pass)])

  splits_to_run <- if (RUN_TEST) c("validation", "test") else "validation"

  all_metrics <- list()
  all_preds   <- list()

  for (split_name in splits_to_run) {
    split_files <- manifest[split == split_name]
    message("\nProcessing split: ", split_name, " (", nrow(split_files), " files)")

    for (holdout_mode in HOLDOUT_MODES) {
      message("  Holdout: ", holdout_mode)
      metric_rows <- list()
      pred_rows   <- list()

      for (i in seq_len(nrow(split_files))) {
        row <- split_files[i]
        dt  <- tryCatch(fread(row$path, showProgress = FALSE), error = function(e) NULL)
        if (is.null(dt) || nrow(dt) == 0) next

        dt <- add_basic_columns(dt, row$pass, row$path)
        dt[, (TARGET) := suppressWarnings(as.numeric(get(TARGET)))]

        obs <- dt[!is.na(get(TARGET))]
        if (nrow(obs) == 0) next
        if (nrow(obs) > MAX_OBSERVED_ROWS_PER_FILE) obs <- obs[sample(.N, MAX_OBSERVED_ROWS_PER_FILE)]

        eval_dt <- mark_holdouts(obs, holdout_mode)
        targets <- eval_dt[eval_role == "target"]
        if (nrow(targets) > MAX_EVAL_TARGET_ROWS_PER_MODE) {
          keep <- sample(nrow(targets), MAX_EVAL_TARGET_ROWS_PER_MODE)
          eval_dt <- rbind(eval_dt[eval_role == "donor"], targets[keep])
        }

        preds <- predict_all_methods(eval_dt)
        if (is.null(preds) || nrow(preds) == 0) next

        for (method_name in unique(preds$method)) {
          mpreds <- preds[method == method_name]
          m <- compute_metrics(mpreds$observed, mpreds$prediction)
          metric_rows[[length(metric_rows) + 1]] <- data.table(
            split = split_name, holdout_mode = holdout_mode, method = method_name,
            rmse = m$rmse, mae = m$mae, bias = m$bias, r2 = m$r2, n = m$n
          )
        }
        preds[, split := split_name][, holdout_mode := holdout_mode]
        pred_rows[[length(pred_rows) + 1]] <- preds
      }

      if (length(metric_rows) > 0) all_metrics[[length(all_metrics) + 1]] <- rbindlist(metric_rows, fill = TRUE)
      if (length(pred_rows)   > 0) all_preds[[length(all_preds)     + 1]] <- rbindlist(pred_rows,   fill = TRUE)
    }
  }

  if (length(all_metrics) == 0) { message("No metrics computed."); return(invisible(NULL)) }

  metrics_dt <- rbindlist(all_metrics, fill = TRUE)
  avg_metrics <- metrics_dt[, .(
    rmse = mean(rmse, na.rm = TRUE), mae = mean(mae, na.rm = TRUE),
    bias = mean(bias, na.rm = TRUE), r2  = mean(r2,  na.rm = TRUE),
    n    = sum(n,    na.rm = TRUE)
  ), by = .(split, holdout_mode, method)]

  metrics_path <- file.path(OUT_DIR, "interpolation_validation_metrics.csv")
  fwrite(avg_metrics, metrics_path)
  message("\nSaved metrics: ", metrics_path)
  print(avg_metrics)

  if (length(all_preds) > 0) {
    preds_dt <- rbindlist(all_preds, fill = TRUE)
    if (!is.null(SAVE_PREDICTION_SAMPLE_ROWS) && nrow(preds_dt) > SAVE_PREDICTION_SAMPLE_ROWS)
      preds_dt <- preds_dt[sample(.N, SAVE_PREDICTION_SAMPLE_ROWS)]
    preds_path <- file.path(OUT_DIR, "interpolation_validation_predictions_sample.csv")
    fwrite(preds_dt, preds_path)
    message("Saved predictions: ", preds_path)
  }

  message("\nDone.")
}

main()