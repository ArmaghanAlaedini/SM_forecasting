#!/usr/bin/env Rscript

# 11b_generate_interpolation_gapfill_predictions.R
#
# Predict REAL original missing SMAP pixels using same-day observed SMAP pixels.
#
# Methods:
#   centroid_ordinary_kriging   -- spatial detrend (x/y polynomial) + krige residuals + add trend
#   nearest_neighbor_same_day
#   regression_kriging          -- IEM covariate trend + krige residuals + add trend
#
# Both kriging methods detrend before kriging. They differ in the trend model:
#   centroid OK uses a spatial x/y polynomial trend.
#   regression kriging uses IEM weather covariates as the trend.
#
# Output:
#   src/data/processed/smap_gap_filling/07_gapfill_predictions/interpolation/

suppressPackageStartupMessages({
  library(data.table)
  library(sp)
  library(gstat)
})


# ============================================================
# PROJECT ROOT
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


# ============================================================
# SETTINGS
# ============================================================

INPUT_DIR <- file.path(PROJECT_ROOT, "src/data/processed/smap_gap_filling/03_full_smap_iem_data")
OUT_DIR   <- file.path(PROJECT_ROOT, "src/data/processed/smap_gap_filling/07_gapfill_predictions/interpolation")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

TARGET <- "soil_moisture"
KEY    <- "smap_pixel_key"
PASSES <- c("am", "pm")
GAPFILL_YEARS <- c(2020, 2021, 2022, 2023, 2024, 2025)

METHODS_TO_USE <- c(
  "centroid_ordinary_kriging",
  "nearest_neighbor_same_day",
  "regression_kriging"
)

MIN_OBSERVED_ROWS_PER_FILE <- 30L
NN_CHUNK_SIZE              <- 500L
CENTROID_OK_NMAX           <- 30L
CENTROID_OK_CRS            <- 3857L

# Spatial detrend control for centroid OK.
# Trend used only if OLS fit is meaningful, else plain OK (never hurts).
CENTROID_OK_DETREND      <- TRUE
CENTROID_OK_TREND_PVALUE <- 0.05
CENTROID_OK_TREND_R2     <- 0.01

RK_IEM_COVARIATES <- c(
  "soil04t_pta", "soil12vwc_pta", "soil24vwc_pta",
  "soil50vwc_pta", "precip_pta", "et_pta", "rh_pta"
)

PRED_PATH     <- file.path(OUT_DIR, "interpolation_gapfill_predictions.csv")
MANIFEST_PATH <- file.path(OUT_DIR, "interpolation_gapfill_manifest.csv")


# ============================================================
# HELPERS
# ============================================================

parse_date_from_filename <- function(path) {
  m <- regmatches(basename(path), regexpr("[0-9]{8}", basename(path)))
  if (length(m) == 0 || m == "") stop("Could not parse date from: ", path)
  as.IDate(m, format = "%Y%m%d")
}

file_id_from_path <- function(pass_name, path) paste0(pass_name, "/", basename(path))

list_complete_files <- function() {
  out <- list()
  for (pass_name in PASSES) {
    d <- file.path(INPUT_DIR, pass_name, "complete")
    if (!dir.exists(d)) stop("Missing folder: ", d)
    out[[pass_name]] <- sort(list.files(d, pattern = "\\.csv$", full.names = TRUE))
  }
  out
}

add_basic_columns <- function(dt, pass_name, path) {
  setDT(dt)
  dt <- dt[, !duplicated(names(dt)), with = FALSE]
  if ("date" %in% names(dt)) dt[, date := as.IDate(date)] else
    dt[, date := parse_date_from_filename(path)]
  dt[, year    := as.integer(format(date, "%Y"))]
  dt[, pass    := pass_name]
  dt[, file_id := file_id_from_path(pass_name, path)]
  if (!(KEY %in% names(dt))) {
    if (all(c("grid_row", "grid_col") %in% names(dt)))
      dt[, (KEY) := paste0(grid_row, "_", grid_col)]
    else
      dt[, (KEY) := as.character(seq_len(.N))]
  }
  dt[, (KEY) := as.character(get(KEY))]
  dt
}


# ============================================================
# SPATIAL TREND HELPER (shared by centroid OK detrending)
# ============================================================

make_spatial_trend_features <- function(x, y, x_mean_km, y_mean_km) {
  x0 <- (x / 1000.0) - x_mean_km
  y0 <- (y / 1000.0) - y_mean_km
  data.frame(x = x0, y = y0, x2 = x0^2, y2 = y0^2, xy = x0 * y0)
}

fit_spatial_trend <- function(x, y, z) {
  x_mean_km <- mean(x / 1000.0, na.rm = TRUE)
  y_mean_km <- mean(y / 1000.0, na.rm = TRUE)
  feats <- make_spatial_trend_features(x, y, x_mean_km, y_mean_km)
  df <- cbind(data.frame(z = z), feats)

  model <- tryCatch(lm(z ~ x + y + x2 + y2 + xy, data = df), error = function(e) NULL)
  if (is.null(model)) return(NULL)

  fstat  <- tryCatch(summary(model)$fstatistic, error = function(e) NULL)
  r2     <- tryCatch(summary(model)$r.squared,   error = function(e) NA_real_)
  pvalue <- if (!is.null(fstat))
    stats::pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE) else NA_real_

  trend_used <- is.finite(pvalue) && is.finite(r2) &&
    pvalue < CENTROID_OK_TREND_PVALUE && r2 > CENTROID_OK_TREND_R2

  list(model = model, trend_used = trend_used,
       x_mean_km = x_mean_km, y_mean_km = y_mean_km, r2 = r2, pvalue = pvalue)
}

predict_spatial_trend <- function(info, x, y) {
  if (is.null(info) || !info$trend_used) return(rep(0.0, length(x)))
  feats <- make_spatial_trend_features(x, y, info$x_mean_km, info$y_mean_km)
  as.numeric(predict(info$model, newdata = feats))
}


# ============================================================
# NEAREST NEIGHBOUR
# ============================================================

predict_nearest_neighbor <- function(observed_dt, missing_dt) {
  empty <- data.table(smap_pixel_key = missing_dt[[KEY]],
                      method = "nearest_neighbor_same_day",
                      prediction = NA_real_, nearest_distance = NA_real_)
  if (nrow(observed_dt) == 0 || nrow(missing_dt) == 0) return(empty)

  obs_x <- as.numeric(observed_dt[["x"]]); obs_y <- as.numeric(observed_dt[["y"]])
  mis_x <- as.numeric(missing_dt[["x"]]);  mis_y <- as.numeric(missing_dt[["y"]])
  n_chunks <- ceiling(nrow(missing_dt) / NN_CHUNK_SIZE)
  preds <- numeric(nrow(missing_dt)); dists <- numeric(nrow(missing_dt))

  for (chunk in seq_len(n_chunks)) {
    i1 <- (chunk - 1L) * NN_CHUNK_SIZE + 1L
    i2 <- min(chunk * NN_CHUNK_SIZE, nrow(missing_dt))
    cx <- mis_x[i1:i2]; cy <- mis_y[i1:i2]
    dx <- outer(cx, obs_x, "-"); dy <- outer(cy, obs_y, "-")
    d2 <- dx^2 + dy^2
    idx <- max.col(-d2, ties.method = "first")
    preds[i1:i2] <- as.numeric(observed_dt[[TARGET]])[idx]
    dists[i1:i2] <- sqrt(d2[cbind(seq_along(cx), idx)])
  }
  data.table(smap_pixel_key = missing_dt[[KEY]],
             method = "nearest_neighbor_same_day",
             prediction = preds, nearest_distance = dists)
}


# ============================================================
# CENTROID ORDINARY KRIGING (with spatial detrend)
# ============================================================

predict_centroid_ok <- function(observed_dt, missing_dt) {
  empty <- data.table(smap_pixel_key = missing_dt[[KEY]],
                      method = "centroid_ordinary_kriging",
                      prediction = NA_real_, kriging_variance = NA_real_)
  if (nrow(observed_dt) < 5L || nrow(missing_dt) == 0L) return(empty)

  obs_x <- as.numeric(observed_dt[["x"]]); obs_y <- as.numeric(observed_dt[["y"]])
  obs_z <- as.numeric(observed_dt[[TARGET]])
  valid <- is.finite(obs_x) & is.finite(obs_y) & is.finite(obs_z)
  if (sum(valid) < 5L) return(empty)

  obs_x <- obs_x[valid]; obs_y <- obs_y[valid]; obs_z <- obs_z[valid]
  mis_x <- as.numeric(missing_dt[["x"]]); mis_y <- as.numeric(missing_dt[["y"]])

  # --- Spatial detrend step ---
  trend_info <- NULL
  if (CENTROID_OK_DETREND) {
    trend_info <- fit_spatial_trend(obs_x, obs_y, obs_z)
  }
  obs_trend <- predict_spatial_trend(trend_info, obs_x, obs_y)
  mis_trend <- predict_spatial_trend(trend_info, mis_x, mis_y)

  obs_resid <- obs_z - obs_trend

  obs_sp <- SpatialPointsDataFrame(coords = cbind(obs_x, obs_y),
                                   data = data.frame(z = obs_resid))
  mis_sp <- SpatialPoints(cbind(mis_x, mis_y))

  tryCatch({
    vfit <- tryCatch(fit.variogram(variogram(z ~ 1, obs_sp), vgm(c("Sph", "Exp", "Gau"))),
                     error = function(e) NULL)
    if (is.null(vfit)) return(empty)
    kg <- krige(z ~ 1, obs_sp, mis_sp, model = vfit, debug.level = 0)
    data.table(smap_pixel_key = missing_dt[[KEY]],
               method = "centroid_ordinary_kriging",
               prediction = as.numeric(kg$var1.pred) + mis_trend,
               kriging_variance = as.numeric(kg$var1.var))
  }, error = function(e) { message("  Centroid OK failed: ", conditionMessage(e)); empty })
}


# ============================================================
# REGRESSION KRIGING (IEM covariate trend)
# ============================================================

predict_regression_kriging <- function(observed_dt, missing_dt) {
  empty <- data.table(smap_pixel_key = missing_dt[[KEY]],
                      method = "regression_kriging",
                      prediction = NA_real_, kriging_variance = NA_real_)
  if (nrow(observed_dt) < 10L || nrow(missing_dt) == 0L) return(empty)

  obs_df <- as.data.frame(observed_dt)
  mis_df <- as.data.frame(missing_dt)

  available_covs <- intersect(RK_IEM_COVARIATES, names(obs_df))
  available_covs <- available_covs[sapply(available_covs, function(v) {
    vals <- suppressWarnings(as.numeric(obs_df[[v]]))
    sum(is.finite(vals)) >= 8
  })]
  for (v in available_covs) {
    obs_df[[v]] <- suppressWarnings(as.numeric(obs_df[[v]]))
    mis_df[[v]] <- suppressWarnings(as.numeric(mis_df[[v]]))
  }
  obs_df[[TARGET]] <- suppressWarnings(as.numeric(obs_df[[TARGET]]))
  obs_df[["x"]] <- suppressWarnings(as.numeric(obs_df[["x"]]))
  obs_df[["y"]] <- suppressWarnings(as.numeric(obs_df[["y"]]))
  mis_df[["x"]] <- suppressWarnings(as.numeric(mis_df[["x"]]))
  mis_df[["y"]] <- suppressWarnings(as.numeric(mis_df[["y"]]))

  tryCatch({
    formula_str <- if (length(available_covs) > 0)
      paste(TARGET, "~", paste(available_covs, collapse = " + "))
    else paste(TARGET, "~ x + y")

    lm_fit <- tryCatch(lm(as.formula(formula_str), data = obs_df, na.action = na.omit),
                       error = function(e) NULL)
    if (is.null(lm_fit)) return(empty)

    obs_df$rk_residual <- NA_real_
    fitted_idx <- as.integer(rownames(model.frame(lm_fit)))
    obs_df$rk_residual[fitted_idx] <- residuals(lm_fit)
    obs_ok <- obs_df[is.finite(obs_df$rk_residual) & is.finite(obs_df$x) & is.finite(obs_df$y), ]
    if (nrow(obs_ok) < 5L) return(empty)

    obs_sp <- SpatialPointsDataFrame(coords = cbind(obs_ok$x, obs_ok$y),
                                     data = data.frame(z = obs_ok$rk_residual))
    mis_sp <- SpatialPoints(cbind(mis_df$x, mis_df$y))

    vfit <- tryCatch(fit.variogram(variogram(z ~ 1, obs_sp), vgm(c("Sph", "Exp", "Gau"))),
                     error = function(e) NULL)
    if (is.null(vfit)) return(empty)
    kg <- krige(z ~ 1, obs_sp, mis_sp, model = vfit, debug.level = 0)

    rk_trend <- tryCatch(predict(lm_fit, newdata = mis_df),
                         error = function(e) rep(mean(obs_df[[TARGET]], na.rm = TRUE), nrow(mis_df)))

    data.table(smap_pixel_key = missing_dt[[KEY]],
               method = "regression_kriging",
               prediction = as.numeric(rk_trend) + as.numeric(kg$var1.pred),
               kriging_variance = as.numeric(kg$var1.var))
  }, error = function(e) { message("  Regression kriging failed: ", conditionMessage(e)); empty })
}


# ============================================================
# PROCESS ONE FILE
# ============================================================

process_one_file <- function(pass_name, path) {
  date <- parse_date_from_filename(path)
  year <- as.integer(format(date, "%Y"))
  if (!(year %in% GAPFILL_YEARS)) return(NULL)

  dt <- tryCatch(fread(path, showProgress = FALSE), error = function(e) NULL)
  if (is.null(dt) || nrow(dt) == 0) return(NULL)

  dt <- add_basic_columns(dt, pass_name, path)
  dt[, (TARGET) := suppressWarnings(as.numeric(get(TARGET)))]

  fid      <- file_id_from_path(pass_name, path)
  observed <- dt[!is.na(get(TARGET))]
  missing  <- dt[is.na(get(TARGET))]

  manifest_row <- list(
    file_id = fid, date = as.character(date), year = year,
    pass = pass_name, source_file = path,
    n_rows = nrow(dt), n_observed = nrow(observed), n_missing = nrow(missing),
    methods_run = paste(METHODS_TO_USE, collapse = ";"),
    status = "ok", message = ""
  )

  if (nrow(missing) == 0) return(list(preds = NULL, manifest = manifest_row))
  if (nrow(observed) < MIN_OBSERVED_ROWS_PER_FILE) {
    manifest_row$status  <- "skipped_too_few_observed"
    manifest_row$message <- paste("Only", nrow(observed), "observed rows")
    return(list(preds = NULL, manifest = manifest_row))
  }

  parts <- list()
  if ("nearest_neighbor_same_day" %in% METHODS_TO_USE) {
    nn <- predict_nearest_neighbor(observed, missing)
    nn[, file_id := fid][, date := as.character(date)][, year := year][, pass := pass_name]
    parts[["nn"]] <- nn
  }
  if ("centroid_ordinary_kriging" %in% METHODS_TO_USE) {
    ok <- predict_centroid_ok(observed, missing)
    ok[, file_id := fid][, date := as.character(date)][, year := year][, pass := pass_name]
    parts[["ok"]] <- ok
  }
  if ("regression_kriging" %in% METHODS_TO_USE) {
    rk <- predict_regression_kriging(observed, missing)
    rk[, file_id := fid][, date := as.character(date)][, year := year][, pass := pass_name]
    parts[["rk"]] <- rk
  }

  if (length(parts) == 0) return(list(preds = NULL, manifest = manifest_row))
  list(preds = rbindlist(parts, fill = TRUE), manifest = manifest_row)
}


# ============================================================
# MAIN
# ============================================================

main <- function() {
  message("11b: Generate interpolation gap-fill predictions")
  message(strrep("=", 70))
  message("Methods: ", paste(METHODS_TO_USE, collapse = ", "))
  message("Centroid OK spatial detrend: ", CENTROID_OK_DETREND)
  message("Years:   ", paste(GAPFILL_YEARS, collapse = ", "))
  message(strrep("=", 70))

  all_files <- list_complete_files()
  header    <- data.table(file_id = character(), date = character(), year = integer(),
                          pass = character(), smap_pixel_key = character(),
                          method = character(), prediction = numeric(),
                          kriging_variance = numeric(), nearest_distance = numeric())
  fwrite(header, PRED_PATH)

  manifest_rows <- list()
  counter <- 0L
  total   <- sum(lengths(all_files))

  for (pass_name in PASSES) {
    message("\nProcessing ", toupper(pass_name), " (", length(all_files[[pass_name]]), " files)")
    for (path in all_files[[pass_name]]) {
      counter <- counter + 1L
      result  <- tryCatch(process_one_file(pass_name, path), error = function(e) {
        message("  FAILED: ", basename(path), " | ", conditionMessage(e))
        list(preds = NULL, manifest = list(
          file_id = file_id_from_path(pass_name, path),
          date = NA_character_, year = NA_integer_, pass = pass_name,
          source_file = path, n_rows = NA_integer_, n_observed = NA_integer_,
          n_missing = NA_integer_, methods_run = paste(METHODS_TO_USE, collapse = ";"),
          status = "failed", message = conditionMessage(e)
        ))
      })
      if (!is.null(result$preds) && nrow(result$preds) > 0)
        fwrite(result$preds, PRED_PATH, append = TRUE)
      manifest_rows[[counter]] <- result$manifest
      if (counter %% 100 == 0 || counter == total)
        message("  processed ", counter, "/", total)
    }
  }

  manifest_dt <- rbindlist(manifest_rows, fill = TRUE)
  fwrite(manifest_dt, MANIFEST_PATH)
  message("\nSaved: ", PRED_PATH)
  message("Saved: ", MANIFEST_PATH)
  message("\nStatus counts:")
  print(manifest_dt[, .N, by = status])
  message("\nDone.")
}

main()