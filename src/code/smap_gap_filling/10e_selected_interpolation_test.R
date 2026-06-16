#!/usr/bin/env Rscript

# 10e_selected_interpolation_test.R
#
# Selected-method interpolation/geostatistical test on 2025 artificial SMAP gaps.
#
# This script does NOT select methods. It only tests the methods selected from
# 2024 validation / 10c:
#
#   centroid_ordinary_kriging   -- spatial detrend (x/y polynomial) + krige residuals + add trend
#   nearest_neighbor_same_day
#   regression_kriging           -- IEM covariate trend + krige residuals + add trend
#
# Both kriging methods detrend before kriging. They differ in the trend model:
#   centroid OK uses a spatial x/y polynomial trend.
#   regression kriging uses IEM weather covariates as the trend.
#
# Design:
#   Test: 2025 observed SMAP with artificial holdouts
#
# Outputs:
#   src/data/processed/smap_gap_filling/06_selected_methods_test/interpolation/

suppressPackageStartupMessages({
  library(data.table)
  library(sp)
  library(gstat)
  library(ggplot2)
})

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT <- normalizePath(getwd(), mustWork = TRUE)

INPUT_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/03_full_smap_iem_data"
)

OUT_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/06_selected_methods_test/interpolation"
)

FIG_DIR <- file.path(OUT_DIR, "figures")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

TARGET <- "soil_moisture"
KEY <- "smap_pixel_key"

PASSES <- c("am", "pm")
TEST_YEARS <- c(2025)

RANDOM_STATE <- 42L

RANDOM_CELL_HOLDOUT_FRACTION <- 0.25
SPATIAL_BLOCK_N_BINS <- 3L
MIN_HOLDOUT_ROWS <- 10L
MIN_OBSERVED_ROWS_PER_FILE <- 30L

# NULL = full run.
MAX_FILES_PER_PASS <- NULL

NN_CHUNK_SIZE <- 500L

# Spatial detrend control for centroid OK.
# Trend used only if OLS fit is meaningful (F p < threshold AND R2 > threshold),
# else plain OK (detrending never hurts).
CENTROID_OK_DETREND      <- TRUE
CENTROID_OK_TREND_PVALUE <- 0.05
CENTROID_OK_TREND_R2     <- 0.01

# IEM covariate columns used as the regression-kriging trend model.
# Only columns present with enough non-NA values are used.
RK_IEM_COVARIATES <- c(
  "soil04t_pta", "soil12vwc_pta", "soil24vwc_pta",
  "soil50vwc_pta", "precip_pta", "et_pta", "rh_pta"
)


# ============================================================
# HELPERS
# ============================================================

parse_date_from_filename <- function(path) {
  base <- basename(path)
  m <- regmatches(base, regexpr("[0-9]{8}", base))

  if (length(m) == 0 || m == "") {
    stop("Could not parse YYYYMMDD from filename: ", path)
  }

  as.IDate(m, format = "%Y%m%d")
}


list_complete_files <- function() {
  out <- list()

  for (pass_name in PASSES) {
    d <- file.path(INPUT_DIR, pass_name, "complete")

    if (!dir.exists(d)) {
      stop("Missing input folder: ", d)
    }

    files <- sort(list.files(d, pattern = "\\.csv$", full.names = TRUE))

    if (!is.null(MAX_FILES_PER_PASS)) {
      files <- head(files, MAX_FILES_PER_PASS)
    }

    out[[pass_name]] <- files
  }

  out
}


add_basic_columns <- function(dt, pass_name, path) {
  setDT(dt)
  dt <- dt[, !duplicated(names(dt)), with = FALSE]

  if ("date" %in% names(dt)) {
    dt[, date := as.IDate(date)]
  } else {
    dt[, date := parse_date_from_filename(path)]
  }

  dt[, year := as.integer(format(date, "%Y"))]
  dt[, month := as.integer(format(date, "%m"))]
  dt[, day_of_year := as.integer(format(date, "%j"))]
  dt[, pass := pass_name]

  if (!(KEY %in% names(dt))) {
    if (all(c("grid_row", "grid_col") %in% names(dt))) {
      dt[, (KEY) := paste0(grid_row, "_", grid_col)]
    } else {
      dt[, (KEY) := as.character(seq_len(.N))]
    }
  }

  dt[, (KEY) := as.character(get(KEY))]

  dt
}


read_one_file <- function(path, pass_name) {
  dt <- fread(path)
  add_basic_columns(dt, pass_name, path)
}


as_num <- function(x) {
  suppressWarnings(as.numeric(x))
}


rmse <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  if (!any(ok)) return(NA_real_)
  sqrt(mean((pred[ok] - obs[ok])^2))
}


mae <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  if (!any(ok)) return(NA_real_)
  mean(abs(pred[ok] - obs[ok]))
}


bias <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  if (!any(ok)) return(NA_real_)
  mean(pred[ok] - obs[ok])
}


r2_score <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)

  if (sum(ok) < 2) {
    return(NA_real_)
  }

  ss_res <- sum((obs[ok] - pred[ok])^2)
  ss_tot <- sum((obs[ok] - mean(obs[ok]))^2)

  if (!is.finite(ss_tot) || ss_tot <= 0) {
    return(NA_real_)
  }

  1 - ss_res / ss_tot
}


make_random_cell_holdout <- function(obs, seed) {
  n <- nrow(obs)

  if (n < MIN_OBSERVED_ROWS_PER_FILE) {
    return(integer(0))
  }

  set.seed(seed)

  k <- round(RANDOM_CELL_HOLDOUT_FRACTION * n)
  k <- max(MIN_HOLDOUT_ROWS, k)
  k <- min(k, n - 1L)

  sample(seq_len(n), size = k, replace = FALSE)
}


make_spatial_block_holdout <- function(obs, seed) {
  n <- nrow(obs)

  if (n < MIN_OBSERVED_ROWS_PER_FILE) {
    return(integer(0))
  }

  work <- copy(obs)

  if (all(c("grid_row", "grid_col") %in% names(work))) {
    row_var <- as_num(work$grid_row)
    col_var <- as_num(work$grid_col)
  } else if (all(c("y", "x") %in% names(work))) {
    row_var <- as_num(work$y)
    col_var <- as_num(work$x)
  } else {
    return(make_random_cell_holdout(obs, seed))
  }

  valid <- is.finite(row_var) & is.finite(col_var)

  work <- work[valid]
  row_var <- row_var[valid]
  col_var <- col_var[valid]

  if (nrow(work) < MIN_OBSERVED_ROWS_PER_FILE) {
    return(make_random_cell_holdout(obs, seed))
  }

  row_bin <- tryCatch(
    as.integer(cut(
      rank(row_var, ties.method = "first"),
      breaks = SPATIAL_BLOCK_N_BINS,
      labels = FALSE
    )),
    error = function(e) rep(NA_integer_, length(row_var))
  )

  col_bin <- tryCatch(
    as.integer(cut(
      rank(col_var, ties.method = "first"),
      breaks = SPATIAL_BLOCK_N_BINS,
      labels = FALSE
    )),
    error = function(e) rep(NA_integer_, length(col_var))
  )

  work[, row_bin := row_bin]
  work[, col_bin := col_bin]
  work[, original_row_id := which(valid)]

  candidates <- work[
    is.finite(row_bin) & is.finite(col_bin),
    .(rows = list(original_row_id), n = .N),
    by = .(row_bin, col_bin)
  ]

  candidates <- candidates[n >= MIN_HOLDOUT_ROWS & n < nrow(obs)]

  if (nrow(candidates) == 0) {
    return(make_random_cell_holdout(obs, seed))
  }

  set.seed(seed)
  chosen <- candidates[sample(seq_len(.N), 1L)]$rows[[1]]

  as.integer(chosen)
}


get_coord_cols <- function(dt) {
  if (all(c("x", "y") %in% names(dt))) {
    return(c("x", "y"))
  }

  if (all(c("lon", "lat") %in% names(dt))) {
    return(c("lon", "lat"))
  }

  stop("No coordinate columns found. Need either x/y or lon/lat.")
}


nearest_neighbor_predict <- function(obs, target, coord_cols) {
  ox <- as.matrix(obs[, ..coord_cols])
  tx <- as.matrix(target[, ..coord_cols])
  oy <- as_num(obs[[TARGET]])

  ox <- apply(ox, 2, as_num)
  tx <- apply(tx, 2, as_num)

  if (!is.matrix(ox)) {
    ox <- matrix(ox, ncol = length(coord_cols))
  }

  if (!is.matrix(tx)) {
    tx <- matrix(tx, ncol = length(coord_cols))
  }

  n_target <- nrow(tx)

  pred <- rep(NA_real_, n_target)
  dist_out <- rep(NA_real_, n_target)

  valid_obs <- is.finite(oy) & complete.cases(ox)
  ox <- ox[valid_obs, , drop = FALSE]
  oy <- oy[valid_obs]

  if (length(oy) == 0 || nrow(ox) == 0) {
    return(data.table(prediction = pred, nearest_distance = dist_out))
  }

  for (start in seq(1L, n_target, by = NN_CHUNK_SIZE)) {
    end <- min(start + NN_CHUNK_SIZE - 1L, n_target)
    idx <- start:end

    block <- tx[idx, , drop = FALSE]
    valid_target <- complete.cases(block)

    if (!any(valid_target)) {
      next
    }

    block_valid <- block[valid_target, , drop = FALSE]

    block_sq <- rowSums(block_valid^2)
    obs_sq <- rowSums(ox^2)

    d2 <- outer(block_sq, obs_sq, "+") - 2 * tcrossprod(block_valid, ox)
    d2[d2 < 0] <- 0

    nn <- max.col(-d2, ties.method = "first")
    local_rows <- idx[valid_target]

    pred[local_rows] <- oy[nn]
    dist_out[local_rows] <- sqrt(d2[cbind(seq_along(nn), nn)])
  }

  data.table(prediction = pred, nearest_distance = dist_out)
}


# ============================================================
# SPATIAL TREND HELPERS (shared by centroid OK detrending)
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


ordinary_kriging_predict <- function(obs, target, coord_cols) {
  out <- data.table(
    prediction = rep(NA_real_, nrow(target)),
    kriging_variance = rep(NA_real_, nrow(target)),
    ok_fallback = rep(FALSE, nrow(target))
  )

  obs <- copy(obs)
  target <- copy(target)

  obs[, (TARGET) := as_num(get(TARGET))]

  for (cc in coord_cols) {
    obs[, (cc) := as_num(get(cc))]
    target[, (cc) := as_num(get(cc))]
  }

  obs <- obs[is.finite(get(TARGET))]
  obs <- obs[complete.cases(obs[, ..coord_cols])]

  target_valid <- complete.cases(target[, ..coord_cols])

  if (nrow(obs) < MIN_OBSERVED_ROWS_PER_FILE || !any(target_valid)) {
    nn <- nearest_neighbor_predict(obs, target, coord_cols)
    out[, prediction := nn$prediction]
    out[, ok_fallback := TRUE]
    return(out)
  }

  result <- tryCatch({
    cx <- coord_cols[1]
    cy <- coord_cols[2]

    obs_x <- as_num(obs[[cx]]); obs_y <- as_num(obs[[cy]])
    obs_z <- as_num(obs[[TARGET]])
    tgt   <- target[target_valid]
    tgt_x <- as_num(tgt[[cx]]); tgt_y <- as_num(tgt[[cy]])

    # --- Spatial detrend step ---
    trend_info <- NULL
    if (CENTROID_OK_DETREND) {
      trend_info <- fit_spatial_trend(obs_x, obs_y, obs_z)
    }
    obs_trend <- predict_spatial_trend(trend_info, obs_x, obs_y)
    tgt_trend <- predict_spatial_trend(trend_info, tgt_x, tgt_y)

    obs_resid <- obs_z - obs_trend

    obs_sp <- data.frame(resid = obs_resid, x = obs_x, y = obs_y)
    target_sp <- data.frame(x = tgt_x, y = tgt_y)

    sp::coordinates(obs_sp) <- ~ x + y
    sp::coordinates(target_sp) <- ~ x + y

    v <- stats::var(obs_resid, na.rm = TRUE)
    if (!is.finite(v) || v <= 0) {
      stop("Non-positive variance in detrended residuals.")
    }

    xr <- range(obs_x, na.rm = TRUE)
    yr <- range(obs_y, na.rm = TRUE)
    diag_range <- sqrt(diff(xr)^2 + diff(yr)^2)
    if (!is.finite(diag_range) || diag_range <= 0) diag_range <- 1

    init_model <- gstat::vgm(
      psill = 0.8 * v, model = "Exp",
      range = diag_range / 3, nugget = 0.2 * v
    )

    emp <- suppressWarnings(gstat::variogram(resid ~ 1, obs_sp))
    fit <- tryCatch(
      suppressWarnings(gstat::fit.variogram(emp, init_model)),
      error = function(e) init_model,
      warning = function(w) init_model
    )

    kr <- suppressWarnings(
      gstat::krige(resid ~ 1, obs_sp, target_sp, model = fit, debug.level = 0)
    )

    list(
      ok = TRUE,
      target_rows = which(target_valid),
      pred = as_num(kr$var1.pred) + tgt_trend,   # add spatial trend back
      var = as_num(kr$var1.var)
    )
  }, error = function(e) {
    list(ok = FALSE, error = conditionMessage(e))
  })

  if (isTRUE(result$ok)) {
    out[result$target_rows, prediction := result$pred]
    out[result$target_rows, kriging_variance := result$var]
    return(out)
  }

  nn <- nearest_neighbor_predict(obs, target, coord_cols)
  out[, prediction := nn$prediction]
  out[, ok_fallback := TRUE]
  out
}


regression_kriging_predict <- function(obs, target, coord_cols) {
  out <- data.table(
    prediction = rep(NA_real_, nrow(target)),
    kriging_variance = rep(NA_real_, nrow(target)),
    ok_fallback = rep(FALSE, nrow(target))
  )

  obs <- copy(obs)
  target <- copy(target)

  cx <- coord_cols[1]
  cy <- coord_cols[2]

  obs[, (TARGET) := as_num(get(TARGET))]
  for (cc in coord_cols) {
    obs[, (cc) := as_num(get(cc))]
    target[, (cc) := as_num(get(cc))]
  }

  obs <- obs[is.finite(get(TARGET))]
  obs <- obs[complete.cases(obs[, ..coord_cols])]
  target_valid <- complete.cases(target[, ..coord_cols])

  if (nrow(obs) < MIN_OBSERVED_ROWS_PER_FILE || !any(target_valid)) {
    nn <- nearest_neighbor_predict(obs, target, coord_cols)
    out[, prediction := nn$prediction]
    out[, ok_fallback := TRUE]
    return(out)
  }

  obs_df <- as.data.frame(obs)
  tgt_df <- as.data.frame(target[target_valid])

  available_covs <- intersect(RK_IEM_COVARIATES, names(obs_df))
  available_covs <- available_covs[sapply(available_covs, function(v) {
    vals <- as_num(obs_df[[v]])
    sum(is.finite(vals)) >= 8
  })]
  for (v in available_covs) {
    obs_df[[v]] <- as_num(obs_df[[v]])
    tgt_df[[v]] <- as_num(tgt_df[[v]])
  }

  result <- tryCatch({
    formula_str <- if (length(available_covs) > 0)
      paste(TARGET, "~", paste(available_covs, collapse = " + "))
    else paste(TARGET, "~", cx, "+", cy)

    lm_fit <- lm(as.formula(formula_str), data = obs_df, na.action = na.omit)

    obs_df$rk_residual <- NA_real_
    fitted_idx <- as.integer(rownames(model.frame(lm_fit)))
    obs_df$rk_residual[fitted_idx] <- residuals(lm_fit)

    obs_ok <- obs_df[is.finite(obs_df$rk_residual) &
                     is.finite(obs_df[[cx]]) & is.finite(obs_df[[cy]]), ]
    if (nrow(obs_ok) < MIN_OBSERVED_ROWS_PER_FILE) stop("Too few residual rows.")

    obs_sp <- data.frame(resid = obs_ok$rk_residual,
                         x = obs_ok[[cx]], y = obs_ok[[cy]])
    target_sp <- data.frame(x = tgt_df[[cx]], y = tgt_df[[cy]])
    sp::coordinates(obs_sp) <- ~ x + y
    sp::coordinates(target_sp) <- ~ x + y

    v <- stats::var(obs_ok$rk_residual, na.rm = TRUE)
    if (!is.finite(v) || v <= 0) stop("Non-positive residual variance.")

    xr <- range(obs_ok[[cx]], na.rm = TRUE)
    yr <- range(obs_ok[[cy]], na.rm = TRUE)
    diag_range <- sqrt(diff(xr)^2 + diff(yr)^2)
    if (!is.finite(diag_range) || diag_range <= 0) diag_range <- 1

    init_model <- gstat::vgm(psill = 0.8 * v, model = "Exp",
                             range = diag_range / 3, nugget = 0.2 * v)
    emp <- suppressWarnings(gstat::variogram(resid ~ 1, obs_sp))
    fit <- tryCatch(
      suppressWarnings(gstat::fit.variogram(emp, init_model)),
      error = function(e) init_model, warning = function(w) init_model
    )
    kr <- suppressWarnings(
      gstat::krige(resid ~ 1, obs_sp, target_sp, model = fit, debug.level = 0)
    )

    rk_trend <- as_num(predict(lm_fit, newdata = tgt_df))

    list(ok = TRUE, target_rows = which(target_valid),
         pred = rk_trend + as_num(kr$var1.pred),
         var = as_num(kr$var1.var))
  }, error = function(e) list(ok = FALSE, error = conditionMessage(e)))

  if (isTRUE(result$ok)) {
    out[result$target_rows, prediction := result$pred]
    out[result$target_rows, kriging_variance := result$var]
    return(out)
  }

  nn <- nearest_neighbor_predict(obs, target, coord_cols)
  out[, prediction := nn$prediction]
  out[, ok_fallback := TRUE]
  out
}


make_metrics <- function(preds) {
  preds[
    ,
    .(
      rmse = rmse(observed, prediction),
      mae = mae(observed, prediction),
      bias = bias(observed, prediction),
      r2 = r2_score(observed, prediction),
      n = sum(is.finite(observed) & is.finite(prediction)),
      n_fallback = sum(ok_fallback %in% TRUE, na.rm = TRUE)
    ),
    by = .(split, holdout_mode, method)
  ][order(holdout_mode, rmse, mae)]
}


plot_metric <- function(metrics, metric, path) {
  if (nrow(metrics) == 0) {
    return(invisible(NULL))
  }

  pdt <- copy(metrics)
  pdt[, label := paste(method, holdout_mode, sep = " | ")]
  setorderv(pdt, c("holdout_mode", metric), c(1L, 1L))

  p <- ggplot(pdt, aes(x = reorder(label, get(metric)), y = get(metric))) +
    geom_col() +
    coord_flip() +
    labs(
      title = paste("Selected interpolation 2025 test", toupper(metric)),
      x = NULL,
      y = toupper(metric)
    ) +
    theme_minimal(base_size = 12)

  ggsave(path, p, width = 9, height = 5)
}


# ============================================================
# MAIN
# ============================================================

main <- function() {
  message("Selected interpolation/geostatistical 2025 test")
  message(strrep("=", 70))
  message("Project root:  ", PROJECT_ROOT)
  message("Input folder:   ", INPUT_DIR)
  message("Output folder:  ", OUT_DIR)
  message("Test years:     ", paste(TEST_YEARS, collapse = ", "))
  message(strrep("=", 70))

  files_by_pass <- list_complete_files()

  pred_parts <- list()
  manifest_parts <- list()

  total_files <- sum(lengths(files_by_pass))
  counter <- 0L

  for (pass_name in names(files_by_pass)) {
    files <- files_by_pass[[pass_name]]

    for (path in files) {
      date <- parse_date_from_filename(path)
      year <- as.integer(format(date, "%Y"))

      if (!(year %in% TEST_YEARS)) {
        next
      }

      counter <- counter + 1L
      message(sprintf("[%d/%d] test %s %s", counter, total_files, pass_name, date))

      dt <- read_one_file(path, pass_name)

      if (!(TARGET %in% names(dt))) {
        warning("Skipping file without target column: ", path)
        next
      }

      coord_cols <- get_coord_cols(dt)

      for (cc in coord_cols) {
        dt[, (cc) := as_num(get(cc))]
      }

      dt[, (TARGET) := as_num(get(TARGET))]

      obs_all <- dt[is.finite(get(TARGET))]

      if (nrow(obs_all) < MIN_OBSERVED_ROWS_PER_FILE) {
        next
      }

      seed_base <- as.integer(format(date, "%Y%m%d")) +
        ifelse(pass_name == "pm", 1L, 0L)

      for (holdout_mode in c("random_cell", "spatial_block")) {
        if (holdout_mode == "random_cell") {
          hidden_pos <- make_random_cell_holdout(
            obs_all,
            seed = seed_base + RANDOM_STATE
          )
        } else {
          hidden_pos <- make_spatial_block_holdout(
            obs_all,
            seed = seed_base + RANDOM_STATE + 10000L
          )
        }

        if (length(hidden_pos) == 0) {
          next
        }

        hidden <- obs_all[hidden_pos]
        train_obs <- obs_all[-hidden_pos]

        if (nrow(train_obs) < MIN_OBSERVED_ROWS_PER_FILE) {
          next
        }

        manifest_parts[[length(manifest_parts) + 1L]] <- data.table(
          date = as.character(date),
          pass = pass_name,
          source_file = path,
          split = "test",
          holdout_mode = holdout_mode,
          n_rows_file = nrow(dt),
          n_observed_file = nrow(obs_all),
          n_train_same_day = nrow(train_obs),
          n_hidden_test = nrow(hidden)
        )

        nn <- nearest_neighbor_predict(train_obs, hidden, coord_cols)

        nn_out <- data.table(
          split = "test",
          holdout_mode = holdout_mode,
          date = as.character(date),
          pass = pass_name,
          smap_pixel_key = hidden[[KEY]],
          method = "nearest_neighbor_same_day",
          observed = hidden[[TARGET]],
          prediction = nn$prediction,
          nearest_distance = nn$nearest_distance,
          kriging_variance = NA_real_,
          ok_fallback = FALSE,
          source_file = path
        )

        pred_parts[[length(pred_parts) + 1L]] <- nn_out

        ok <- ordinary_kriging_predict(train_obs, hidden, coord_cols)

        ok_out <- data.table(
          split = "test",
          holdout_mode = holdout_mode,
          date = as.character(date),
          pass = pass_name,
          smap_pixel_key = hidden[[KEY]],
          method = "centroid_ordinary_kriging",
          observed = hidden[[TARGET]],
          prediction = ok$prediction,
          nearest_distance = NA_real_,
          kriging_variance = ok$kriging_variance,
          ok_fallback = ok$ok_fallback,
          source_file = path
        )

        pred_parts[[length(pred_parts) + 1L]] <- ok_out

        rk <- regression_kriging_predict(train_obs, hidden, coord_cols)

        rk_out <- data.table(
          split = "test",
          holdout_mode = holdout_mode,
          date = as.character(date),
          pass = pass_name,
          smap_pixel_key = hidden[[KEY]],
          method = "regression_kriging",
          observed = hidden[[TARGET]],
          prediction = rk$prediction,
          nearest_distance = NA_real_,
          kriging_variance = rk$kriging_variance,
          ok_fallback = rk$ok_fallback,
          source_file = path
        )

        pred_parts[[length(pred_parts) + 1L]] <- rk_out
      }
    }
  }

  if (length(pred_parts) == 0) {
    stop("No interpolation/geostatistical test predictions were created.")
  }

  preds <- rbindlist(pred_parts, fill = TRUE)
  manifest <- rbindlist(manifest_parts, fill = TRUE)

  metrics <- make_metrics(preds)

  metrics_path <- file.path(OUT_DIR, "interpolation_selected_test_metrics.csv")
  preds_path <- file.path(OUT_DIR, "interpolation_selected_test_predictions.csv")
  manifest_path <- file.path(OUT_DIR, "interpolation_selected_test_manifest.csv")

  fwrite(metrics, metrics_path)
  fwrite(preds, preds_path)
  fwrite(manifest, manifest_path)

  plot_metric(metrics, "rmse", file.path(FIG_DIR, "interp_selected_test_rmse.pdf"))
  plot_metric(metrics, "bias", file.path(FIG_DIR, "interp_selected_test_bias.pdf"))

  message("\nSaved:")
  message("  ", metrics_path)
  message("  ", preds_path)
  message("  ", manifest_path)
  message("  ", FIG_DIR)

  message("\nSelected interpolation/geostatistical 2025 test results:")
  print(metrics)

  message("\nDone.")
}


main()