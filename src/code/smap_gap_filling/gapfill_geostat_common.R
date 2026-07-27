#!/usr/bin/env Rscript

# Shared geostatistical functions for 10b, 10e, and 11b.
# All three scripts use the same settings and prediction functions so that
# validation, independent testing, and final gap filling are comparable.

suppressPackageStartupMessages({
  library(data.table)
  library(sp)
  library(gstat)
})

RANDOM_SEED <- 1234L
TARGET <- "soil_moisture"
KEY <- "smap_pixel_key"
PASSES <- c("am", "pm")

MIN_DONOR_ROWS <- 20L
MIN_NN_DONORS <- 1L
CENTROID_OK_NMAX <- 30L
CENTROID_OK_DETREND <- TRUE
CENTROID_OK_TREND_PVALUE <- 0.05
CENTROID_OK_TREND_R2 <- 0.01
VARIOGRAM_MODEL <- "Sph"
VARIOGRAM_INITIAL_RANGE_M <- 50000.0
VARIOGRAM_NUGGET_FRACTION <- 0.20

RK_IEM_COVARIATES <- c(
  "soil04t_pta",
  "soil12vwc_pta",
  "soil24vwc_pta",
  "soil50vwc_pta",
  "precip_pta",
  "et_pta",
  "rh_pta"
)
RK_MAX_COVARIATES <- 3L
RK_MIN_FINITE_PER_COVARIATE <- 10L
RK_MIN_OBS_PER_PARAMETER <- 5L
NN_CHUNK_SIZE <- 500L

SELECTED_INTERPOLATION_METHODS <- c(
  "centroid_ordinary_kriging",
  "nearest_neighbor_same_day",
  "regression_kriging"
)


find_project_root <- function() {
  env_root <- Sys.getenv("SMAP_PROJECT_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, mustWork = TRUE))
  }

  script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(script_arg) > 0L) {
    p <- dirname(normalizePath(sub("^--file=", "", script_arg[1]), mustWork = TRUE))
  } else {
    p <- normalizePath(getwd(), mustWork = TRUE)
  }

  for (i in seq_len(8L)) {
    marker <- file.exists(file.path(p, ".git")) ||
      dir.exists(file.path(p, "renv")) ||
      file.exists(file.path(p, "environment.yml"))
    if (dir.exists(file.path(p, "src")) && marker) {
      return(normalizePath(p, mustWork = TRUE))
    }
    parent <- dirname(p)
    if (identical(parent, p)) break
    p <- parent
  }
  stop("Could not find project root. Set SMAP_PROJECT_ROOT.")
}


find_script_dir <- function() {
  script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(script_arg) > 0L) {
    return(dirname(normalizePath(sub("^--file=", "", script_arg[1]), mustWork = TRUE)))
  }
  normalizePath(getwd(), mustWork = TRUE)
}


parse_date_from_filename <- function(path) {
  base <- basename(path)
  m <- regmatches(base, regexpr("[0-9]{8}", base))
  if (length(m) == 0L || identical(m, "")) {
    stop("Could not parse YYYYMMDD from filename: ", path)
  }
  as.IDate(m, format = "%Y%m%d")
}


file_id_from_path <- function(pass_name, path) {
  paste0(tolower(pass_name), "/", basename(path))
}


as_num <- function(x) suppressWarnings(as.numeric(x))


add_basic_columns <- function(dt, pass_name, path) {
  setDT(dt)
  dt <- dt[, !duplicated(names(dt)), with = FALSE]

  if ("date" %in% names(dt)) {
    parsed <- as.IDate(dt$date)
    parsed[is.na(parsed)] <- parse_date_from_filename(path)
    dt[, date := parsed]
  } else {
    dt[, date := parse_date_from_filename(path)]
  }

  dt[, year := as.integer(format(date, "%Y"))]
  dt[, month := as.integer(format(date, "%m"))]
  dt[, day_of_year := as.integer(format(date, "%j"))]
  dt[, sin_doy := sin(2.0 * pi * day_of_year / 366.0)]
  dt[, cos_doy := cos(2.0 * pi * day_of_year / 366.0)]
  dt[, pass := tolower(pass_name)]
  dt[, pass_pm := as.integer(pass == "pm")]
  dt[, file_id := file_id_from_path(pass_name, path)]

  if (!(KEY %in% names(dt))) {
    if (all(c("grid_row", "grid_col") %in% names(dt))) {
      dt[, (KEY) := paste0(round(as_num(grid_row)), "_", round(as_num(grid_col)))]
    } else if (all(c("x", "y") %in% names(dt))) {
      dt[, (KEY) := paste0(round(as_num(x)), "_", round(as_num(y)))]
    } else {
      stop("No smap_pixel_key, grid_row/grid_col, or x/y in ", path)
    }
  }
  dt[, (KEY) := as.character(get(KEY))]

  for (column in intersect(c(TARGET, "x", "y", "grid_row", "grid_col"), names(dt))) {
    set(dt, j = column, value = as_num(dt[[column]]))
  }
  dt
}


compute_metrics <- function(observed, predicted) {
  observed <- as_num(observed)
  predicted <- as_num(predicted)
  valid <- is.finite(observed) & is.finite(predicted)
  n <- sum(valid)
  if (n == 0L) {
    return(list(rmse = NA_real_, mae = NA_real_, bias = NA_real_, r2 = NA_real_, n = 0L))
  }
  obs <- observed[valid]
  pred <- predicted[valid]
  ss_res <- sum((obs - pred)^2)
  ss_tot <- sum((obs - mean(obs))^2)
  list(
    rmse = sqrt(mean((obs - pred)^2)),
    mae = mean(abs(obs - pred)),
    bias = mean(pred - obs),
    r2 = if (n >= 2L && is.finite(ss_tot) && ss_tot > 0) 1 - ss_res / ss_tot else NA_real_,
    n = n
  )
}


make_spatial_trend_features <- function(x, y, x_mean_km, y_mean_km) {
  x0 <- (as_num(x) / 1000.0) - x_mean_km
  y0 <- (as_num(y) / 1000.0) - y_mean_km
  data.frame(x = x0, y = y0, x2 = x0^2, y2 = y0^2, xy = x0 * y0)
}


fit_spatial_trend <- function(x, y, z) {
  x <- as_num(x); y <- as_num(y); z <- as_num(z)
  valid <- is.finite(x) & is.finite(y) & is.finite(z)
  if (sum(valid) < MIN_DONOR_ROWS) return(NULL)
  x <- x[valid]; y <- y[valid]; z <- z[valid]

  x_mean_km <- mean(x / 1000.0)
  y_mean_km <- mean(y / 1000.0)
  features <- make_spatial_trend_features(x, y, x_mean_km, y_mean_km)
  model <- tryCatch(
    lm(z ~ x + y + x2 + y2 + xy, data = cbind(data.frame(z = z), features)),
    error = function(e) NULL
  )
  if (is.null(model)) return(NULL)

  model_summary <- summary(model)
  fstat <- model_summary$fstatistic
  pvalue <- if (!is.null(fstat)) {
    stats::pf(fstat[1], fstat[2], fstat[3], lower.tail = FALSE)
  } else {
    NA_real_
  }
  r2 <- model_summary$r.squared
  trend_used <- is.finite(pvalue) && is.finite(r2) &&
    pvalue < CENTROID_OK_TREND_PVALUE && r2 > CENTROID_OK_TREND_R2

  list(
    model = model,
    trend_used = trend_used,
    x_mean_km = x_mean_km,
    y_mean_km = y_mean_km,
    pvalue = pvalue,
    r2 = r2
  )
}


predict_spatial_trend <- function(info, x, y) {
  if (is.null(info) || !isTRUE(info$trend_used)) {
    return(rep(0.0, length(x)))
  }
  features <- make_spatial_trend_features(x, y, info$x_mean_km, info$y_mean_km)
  as_num(predict(info$model, newdata = features))
}


fit_residual_variogram <- function(obs_sp, residuals) {
  variance <- stats::var(residuals, na.rm = TRUE)
  if (!is.finite(variance) || variance <= 0) {
    stop("Residual variance is non-positive.")
  }
  initial <- gstat::vgm(
    psill = max((1.0 - VARIOGRAM_NUGGET_FRACTION) * variance, .Machine$double.eps),
    model = VARIOGRAM_MODEL,
    range = VARIOGRAM_INITIAL_RANGE_M,
    nugget = max(VARIOGRAM_NUGGET_FRACTION * variance, 0)
  )
  empirical <- suppressWarnings(gstat::variogram(resid ~ 1, obs_sp))
  if (nrow(empirical) < 2L) return(initial)
  fitted <- tryCatch(
    suppressWarnings(gstat::fit.variogram(empirical, initial)),
    error = function(e) initial,
    warning = function(w) initial
  )
  if (any(!is.finite(fitted$psill)) || any(fitted$psill < 0)) initial else fitted
}


empty_method_result <- function(target_dt, method, status) {
  data.table(
    row_id = target_dt$row_id,
    method = method,
    prediction = NA_real_,
    kriging_variance = NA_real_,
    nearest_distance_m = NA_real_,
    prediction_status = status
  )
}


predict_nearest_neighbor <- function(observed_dt, target_dt) {
  method <- "nearest_neighbor_same_day"
  donors <- copy(observed_dt)[
    is.finite(as_num(get(TARGET))) & is.finite(as_num(x)) & is.finite(as_num(y))
  ]
  targets <- copy(target_dt)
  valid_target <- is.finite(as_num(targets$x)) & is.finite(as_num(targets$y))
  out <- empty_method_result(targets, method, "missing_coordinates_or_donors")

  if (nrow(donors) < MIN_NN_DONORS || !any(valid_target)) return(out)

  ox <- as.matrix(donors[, .(as_num(x), as_num(y))])
  tx <- as.matrix(targets[valid_target, .(as_num(x), as_num(y))])
  oz <- as_num(donors[[TARGET]])

  pred <- rep(NA_real_, nrow(tx))
  dist <- rep(NA_real_, nrow(tx))
  for (start in seq(1L, nrow(tx), by = NN_CHUNK_SIZE)) {
    stop_at <- min(start + NN_CHUNK_SIZE - 1L, nrow(tx))
    idx <- start:stop_at
    block <- tx[idx, , drop = FALSE]
    block_sq <- rowSums(block^2)
    obs_sq <- rowSums(ox^2)
    d2 <- outer(block_sq, obs_sq, "+") - 2 * tcrossprod(block, ox)
    d2[d2 < 0] <- 0
    nearest <- max.col(-d2, ties.method = "first")
    pred[idx] <- oz[nearest]
    dist[idx] <- sqrt(d2[cbind(seq_along(nearest), nearest)])
  }

  rows <- which(valid_target)
  out[rows, prediction := pred]
  out[rows, nearest_distance_m := dist]
  out[rows, prediction_status := "ok"]
  out
}


predict_centroid_ok <- function(observed_dt, target_dt) {
  method <- "centroid_ordinary_kriging"
  donors <- copy(observed_dt)[
    is.finite(as_num(get(TARGET))) & is.finite(as_num(x)) & is.finite(as_num(y))
  ]
  targets <- copy(target_dt)
  valid_target <- is.finite(as_num(targets$x)) & is.finite(as_num(targets$y))
  out <- empty_method_result(targets, method, "missing_coordinates_or_donors")
  if (nrow(donors) < MIN_DONOR_ROWS || !any(valid_target)) return(out)

  result <- tryCatch({
    donor_x <- as_num(donors$x)
    donor_y <- as_num(donors$y)
    donor_z <- as_num(donors[[TARGET]])
    target_x <- as_num(targets$x[valid_target])
    target_y <- as_num(targets$y[valid_target])

    trend <- if (CENTROID_OK_DETREND) fit_spatial_trend(donor_x, donor_y, donor_z) else NULL
    donor_trend <- predict_spatial_trend(trend, donor_x, donor_y)
    target_trend <- predict_spatial_trend(trend, target_x, target_y)
    residual <- donor_z - donor_trend

    obs_sp <- data.frame(resid = residual, x = donor_x, y = donor_y)
    target_sp <- data.frame(x = target_x, y = target_y)
    sp::coordinates(obs_sp) <- ~ x + y
    sp::coordinates(target_sp) <- ~ x + y

    variogram_model <- fit_residual_variogram(obs_sp, residual)
    kriged <- suppressWarnings(
      gstat::krige(
        resid ~ 1,
        obs_sp,
        target_sp,
        model = variogram_model,
        nmax = CENTROID_OK_NMAX,
        debug.level = 0
      )
    )

    list(
      prediction = as_num(kriged$var1.pred) + target_trend,
      variance = as_num(kriged$var1.var),
      status = if (!is.null(trend) && isTRUE(trend$trend_used)) "ok_detrended" else "ok_plain"
    )
  }, error = function(e) list(error = conditionMessage(e)))

  if (!is.null(result$error)) {
    out[, prediction_status := paste0("failed: ", result$error)]
    return(out)
  }

  rows <- which(valid_target)
  out[rows, prediction := result$prediction]
  out[rows, kriging_variance := result$variance]
  out[rows, prediction_status := result$status]
  out
}


select_rk_covariates <- function(donors) {
  candidate <- intersect(RK_IEM_COVARIATES, names(donors))
  scores <- lapply(candidate, function(column) {
    values <- as_num(donors[[column]])
    target <- as_num(donors[[TARGET]])
    valid <- is.finite(values) & is.finite(target)
    if (sum(valid) < RK_MIN_FINITE_PER_COVARIATE ||
        length(unique(values[valid])) <= 1L) {
      return(NULL)
    }
    correlation <- suppressWarnings(cor(values[valid], target[valid], method = "spearman"))
    if (!is.finite(correlation)) correlation <- 0
    data.table(covariate = column, abs_correlation = abs(correlation))
  })
  scores <- scores[!vapply(scores, is.null, logical(1))]
  if (length(scores) == 0L) return(character())
  ranking <- rbindlist(scores)[order(-abs_correlation, covariate)]
  head(ranking$covariate, RK_MAX_COVARIATES)
}


prepare_rk_data <- function(donors, targets, covariates) {
  obs <- as.data.frame(donors)
  target <- as.data.frame(targets)
  for (column in covariates) {
    obs[[column]] <- as_num(obs[[column]])
    target[[column]] <- as_num(target[[column]])
    median_value <- median(obs[[column]], na.rm = TRUE)
    if (!is.finite(median_value)) median_value <- 0
    obs[[column]][!is.finite(obs[[column]])] <- median_value
    target[[column]][!is.finite(target[[column]])] <- median_value
  }
  list(observed = obs, target = target)
}


predict_regression_kriging <- function(observed_dt, target_dt) {
  method <- "regression_kriging"
  donors <- copy(observed_dt)[
    is.finite(as_num(get(TARGET))) & is.finite(as_num(x)) & is.finite(as_num(y))
  ]
  targets <- copy(target_dt)
  valid_target <- is.finite(as_num(targets$x)) & is.finite(as_num(targets$y))
  out <- empty_method_result(targets, method, "missing_coordinates_or_donors")
  if (nrow(donors) < MIN_DONOR_ROWS || !any(valid_target)) return(out)

  result <- tryCatch({
    target_valid <- targets[valid_target]
    covariates <- select_rk_covariates(donors)

    if (length(covariates) > 0L) {
      prepared <- prepare_rk_data(donors, target_valid, covariates)
      obs_df <- prepared$observed
      target_df <- prepared$target
      min_required <- max(
        MIN_DONOR_ROWS,
        RK_MIN_OBS_PER_PARAMETER * (length(covariates) + 1L)
      )
      if (nrow(obs_df) < min_required) covariates <- character()
    }

    if (length(covariates) == 0L) {
      obs_df <- as.data.frame(donors)
      target_df <- as.data.frame(target_valid)
      trend_formula <- as.formula(paste(TARGET, "~ x + y"))
      trend_name <- "coordinates"
    } else {
      trend_formula <- as.formula(
        paste(TARGET, "~", paste(covariates, collapse = " + "))
      )
      trend_name <- paste(covariates, collapse = ";")
    }

    obs_df[[TARGET]] <- as_num(obs_df[[TARGET]])
    obs_df$x <- as_num(obs_df$x); obs_df$y <- as_num(obs_df$y)
    target_df$x <- as_num(target_df$x); target_df$y <- as_num(target_df$y)

    fit <- lm(trend_formula, data = obs_df, na.action = na.fail)
    trend_obs <- as_num(predict(fit, newdata = obs_df))
    trend_target <- as_num(predict(fit, newdata = target_df))
    residual <- obs_df[[TARGET]] - trend_obs

    obs_sp <- data.frame(resid = residual, x = obs_df$x, y = obs_df$y)
    target_sp <- data.frame(x = target_df$x, y = target_df$y)
    sp::coordinates(obs_sp) <- ~ x + y
    sp::coordinates(target_sp) <- ~ x + y

    variogram_model <- fit_residual_variogram(obs_sp, residual)
    kriged <- suppressWarnings(
      gstat::krige(
        resid ~ 1,
        obs_sp,
        target_sp,
        model = variogram_model,
        nmax = CENTROID_OK_NMAX,
        debug.level = 0
      )
    )

    list(
      prediction = trend_target + as_num(kriged$var1.pred),
      variance = as_num(kriged$var1.var),
      status = paste0("ok_trend=", trend_name)
    )
  }, error = function(e) list(error = conditionMessage(e)))

  if (!is.null(result$error)) {
    out[, prediction_status := paste0("failed: ", result$error)]
    return(out)
  }

  rows <- which(valid_target)
  out[rows, prediction := result$prediction]
  out[rows, kriging_variance := result$variance]
  out[rows, prediction_status := result$status]
  out
}


predict_all_geostat_methods <- function(observed_dt, target_dt) {
  target <- copy(target_dt)
  if (!("row_id" %in% names(target))) target[, row_id := .I]
  parts <- list(
    predict_centroid_ok(observed_dt, target),
    predict_nearest_neighbor(observed_dt, target),
    predict_regression_kriging(observed_dt, target)
  )
  rbindlist(parts, use.names = TRUE, fill = TRUE)
}
