#!/usr/bin/env Rscript

# 10b_interpolation_validation.R
#
# Interpolation / geostatistical validation for SMAP gap filling.
#
# Fully implemented:
#   nearest_neighbor_same_day
#   centroid ordinary kriging
#
# Optional / experimental:
#   area-to-area ordinary kriging with atakrig
#   area-to-area cokriging with atakrig
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

# Laptop quick test: 20
# Full HPC run: NULL
MAX_FILES_PER_SPLIT_PER_PASS <- NULL

HOLDOUT_MODES <- c("random_cell", "spatial_block")

MAX_OBSERVED_ROWS_PER_FILE <- 800
MAX_EVAL_TARGET_ROWS_PER_MODE <- 120000

EVAL_HOLDOUT_FRACTION <- 0.25
MIN_DONOR_ROWS <- 5
BLOCK_ATTEMPTS_PER_GROUP <- 60

RANDOM_STATE <- 42

# Centroid OK settings
RUN_NEAREST_NEIGHBOR <- TRUE
RUN_CENTROID_OK <- TRUE
CENTROID_OK_NMAX <- 30
CENTROID_OK_CRS <- 3857

# ATA settings
# Keep FALSE first. Turn on after a small successful run.
RUN_ATA_OK <- FALSE
RUN_ATA_COKRIGING <- FALSE

ATA_CELL_SIZE <- 1500
ATA_NMAX <- 10
ATA_VARIOMODEL <- "Exp"
ATA_NGROUP <- 12
ATA_RD <- 0.75
ATA_COKRIGING_AUX <- c("soil12vwc_pta", "soil24vwc_pta", "soil50vwc_pta")

SAVE_PREDICTION_SAMPLE_ROWS <- 250000


# ============================================================
# PATHS
# ============================================================

find_project_root <- function() {
  env_root <- Sys.getenv("SMAP_PROJECT_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, mustWork = TRUE))
  }

  wd <- normalizePath(getwd(), mustWork = TRUE)

  # If run from project root, this is enough.
  if (dir.exists(file.path(wd, "src", "data"))) {
    return(wd)
  }

  # If run from src/code/smap_gap_filling, climb upward.
  p <- wd
  for (i in 1:6) {
    candidate <- normalizePath(file.path(p, ".."), mustWork = FALSE)
    if (dir.exists(file.path(candidate, "src", "data"))) {
      return(candidate)
    }
    p <- candidate
  }

  stop("Could not find project root. Run from project root or set SMAP_PROJECT_ROOT.")
}

PROJECT_ROOT <- find_project_root()

GAP_DIR <- file.path(PROJECT_ROOT, "src", "data", "processed", "smap_gap_filling")
FULL_DIR <- file.path(GAP_DIR, "03_full_smap_iem_data")
OUT_DIR <- file.path(GAP_DIR, "05_gapfill_model_validation", "interpolation")
FIG_DIR <- file.path(OUT_DIR, "figures")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)


# ============================================================
# UTILITIES
# ============================================================

parse_date_from_file <- function(path) {
  nm <- basename(path)
  m <- regexpr("20[0-9]{6}", nm)
  if (m < 0) stop(paste("Could not parse date from", nm))
  tag <- regmatches(nm, m)
  as.Date(tag, format = "%Y%m%d")
}

year_to_split <- function(year) {
  if (year %in% VALIDATION_YEARS) return("validation")
  if (year %in% TEST_YEARS) return("test")
  return("unused")
}

rmse <- function(obs, pred) {
  sqrt(mean((pred - obs)^2, na.rm = TRUE))
}

mae <- function(obs, pred) {
  mean(abs(pred - obs), na.rm = TRUE)
}

bias <- function(obs, pred) {
  mean(pred - obs, na.rm = TRUE)
}

r2_score <- function(obs, pred) {
  ok <- is.finite(obs) & is.finite(pred)
  obs <- obs[ok]
  pred <- pred[ok]
  if (length(obs) < 2) return(NA_real_)
  sse <- sum((obs - pred)^2)
  sst <- sum((obs - mean(obs))^2)
  if (sst == 0) return(NA_real_)
  1 - sse / sst
}

metric_row <- function(split, holdout_mode, method, obs, pred, n_features = NA_integer_, features = "") {
  data.table(
    split = split,
    holdout_mode = holdout_mode,
    method = method,
    n_features = n_features,
    features = features,
    rmse = rmse(obs, pred),
    mae = mae(obs, pred),
    bias = bias(obs, pred),
    r2 = r2_score(obs, pred),
    n = sum(is.finite(obs) & is.finite(pred))
  )
}

sample_if_needed <- function(dt, max_rows, seed) {
  if (is.null(max_rows) || nrow(dt) <= max_rows) return(copy(dt))
  set.seed(seed)
  dt[sample(.N, max_rows)]
}

add_date_features <- function(dt, date, pass_name) {
  dt[, date := as.character(date)]
  dt[, year := as.integer(format(as.Date(date), "%Y"))]
  dt[, month := as.integer(format(as.Date(date), "%m"))]
  dt[, day_of_year := as.integer(format(as.Date(date), "%j"))]
  dt[, sin_doy := sin(2 * pi * day_of_year / 366)]
  dt[, cos_doy := cos(2 * pi * day_of_year / 366)]
  dt[, pass := pass_name]
  dt[, pass_pm := ifelse(pass_name == "pm", 1L, 0L)]
  dt
}


# ============================================================
# MANIFEST
# ============================================================

build_manifest <- function() {
  rows <- list()

  for (pass_name in PASSES_TO_USE) {
    folder <- file.path(FULL_DIR, pass_name, "complete")
    files <- sort(list.files(
      folder,
      pattern = paste0("^smap_iem_", pass_name, "_complete_.*\\.csv$"),
      full.names = TRUE
    ))

    for (p in files) {
      d <- parse_date_from_file(p)
      yr <- as.integer(format(d, "%Y"))
      split <- year_to_split(yr)

      if (split == "unused") next

      rows[[length(rows) + 1]] <- data.table(
        path = p,
        file_name = basename(p),
        date = as.character(d),
        year = yr,
        pass = pass_name,
        split = split
      )
    }
  }

  manifest <- rbindlist(rows, fill = TRUE)

  if (nrow(manifest) == 0) {
    stop(paste("No complete files found under", FULL_DIR))
  }

  setorder(manifest, split, pass, date)

  if (!is.null(MAX_FILES_PER_SPLIT_PER_PASS)) {
    manifest <- manifest[, head(.SD, MAX_FILES_PER_SPLIT_PER_PASS), by = .(split, pass)]
  }

  manifest
}


# ============================================================
# HOLDOUTS
# ============================================================

mark_random_cell_holdouts <- function(dt, split_name) {
  dt <- copy(dt)
  dt[, eval_role := "donor"]

  set.seed(RANDOM_STATE + ifelse(split_name == "validation", 100, 200))

  n <- nrow(dt)
  if (n <= MIN_DONOR_ROWS + 1) return(dt)

  n_holdout <- max(1, round(n * EVAL_HOLDOUT_FRACTION))
  n_holdout <- min(n_holdout, n - MIN_DONOR_ROWS)

  target_idx <- sample(seq_len(n), n_holdout)
  dt[target_idx, eval_role := "target"]

  dt
}

choose_spatial_block_indices <- function(dt) {
  n <- nrow(dt)
  if (n <= MIN_DONOR_ROWS + 1) return(integer(0))

  desired <- max(1, round(n * EVAL_HOLDOUT_FRACTION))
  max_allowed <- n - MIN_DONOR_ROWS

  best_idx <- integer(0)
  best_score <- Inf

  if (all(c("grid_row", "grid_col") %in% names(dt)) &&
      any(!is.na(dt$grid_row)) &&
      any(!is.na(dt$grid_col))) {

    rows <- sort(unique(dt$grid_row[!is.na(dt$grid_row)]))
    cols <- sort(unique(dt$grid_col[!is.na(dt$grid_col)]))

    n_row_block <- max(1, ceiling(sqrt(EVAL_HOLDOUT_FRACTION) * length(rows)))
    n_col_block <- max(1, ceiling(sqrt(EVAL_HOLDOUT_FRACTION) * length(cols)))

    n_row_block <- min(n_row_block, length(rows))
    n_col_block <- min(n_col_block, length(cols))

    for (a in seq_len(BLOCK_ATTEMPTS_PER_GROUP)) {
      row_start <- sample(seq_len(max(1, length(rows) - n_row_block + 1)), 1)
      col_start <- sample(seq_len(max(1, length(cols) - n_col_block + 1)), 1)

      row_vals <- rows[row_start:(row_start + n_row_block - 1)]
      col_vals <- cols[col_start:(col_start + n_col_block - 1)]

      idx <- which(dt$grid_row %in% row_vals & dt$grid_col %in% col_vals)

      if (length(idx) < 1 || length(idx) > max_allowed) next

      score <- abs(length(idx) - desired)
      if (score < best_score) {
        best_score <- score
        best_idx <- idx
      }
    }

    if (length(best_idx) > 0) return(best_idx)
  }

  if (!all(c("x", "y") %in% names(dt))) return(integer(0))

  good <- which(is.finite(dt$x) & is.finite(dt$y))
  if (length(good) <= MIN_DONOR_ROWS + 1) return(integer(0))

  x_min <- min(dt$x[good], na.rm = TRUE)
  x_max <- max(dt$x[good], na.rm = TRUE)
  y_min <- min(dt$y[good], na.rm = TRUE)
  y_max <- max(dt$y[good], na.rm = TRUE)

  x_width <- (x_max - x_min) * sqrt(EVAL_HOLDOUT_FRACTION)
  y_width <- (y_max - y_min) * sqrt(EVAL_HOLDOUT_FRACTION)

  if (x_width <= 0 || y_width <= 0) return(integer(0))

  for (a in seq_len(BLOCK_ATTEMPTS_PER_GROUP)) {
    cx <- runif(1, x_min, x_max)
    cy <- runif(1, y_min, y_max)

    idx <- which(
      dt$x >= cx - x_width / 2 &
        dt$x <= cx + x_width / 2 &
        dt$y >= cy - y_width / 2 &
        dt$y <= cy + y_width / 2
    )

    if (length(idx) < 1 || length(idx) > max_allowed) next

    score <- abs(length(idx) - desired)
    if (score < best_score) {
      best_score <- score
      best_idx <- idx
    }
  }

  best_idx
}

mark_spatial_block_holdouts <- function(dt, split_name) {
  dt <- copy(dt)
  dt[, eval_role := "donor"]

  set.seed(RANDOM_STATE + ifelse(split_name == "validation", 300, 400))

  idx <- choose_spatial_block_indices(dt)

  if (length(idx) > 0) {
    dt[idx, eval_role := "target"]
  }

  dt
}

mark_holdouts <- function(dt, split_name, holdout_mode) {
  if (holdout_mode == "random_cell") {
    return(mark_random_cell_holdouts(dt, split_name))
  }
  if (holdout_mode == "spatial_block") {
    return(mark_spatial_block_holdouts(dt, split_name))
  }
  stop(paste("Unknown holdout mode:", holdout_mode))
}


# ============================================================
# DATA LOADING
# ============================================================

load_observed_file <- function(path, date, pass_name) {
  dt <- fread(path, showProgress = FALSE)
  dt <- add_date_features(dt, as.Date(date), pass_name)

  if (!(TARGET %in% names(dt))) {
    stop(paste("Missing target column", TARGET, "in", path))
  }

  dt[, (TARGET) := as.numeric(get(TARGET))]

  obs <- dt[!is.na(get(TARGET))]

  if (nrow(obs) == 0) return(obs)

  numeric_cols <- intersect(
    c("x", "y", "grid_row", "grid_col", "sin_doy", "cos_doy", "pass_pm"),
    names(obs)
  )

  for (cc in numeric_cols) {
    obs[, (cc) := as.numeric(get(cc))]
  }

  if (!is.null(MAX_OBSERVED_ROWS_PER_FILE) && nrow(obs) > MAX_OBSERVED_ROWS_PER_FILE) {
    seed <- RANDOM_STATE + as.integer(format(as.Date(date), "%Y%m%d")) + ifelse(pass_name == "pm", 1, 0)
    obs <- sample_if_needed(obs, MAX_OBSERVED_ROWS_PER_FILE, seed)
  }

  obs
}


# ============================================================
# METHODS
# ============================================================

predict_nearest_neighbor <- function(eval_dt) {
  donors <- eval_dt[eval_role == "donor" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  targets <- eval_dt[eval_role == "target" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]

  if (nrow(donors) < MIN_DONOR_ROWS || nrow(targets) == 0) {
    return(NULL)
  }

  dx <- outer(targets$x, donors$x, "-")
  dy <- outer(targets$y, donors$y, "-")
  d2 <- dx^2 + dy^2

  nearest <- max.col(-d2, ties.method = "first")

  pred <- donors[[TARGET]][nearest]
  dist <- sqrt(d2[cbind(seq_len(nrow(targets)), nearest)])

  data.table(
    date = targets$date,
    pass = targets$pass,
    smap_pixel_key = targets$smap_pixel_key,
    observed = targets[[TARGET]],
    prediction = pred,
    method = "nearest_neighbor_same_day",
    nearest_distance_m = dist
  )
}

predict_centroid_ok <- function(eval_dt) {
  if (!requireNamespace("sf", quietly = TRUE) ||
      !requireNamespace("gstat", quietly = TRUE) ||
      !requireNamespace("sp", quietly = TRUE)) {
    message("[skip] centroid_ok needs packages: sf, sp, gstat")
    return(NULL)
  }

  donors <- eval_dt[eval_role == "donor" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]
  targets <- eval_dt[eval_role == "target" & is.finite(x) & is.finite(y) & !is.na(get(TARGET))]

  if (nrow(donors) < 20 || nrow(targets) == 0) return(NULL)

  donors_df <- as.data.frame(donors)
  targets_df <- as.data.frame(targets)

  donors_sf <- sf::st_as_sf(donors_df, coords = c("x", "y"), crs = CENTROID_OK_CRS, remove = FALSE)
  targets_sf <- sf::st_as_sf(targets_df, coords = c("x", "y"), crs = CENTROID_OK_CRS, remove = FALSE)

  names(donors_sf)[names(donors_sf) == TARGET] <- "z"

  out <- tryCatch({
    vg_emp <- gstat::variogram(z ~ 1, donors_sf)

    initial_model <- gstat::vgm(
      psill = stats::var(donors_sf$z, na.rm = TRUE),
      model = "Sph",
      range = 50000,
      nugget = 0.001
    )

    vg_fit <- tryCatch(
      gstat::fit.variogram(vg_emp, initial_model),
      error = function(e) initial_model
    )

    pred <- gstat::krige(
      z ~ 1,
      locations = donors_sf,
      newdata = targets_sf,
      model = vg_fit,
      nmax = CENTROID_OK_NMAX,
      debug.level = 0
    )

    data.table(
      date = targets$date,
      pass = targets$pass,
      smap_pixel_key = targets$smap_pixel_key,
      observed = targets[[TARGET]],
      prediction = as.numeric(pred$var1.pred),
      method = "centroid_ordinary_kriging",
      nearest_distance_m = NA_real_
    )
  }, error = function(e) {
    message("[centroid_ok failed] ", conditionMessage(e))
    NULL
  })

  out
}

make_sf_polygons <- function(dt, value_col = NULL) {
  if (!requireNamespace("sf", quietly = TRUE)) {
    stop("Package sf is needed.")
  }

  if (!("geometry_wkt" %in% names(dt))) {
    stop("geometry_wkt column not found.")
  }

  x <- as.data.frame(dt)
  x$ata_area_id <- seq_len(nrow(x))

  geom <- sf::st_as_sfc(x$geometry_wkt, crs = CENTROID_OK_CRS)
  sfobj <- sf::st_as_sf(x, geometry = geom, crs = CENTROID_OK_CRS)

  if (!is.null(value_col) && value_col %in% names(sfobj)) {
    sfobj$value <- sfobj[[value_col]]
  }

  sfobj
}

predict_ata_ok <- function(eval_dt) {
  if (!RUN_ATA_OK) return(NULL)

  if (!requireNamespace("atakrig", quietly = TRUE) ||
      !requireNamespace("sf", quietly = TRUE)) {
    message("[skip] ata_ok needs packages: atakrig, sf")
    return(NULL)
  }

  donors <- eval_dt[eval_role == "donor" & !is.na(get(TARGET))]
  targets <- eval_dt[eval_role == "target" & !is.na(get(TARGET))]

  if (nrow(donors) < 20 || nrow(targets) == 0) return(NULL)
  if (!("geometry_wkt" %in% names(eval_dt))) return(NULL)

  out <- tryCatch({
    donors_sf <- make_sf_polygons(donors, value_col = TARGET)
    targets_sf <- make_sf_polygons(targets, value_col = NULL)

    obs_d <- atakrig::discretizePolygon(
      donors_sf,
      cellsize = ATA_CELL_SIZE,
      id = "ata_area_id",
      value = "value"
    )

    pred_d <- atakrig::discretizePolygon(
      targets_sf,
      cellsize = ATA_CELL_SIZE,
      id = "ata_area_id"
    )

    ptvgm <- atakrig::deconvPointVgm(
      obs_d,
      model = ATA_VARIOMODEL,
      ngroup = ATA_NGROUP,
      rd = ATA_RD,
      fig = FALSE
    )

    pred <- atakrig::ataKriging(
      obs_d,
      pred_d,
      ptvgm$pointVariogram,
      nmax = ATA_NMAX,
      longlat = FALSE,
      showProgress = FALSE,
      nopar = TRUE
    )

    pred_dt <- as.data.table(pred)
    setnames(pred_dt, old = names(pred_dt), new = make.names(names(pred_dt)))

    pred_col <- if ("pred" %in% names(pred_dt)) "pred" else names(pred_dt)[grepl("pred", names(pred_dt), ignore.case = TRUE)][1]
    id_col <- if ("areaId" %in% names(pred_dt)) "areaId" else names(pred_dt)[1]

    targets_out <- copy(targets)
    targets_out[, ata_area_id := seq_len(.N)]

    merged <- merge(
      targets_out,
      pred_dt[, .(ata_area_id = as.integer(get(id_col)), prediction = as.numeric(get(pred_col)))],
      by = "ata_area_id",
      all.x = TRUE
    )

    data.table(
      date = merged$date,
      pass = merged$pass,
      smap_pixel_key = merged$smap_pixel_key,
      observed = merged[[TARGET]],
      prediction = merged$prediction,
      method = "ata_ordinary_kriging",
      nearest_distance_m = NA_real_
    )
  }, error = function(e) {
    message("[ata_ok failed] ", conditionMessage(e))
    NULL
  })

  out
}

predict_ata_cokriging_one_aux <- function(eval_dt, aux_col) {
  if (!RUN_ATA_COKRIGING) return(NULL)

  if (!requireNamespace("atakrig", quietly = TRUE) ||
      !requireNamespace("sf", quietly = TRUE)) {
    message("[skip] ata_cokriging needs packages: atakrig, sf")
    return(NULL)
  }

  if (!(aux_col %in% names(eval_dt))) return(NULL)
  if (!("geometry_wkt" %in% names(eval_dt))) return(NULL)

  donors <- eval_dt[eval_role == "donor" & !is.na(get(TARGET)) & !is.na(get(aux_col))]
  targets <- eval_dt[eval_role == "target" & !is.na(get(TARGET)) & !is.na(get(aux_col))]
  aux_all <- eval_dt[!is.na(get(aux_col))]

  if (nrow(donors) < 20 || nrow(targets) == 0 || nrow(aux_all) < 20) return(NULL)

  out <- tryCatch({
    smap_sf <- make_sf_polygons(donors, value_col = TARGET)

    aux_tmp <- copy(aux_all)
    aux_tmp[, aux_value := as.numeric(get(aux_col))]
    aux_sf <- make_sf_polygons(aux_tmp, value_col = "aux_value")

    targets_sf <- make_sf_polygons(targets, value_col = NULL)

    smap_d <- atakrig::discretizePolygon(
      smap_sf,
      cellsize = ATA_CELL_SIZE,
      id = "ata_area_id",
      value = "value"
    )

    aux_d <- atakrig::discretizePolygon(
      aux_sf,
      cellsize = ATA_CELL_SIZE,
      id = "ata_area_id",
      value = "value"
    )

    target_d <- atakrig::discretizePolygon(
      targets_sf,
      cellsize = ATA_CELL_SIZE,
      id = "ata_area_id"
    )

    data_list <- list(
      smap = smap_d,
      aux = aux_d
    )

    ptvgms <- atakrig::deconvPointVgmForCoKriging(
      data_list,
      model = ATA_VARIOMODEL,
      ngroup = ATA_NGROUP,
      rd = ATA_RD,
      fig = FALSE
    )

    pred <- atakrig::ataCoKriging(
      data_list,
      unknownVarId = "smap",
      unknown = target_d,
      ptVgms = ptvgms,
      nmax = ATA_NMAX,
      longlat = FALSE,
      oneCondition = TRUE,
      auxRatioAdj = TRUE,
      showProgress = FALSE,
      nopar = TRUE
    )

    pred_dt <- as.data.table(pred)
    setnames(pred_dt, old = names(pred_dt), new = make.names(names(pred_dt)))

    pred_col <- if ("pred" %in% names(pred_dt)) "pred" else names(pred_dt)[grepl("pred", names(pred_dt), ignore.case = TRUE)][1]
    id_col <- if ("areaId" %in% names(pred_dt)) "areaId" else names(pred_dt)[1]

    targets_out <- copy(targets)
    targets_out[, ata_area_id := seq_len(.N)]

    merged <- merge(
      targets_out,
      pred_dt[, .(ata_area_id = as.integer(get(id_col)), prediction = as.numeric(get(pred_col)))],
      by = "ata_area_id",
      all.x = TRUE
    )

    data.table(
      date = merged$date,
      pass = merged$pass,
      smap_pixel_key = merged$smap_pixel_key,
      observed = merged[[TARGET]],
      prediction = merged$prediction,
      method = paste0("ata_cokriging_", aux_col),
      nearest_distance_m = NA_real_
    )
  }, error = function(e) {
    message("[ata_cokriging failed: ", aux_col, "] ", conditionMessage(e))
    NULL
  })

  out
}

predict_all_methods <- function(eval_dt) {
  parts <- list()

  if (RUN_NEAREST_NEIGHBOR) {
    parts[[length(parts) + 1]] <- predict_nearest_neighbor(eval_dt)
  }

  if (RUN_CENTROID_OK) {
    parts[[length(parts) + 1]] <- predict_centroid_ok(eval_dt)
  }

  if (RUN_ATA_OK) {
    parts[[length(parts) + 1]] <- predict_ata_ok(eval_dt)
  }

  if (RUN_ATA_COKRIGING) {
    for (aux in ATA_COKRIGING_AUX) {
      parts[[length(parts) + 1]] <- predict_ata_cokriging_one_aux(eval_dt, aux)
    }
  }

  parts <- parts[!vapply(parts, is.null, logical(1))]

  if (length(parts) == 0) return(NULL)

  rbindlist(parts, fill = TRUE)
}


# ============================================================
# EVALUATION
# ============================================================

evaluate_one_file <- function(path, date, pass_name, split_name) {
  obs <- load_observed_file(path, date, pass_name)

  if (nrow(obs) == 0) return(list(metrics = NULL, predictions = NULL))

  all_preds <- list()
  all_metrics <- list()

  for (mode in HOLDOUT_MODES) {
    eval_dt <- mark_holdouts(obs, split_name, mode)
    eval_dt[, holdout_mode := mode]
    eval_dt[, split := split_name]

    targets <- eval_dt[eval_role == "target"]

    if (nrow(targets) == 0) next

    preds <- predict_all_methods(eval_dt)

    if (is.null(preds) || nrow(preds) == 0) next

    preds[, split := split_name]
    preds[, holdout_mode := mode]
    preds[, file_name := basename(path)]

    metrics <- preds[
      ,
      metric_row(
        split = split_name,
        holdout_mode = mode,
        method = method[1],
        obs = observed,
        pred = prediction
      ),
      by = method
    ]

    all_preds[[length(all_preds) + 1]] <- preds
    all_metrics[[length(all_metrics) + 1]] <- metrics
  }

  list(
    metrics = if (length(all_metrics) > 0) rbindlist(all_metrics, fill = TRUE) else NULL,
    predictions = if (length(all_preds) > 0) rbindlist(all_preds, fill = TRUE) else NULL
  )
}

aggregate_metrics <- function(preds) {
  out <- preds[
    ,
    .(
      n_features = NA_integer_,
      features = "",
      rmse = rmse(observed, prediction),
      mae = mae(observed, prediction),
      bias = bias(observed, prediction),
      r2 = r2_score(observed, prediction),
      n = sum(is.finite(observed) & is.finite(prediction))
    ),
    by = .(split, holdout_mode, method)
  ]

  setorder(out, split, holdout_mode, rmse)
  out
}


# ============================================================
# PLOTS
# ============================================================

plot_rmse <- function(metrics) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("[plot skip] ggplot2 not installed")
    return(invisible(NULL))
  }

  for (mode in unique(metrics$holdout_mode)) {
    sub <- metrics[split == "validation" & holdout_mode == mode]
    if (nrow(sub) == 0) next

    sub <- sub[order(rmse)]
    sub[, method := factor(method, levels = rev(method))]

    p <- ggplot2::ggplot(sub, ggplot2::aes(x = rmse, y = method)) +
      ggplot2::geom_col() +
      ggplot2::geom_text(
        ggplot2::aes(label = sprintf("%.4f", rmse)),
        hjust = -0.05,
        size = 3
      ) +
      ggplot2::labs(
        title = paste("Interpolation/geostat validation RMSE:", mode),
        x = "Validation RMSE, lower is better",
        y = NULL
      ) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())

    out <- file.path(FIG_DIR, paste0("interp_validation_rmse_", mode, ".pdf"))
    ggplot2::ggsave(out, p, width = 9, height = 6)
    message("Saved: ", out)
  }
}

plot_bias <- function(metrics) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    message("[plot skip] ggplot2 not installed")
    return(invisible(NULL))
  }

  for (mode in unique(metrics$holdout_mode)) {
    sub <- metrics[split == "validation" & holdout_mode == mode]
    if (nrow(sub) == 0) next

    sub <- sub[order(rmse)]
    sub[, method := factor(method, levels = rev(method))]

    p <- ggplot2::ggplot(sub, ggplot2::aes(x = bias, y = method)) +
      ggplot2::geom_col() +
      ggplot2::geom_vline(xintercept = 0, linewidth = 0.3) +
      ggplot2::labs(
        title = paste("Interpolation/geostat validation bias:", mode),
        x = "Bias = mean(prediction - observed)",
        y = NULL
      ) +
      ggplot2::theme_minimal(base_size = 12) +
      ggplot2::theme(panel.grid.major.y = ggplot2::element_blank())

    out <- file.path(FIG_DIR, paste0("interp_validation_bias_", mode, ".pdf"))
    ggplot2::ggsave(out, p, width = 9, height = 6)
    message("Saved: ", out)
  }
}


# ============================================================
# MAIN
# ============================================================

main <- function() {
  message("\nInterpolation/geostatistical SMAP validation")
  message("============================================================")
  message("Input folder:      ", FULL_DIR)
  message("Output folder:     ", OUT_DIR)
  message("Max files/split:   ", ifelse(is.null(MAX_FILES_PER_SPLIT_PER_PASS), "NULL", MAX_FILES_PER_SPLIT_PER_PASS))
  message("RUN_ATA_OK:        ", RUN_ATA_OK)
  message("RUN_ATA_COKRIGING: ", RUN_ATA_COKRIGING)
  message("============================================================")

  manifest <- build_manifest()
  fwrite(manifest, file.path(OUT_DIR, "interpolation_validation_manifest.csv"))

  message("\nFiles by split/pass:")
  print(manifest[, .N, by = .(split, pass)])

  all_preds <- list()

  for (i in seq_len(nrow(manifest))) {
    row <- manifest[i]

    if (row$split == "test" && !RUN_TEST) next

    message(
      sprintf(
        "[%d/%d] %s %s %s",
        i, nrow(manifest), row$split, row$pass, row$date
      )
    )

    res <- evaluate_one_file(
      path = row$path,
      date = row$date,
      pass_name = row$pass,
      split_name = row$split
    )

    if (!is.null(res$predictions) && nrow(res$predictions) > 0) {
      all_preds[[length(all_preds) + 1]] <- res$predictions
    }
  }

  if (length(all_preds) == 0) {
    stop("No predictions were generated.")
  }

  preds <- rbindlist(all_preds, fill = TRUE)

  if (!is.null(SAVE_PREDICTION_SAMPLE_ROWS) && nrow(preds) > SAVE_PREDICTION_SAMPLE_ROWS) {
    set.seed(RANDOM_STATE)
    preds_out <- preds[sample(.N, SAVE_PREDICTION_SAMPLE_ROWS)]
  } else {
    preds_out <- preds
  }

  metrics <- aggregate_metrics(preds)

  metrics_path <- file.path(OUT_DIR, "interpolation_validation_metrics.csv")
  preds_path <- file.path(OUT_DIR, "interpolation_validation_predictions_sample.csv")

  fwrite(metrics, metrics_path)
  fwrite(preds_out, preds_path)

  message("\nSaved metrics:     ", metrics_path)
  message("Saved predictions: ", preds_path)

  message("\nBest validation results:")
  print(metrics[split == "validation"][order(holdout_mode, rmse)])

  plot_rmse(metrics)
  plot_bias(metrics)

  message("\nDone.")
}

main()