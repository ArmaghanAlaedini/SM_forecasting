#!/usr/bin/env Rscript

# 11b_generate_interpolation_gapfill_predictions.R
#
# Predict REAL original missing SMAP pixels using same-day observed SMAP pixels.
#
# Methods:
#   centroid_ordinary_kriging
#   nearest_neighbor_same_day
#
# Output:
#   src/data/processed/smap_gap_filling/07_gapfill_predictions/interpolation/

suppressPackageStartupMessages({
  library(data.table)
  library(sp)
  library(gstat)
  library(ggplot2)
})


# ============================================================
# MANUAL SETTINGS
# ============================================================

PROJECT_ROOT <- normalizePath(getwd(), mustWork = TRUE)

INPUT_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/03_full_smap_iem_data"
)

OUT_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/07_gapfill_predictions/interpolation"
)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

TARGET <- "soil_moisture"
KEY <- "smap_pixel_key"

PASSES <- c("am", "pm")
GAPFILL_YEARS <- c(2020, 2021, 2022, 2023, 2024, 2025)

METHODS_TO_USE <- c(
  "centroid_ordinary_kriging",
  "nearest_neighbor_same_day"
)

MIN_OBSERVED_ROWS_PER_FILE <- 30L
NN_CHUNK_SIZE <- 500L

PRED_PATH <- file.path(OUT_DIR, "interpolation_gapfill_predictions.csv")
MANIFEST_PATH <- file.path(OUT_DIR, "interpolation_gapfill_manifest.csv")


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


file_id_from_path <- function(pass_name, path) {
  paste0(pass_name, "/", basename(path))
}


list_complete_files <- function() {
  out <- list()

  for (pass_name in PASSES) {
    d <- file.path(INPUT_DIR, pass_name, "complete")

    if (!dir.exists(d)) {
      stop("Missing input folder: ", d)
    }

    files <- sort(list.files(d, pattern = "\\.csv$", full.names = TRUE))
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
  dt[, file_id := file_id_from_path(pass_name, path)]

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
    obs_sp <- as.data.frame(obs)
    target_sp <- as.data.frame(target[target_valid])

    sp::coordinates(obs_sp) <- stats::as.formula(
      paste("~", paste(coord_cols, collapse = "+"))
    )

    sp::coordinates(target_sp) <- stats::as.formula(
      paste("~", paste(coord_cols, collapse = "+"))
    )

    vals <- obs[[TARGET]]
    v <- stats::var(vals, na.rm = TRUE)

    if (!is.finite(v) || v <= 0) {
      stop("Non-positive variance in observed values.")
    }

    xr <- range(obs[[coord_cols[1]]], na.rm = TRUE)
    yr <- range(obs[[coord_cols[2]]], na.rm = TRUE)
    diag_range <- sqrt(diff(xr)^2 + diff(yr)^2)

    if (!is.finite(diag_range) || diag_range <= 0) {
      diag_range <- 1
    }

    init_model <- gstat::vgm(
      psill = 0.8 * v,
      model = "Exp",
      range = diag_range / 3,
      nugget = 0.2 * v
    )

    emp <- suppressWarnings(
      gstat::variogram(
        stats::as.formula(paste(TARGET, "~ 1")),
        obs_sp
      )
    )

    fit <- tryCatch(
      suppressWarnings(gstat::fit.variogram(emp, init_model)),
      error = function(e) init_model,
      warning = function(w) init_model
    )

    kr <- suppressWarnings(
      gstat::krige(
        stats::as.formula(paste(TARGET, "~ 1")),
        obs_sp,
        target_sp,
        model = fit,
        debug.level = 0
      )
    )

    list(
      ok = TRUE,
      target_rows = which(target_valid),
      pred = as_num(kr$var1.pred),
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


initialize_prediction_file <- function() {
  if (file.exists(PRED_PATH)) {
    file.remove(PRED_PATH)
  }

  header <- data.table(
    file_id = character(),
    date = character(),
    year = integer(),
    pass = character(),
    smap_pixel_key = character(),
    method = character(),
    prediction = numeric(),
    nearest_distance = numeric(),
    kriging_variance = numeric(),
    ok_fallback = logical(),
    source_file = character()
  )

  fwrite(header, PRED_PATH)
}


append_predictions <- function(dt) {
  out_cols <- c(
    "file_id",
    "date",
    "year",
    "pass",
    "smap_pixel_key",
    "method",
    "prediction",
    "nearest_distance",
    "kriging_variance",
    "ok_fallback",
    "source_file"
  )

  missing_cols <- setdiff(out_cols, names(dt))
  for (cc in missing_cols) {
    dt[, (cc) := NA]
  }

  fwrite(dt[, ..out_cols], PRED_PATH, append = TRUE)
}


# ============================================================
# MAIN
# ============================================================

main <- function() {
  message("11b: Generate interpolation predictions for real SMAP gaps")
  message(strrep("=", 80))
  message("Project root:  ", PROJECT_ROOT)
  message("Input folder:  ", INPUT_DIR)
  message("Output folder: ", OUT_DIR)
  message("Methods:       ", paste(METHODS_TO_USE, collapse = ", "))
  message("Gapfill years: ", paste(GAPFILL_YEARS, collapse = ", "))
  message(strrep("=", 80))

  files_by_pass <- list_complete_files()
  initialize_prediction_file()

  manifest_parts <- list()

  total_files <- sum(lengths(files_by_pass))
  scanned <- 0L
  total_missing <- 0L
  total_prediction_rows <- 0L

  for (pass_name in names(files_by_pass)) {
    files <- files_by_pass[[pass_name]]

    for (path in files) {
      scanned <- scanned + 1L

      date <- parse_date_from_filename(path)
      year <- as.integer(format(date, "%Y"))

      if (!(year %in% GAPFILL_YEARS)) {
        next
      }

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

      observed <- dt[is.finite(get(TARGET))]
      missing <- dt[!is.finite(get(TARGET))]

      n_obs <- nrow(observed)
      n_miss <- nrow(missing)

      if (n_miss == 0) {
        manifest_parts[[length(manifest_parts) + 1L]] <- data.table(
          file_id = file_id_from_path(pass_name, path),
          date = as.character(date),
          year = year,
          pass = pass_name,
          source_file = path,
          n_rows = nrow(dt),
          n_observed = n_obs,
          n_missing_target = 0L,
          n_methods = length(METHODS_TO_USE),
          n_prediction_rows = 0L
        )
        next
      }

      if (n_obs < MIN_OBSERVED_ROWS_PER_FILE) {
        warning("Skipping interpolation for file with too few observed rows: ", path)
        next
      }

      base_cols <- data.table(
        file_id = missing$file_id,
        date = as.character(missing$date),
        year = missing$year,
        pass = missing$pass,
        smap_pixel_key = missing[[KEY]],
        source_file = path
      )

      if ("nearest_neighbor_same_day" %in% METHODS_TO_USE) {
        nn <- nearest_neighbor_predict(observed, missing, coord_cols)

        nn_out <- copy(base_cols)
        nn_out[, method := "nearest_neighbor_same_day"]
        nn_out[, prediction := nn$prediction]
        nn_out[, nearest_distance := nn$nearest_distance]
        nn_out[, kriging_variance := NA_real_]
        nn_out[, ok_fallback := FALSE]

        append_predictions(nn_out)
        total_prediction_rows <- total_prediction_rows + nrow(nn_out)
      }

      if ("centroid_ordinary_kriging" %in% METHODS_TO_USE) {
        ok <- ordinary_kriging_predict(observed, missing, coord_cols)

        ok_out <- copy(base_cols)
        ok_out[, method := "centroid_ordinary_kriging"]
        ok_out[, prediction := ok$prediction]
        ok_out[, nearest_distance := NA_real_]
        ok_out[, kriging_variance := ok$kriging_variance]
        ok_out[, ok_fallback := ok$ok_fallback]

        append_predictions(ok_out)
        total_prediction_rows <- total_prediction_rows + nrow(ok_out)
      }

      total_missing <- total_missing + n_miss

      manifest_parts[[length(manifest_parts) + 1L]] <- data.table(
        file_id = file_id_from_path(pass_name, path),
        date = as.character(date),
        year = year,
        pass = pass_name,
        source_file = path,
        n_rows = nrow(dt),
        n_observed = n_obs,
        n_missing_target = n_miss,
        n_methods = length(METHODS_TO_USE),
        n_prediction_rows = n_miss * length(METHODS_TO_USE)
      )

      if (scanned %% 100 == 0) {
        message(
          "  scanned ", scanned, "/", total_files,
          "; missing rows so far: ", total_missing
        )
      }
    }
  }

  manifest <- rbindlist(manifest_parts, fill = TRUE)
  fwrite(manifest, MANIFEST_PATH)

  message("\nSaved:")
  message("  ", PRED_PATH)
  message("  ", MANIFEST_PATH)

  message("\nSummary:")
  message("  Files scanned: ", total_files)
  message("  Real missing rows found: ", total_missing)
  message("  Interpolation prediction rows written: ", total_prediction_rows)
  message("  Methods used: ", paste(METHODS_TO_USE, collapse = ", "))

  message("\nDone.")
}


main()