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
# PROJECT ROOT  ← FIXED: uses env var, not getwd()
# ============================================================

find_project_root <- function() {
  env_root <- Sys.getenv("SMAP_PROJECT_ROOT", unset = "")
  if (nzchar(env_root)) {
    return(normalizePath(env_root, mustWork = TRUE))
  }

  # Walk up from script location
  script_path <- tryCatch(
    normalizePath(sys.frame(1)$ofile, mustWork = FALSE),
    error = function(e) ""
  )

  if (nzchar(script_path)) {
    p <- dirname(script_path)
    for (i in 1:6) {
      if (dir.exists(file.path(p, "src")) &&
          (file.exists(file.path(p, ".git")) ||
           dir.exists(file.path(p, "renv")) ||
           file.exists(file.path(p, "environment.yml")))) {
        return(normalizePath(p, mustWork = TRUE))
      }
      p <- dirname(p)
    }
  }

  # Last resort: walk up from working directory
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

  stop(
    "Could not find project root.\n",
    "Set SMAP_PROJECT_ROOT env var or run from inside the project folder.\n",
    "Example:  export SMAP_PROJECT_ROOT=/work/estherjo/alaedini/projects/iem_pta_kriging"
  )
}

PROJECT_ROOT <- find_project_root()
message("Project root: ", PROJECT_ROOT)


# ============================================================
# MANUAL SETTINGS
# ============================================================

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
KEY    <- "smap_pixel_key"

PASSES        <- c("am", "pm")
GAPFILL_YEARS <- c(2020, 2021, 2022, 2023, 2024, 2025)

METHODS_TO_USE <- c(
  "centroid_ordinary_kriging",
  "nearest_neighbor_same_day"
)

MIN_OBSERVED_ROWS_PER_FILE <- 30L
NN_CHUNK_SIZE              <- 500L

PRED_PATH     <- file.path(OUT_DIR, "interpolation_gapfill_predictions.csv")
MANIFEST_PATH <- file.path(OUT_DIR, "interpolation_gapfill_manifest.csv")


# ============================================================
# HELPERS
# ============================================================

parse_date_from_filename <- function(path) {
  base <- basename(path)
  m    <- regmatches(base, regexpr("[0-9]{8}", base))
  if (length(m) == 0 || m == "") stop("Could not parse YYYYMMDD from: ", path)
  as.IDate(m, format = "%Y%m%d")
}

file_id_from_path <- function(pass_name, path) {
  paste0(pass_name, "/", basename(path))
}

list_complete_files <- function() {
  out <- list()
  for (pass_name in PASSES) {
    d <- file.path(INPUT_DIR, pass_name, "complete")
    if (!dir.exists(d)) stop("Missing input folder: ", d)
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

  dt[, year        := as.integer(format(date, "%Y"))]
  dt[, month       := as.integer(format(date, "%m"))]
  dt[, day_of_year := as.integer(format(date, "%j"))]
  dt[, pass        := pass_name]
  dt[, file_id     := file_id_from_path(pass_name, path)]

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


# ============================================================
# NEAREST NEIGHBOUR (vectorised)
# ============================================================

predict_nearest_neighbor <- function(observed_dt, missing_dt) {
  if (nrow(observed_dt) == 0 || nrow(missing_dt) == 0) {
    return(data.table(
      smap_pixel_key   = missing_dt[[KEY]],
      method           = "nearest_neighbor_same_day",
      prediction       = NA_real_,
      nearest_distance = NA_real_
    ))
  }

  obs_x <- as.numeric(observed_dt[["x"]])
  obs_y <- as.numeric(observed_dt[["y"]])
  mis_x <- as.numeric(missing_dt[["x"]])
  mis_y <- as.numeric(missing_dt[["y"]])

  n_chunks <- ceiling(nrow(missing_dt) / NN_CHUNK_SIZE)
  preds    <- numeric(nrow(missing_dt))
  dists    <- numeric(nrow(missing_dt))

  for (chunk in seq_len(n_chunks)) {
    idx_start <- (chunk - 1L) * NN_CHUNK_SIZE + 1L
    idx_end   <- min(chunk * NN_CHUNK_SIZE, nrow(missing_dt))
    cx        <- mis_x[idx_start:idx_end]
    cy        <- mis_y[idx_start:idx_end]

    dx  <- outer(cx, obs_x, "-")
    dy  <- outer(cy, obs_y, "-")
    d2  <- dx^2 + dy^2
    idx <- max.col(-d2, ties.method = "first")

    preds[idx_start:idx_end] <- as.numeric(observed_dt[[TARGET]])[idx]
    dists[idx_start:idx_end] <- sqrt(d2[cbind(seq_along(cx), idx)])
  }

  data.table(
    smap_pixel_key   = missing_dt[[KEY]],
    method           = "nearest_neighbor_same_day",
    prediction       = preds,
    nearest_distance = dists
  )
}


# ============================================================
# CENTROID ORDINARY KRIGING
# ============================================================

predict_centroid_ok <- function(observed_dt, missing_dt) {
  empty_result <- data.table(
    smap_pixel_key  = missing_dt[[KEY]],
    method          = "centroid_ordinary_kriging",
    prediction      = NA_real_,
    kriging_variance = NA_real_
  )

  if (nrow(observed_dt) < 5L || nrow(missing_dt) == 0L) return(empty_result)

  obs_x <- as.numeric(observed_dt[["x"]])
  obs_y <- as.numeric(observed_dt[["y"]])
  obs_z <- as.numeric(observed_dt[[TARGET]])

  valid <- is.finite(obs_x) & is.finite(obs_y) & is.finite(obs_z)
  if (sum(valid) < 5L) return(empty_result)

  obs_x <- obs_x[valid]
  obs_y <- obs_y[valid]
  obs_z <- obs_z[valid]

  mis_x <- as.numeric(missing_dt[["x"]])
  mis_y <- as.numeric(missing_dt[["y"]])

  result <- tryCatch({
    obs_sp <- SpatialPointsDataFrame(
      coords = cbind(obs_x, obs_y),
      data   = data.frame(z = obs_z)
    )
    mis_sp <- SpatialPoints(cbind(mis_x, mis_y))

    vfit <- tryCatch(
      fit.variogram(
        variogram(z ~ 1, obs_sp),
        vgm(c("Sph", "Exp", "Gau"))
      ),
      error = function(e) NULL
    )

    if (is.null(vfit)) return(empty_result)

    kg <- krige(z ~ 1, obs_sp, mis_sp, model = vfit, debug.level = 0)

    data.table(
      smap_pixel_key   = missing_dt[[KEY]],
      method           = "centroid_ordinary_kriging",
      prediction       = as.numeric(kg$var1.pred),
      kriging_variance = as.numeric(kg$var1.var)
    )
  }, error = function(e) {
    message("  Kriging failed: ", conditionMessage(e))
    empty_result
  })

  result
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
    file_id            = fid,
    date               = as.character(date),
    year               = year,
    pass               = pass_name,
    source_file        = path,
    n_rows             = nrow(dt),
    n_observed         = nrow(observed),
    n_missing          = nrow(missing),
    methods_run        = paste(METHODS_TO_USE, collapse = ";"),
    status             = "ok",
    message            = ""
  )

  if (nrow(missing) == 0) {
    return(list(preds = NULL, manifest = manifest_row))
  }

  if (nrow(observed) < MIN_OBSERVED_ROWS_PER_FILE) {
    manifest_row$status  <- "skipped_too_few_observed"
    manifest_row$message <- paste("Only", nrow(observed), "observed rows")
    return(list(preds = NULL, manifest = manifest_row))
  }

  parts <- list()

  if ("nearest_neighbor_same_day" %in% METHODS_TO_USE) {
    nn <- predict_nearest_neighbor(observed, missing)
    nn[, file_id    := fid]
    nn[, date       := as.character(date)]
    nn[, year       := year]
    nn[, pass       := pass_name]
    nn[, (KEY)      := missing[[KEY]]]
    parts[["nn"]] <- nn
  }

  if ("centroid_ordinary_kriging" %in% METHODS_TO_USE) {
    ok <- predict_centroid_ok(observed, missing)
    ok[, file_id := fid]
    ok[, date    := as.character(date)]
    ok[, year    := year]
    ok[, pass    := pass_name]
    ok[, (KEY)   := missing[[KEY]]]
    parts[["ok"]] <- ok
  }

  if (length(parts) == 0) return(list(preds = NULL, manifest = manifest_row))

  preds <- rbindlist(parts, fill = TRUE)
  list(preds = preds, manifest = manifest_row)
}


# ============================================================
# MAIN
# ============================================================

main <- function() {
  message("11b: Generate interpolation gap-fill predictions")
  message(strrep("=", 70))
  message("Project root: ", PROJECT_ROOT)
  message("Input dir:    ", INPUT_DIR)
  message("Output dir:   ", OUT_DIR)
  message("Methods:      ", paste(METHODS_TO_USE, collapse = ", "))
  message("Years:        ", paste(GAPFILL_YEARS, collapse = ", "))
  message(strrep("=", 70))

  all_files <- list_complete_files()

  # Write output header
  header <- data.table(
    file_id          = character(),
    date             = character(),
    year             = integer(),
    pass             = character(),
    smap_pixel_key   = character(),
    method           = character(),
    prediction       = numeric(),
    kriging_variance = numeric(),
    nearest_distance = numeric()
  )
  fwrite(header, PRED_PATH)

  manifest_rows <- list()
  total_files   <- sum(lengths(all_files))
  counter       <- 0L

  for (pass_name in PASSES) {
    files <- all_files[[pass_name]]
    message("\nProcessing ", toupper(pass_name), " files: ", length(files))

    for (path in files) {
      counter <- counter + 1L
      result  <- tryCatch(
        process_one_file(pass_name, path),
        error = function(e) {
          message("  FAILED: ", basename(path), " | ", conditionMessage(e))
          list(
            preds    = NULL,
            manifest = list(
              file_id     = file_id_from_path(pass_name, path),
              date        = NA_character_,
              year        = NA_integer_,
              pass        = pass_name,
              source_file = path,
              n_rows      = NA_integer_,
              n_observed  = NA_integer_,
              n_missing   = NA_integer_,
              methods_run = paste(METHODS_TO_USE, collapse = ";"),
              status      = "failed",
              message     = conditionMessage(e)
            )
          )
        }
      )

      if (!is.null(result$preds) && nrow(result$preds) > 0) {
        fwrite(result$preds, PRED_PATH, append = TRUE)
      }

      manifest_rows[[counter]] <- result$manifest

      if (counter %% 100 == 0 || counter == total_files) {
        message("  processed ", counter, "/", total_files, " files")
      }
    }
  }

  manifest_dt <- rbindlist(manifest_rows, fill = TRUE)
  fwrite(manifest_dt, MANIFEST_PATH)

  message("\nSaved predictions: ", PRED_PATH)
  message("Saved manifest:    ", MANIFEST_PATH)
  message("\nStatus counts:")
  print(manifest_dt[, .N, by = status])
  message("\nDone.")
}

main()