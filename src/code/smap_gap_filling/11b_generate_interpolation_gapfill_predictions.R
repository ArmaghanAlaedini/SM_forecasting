#!/usr/bin/env Rscript

# 11b_generate_interpolation_gapfill_predictions.R
#
# Predict original missing SMAP pixels with the same GI implementations and
# settings used in 10b validation and 10e independent testing.  Each method is
# written separately; no method silently falls back to nearest neighbor here.
# Final fallback decisions are made transparently by 11c.

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
SCRIPT_DIR <- if (length(file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = TRUE))
} else {
  normalizePath(getwd(), mustWork = TRUE)
}
source(file.path(SCRIPT_DIR, "gapfill_geostat_common.R"))

PROJECT_ROOT <- find_project_root()
INPUT_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/03_full_smap_iem_data"
)
OUT_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/07_gapfill_predictions/interpolation"
)
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

GAPFILL_YEARS <- 2020:2025
PREDICTION_PATH <- file.path(OUT_DIR, "interpolation_gapfill_predictions.csv")
MANIFEST_PATH <- file.path(OUT_DIR, "interpolation_gapfill_manifest.csv")


list_complete_files <- function() {
  rows <- list()
  for (pass_name in PASSES) {
    folder <- file.path(INPUT_DIR, pass_name, "complete")
    if (!dir.exists(folder)) stop("Missing input folder: ", folder)
    paths <- sort(list.files(folder, pattern = "\\.csv$", full.names = TRUE))
    for (path in paths) {
      date <- parse_date_from_filename(path)
      year <- as.integer(format(date, "%Y"))
      if (year %in% GAPFILL_YEARS) {
        rows[[length(rows) + 1L]] <- data.table(
          pass = pass_name,
          path = path,
          date = date,
          year = year
        )
      }
    }
  }
  if (length(rows) == 0L) stop("No complete files found under ", INPUT_DIR)
  rbindlist(rows)
}


initialize_output <- function() {
  if (file.exists(PREDICTION_PATH)) file.remove(PREDICTION_PATH)
  header <- data.table(
    file_id = character(),
    date = character(),
    year = integer(),
    pass = character(),
    smap_pixel_key = character(),
    method = character(),
    prediction = numeric(),
    kriging_variance = numeric(),
    nearest_distance_m = numeric(),
    prediction_status = character(),
    source_file = character()
  )
  fwrite(header, PREDICTION_PATH)
}


append_predictions <- function(dt) {
  fwrite(dt, PREDICTION_PATH, append = TRUE, col.names = FALSE)
}


process_one_file <- function(pass_name, path) {
  date <- parse_date_from_filename(path)
  dt <- fread(path)
  dt <- add_basic_columns(dt, pass_name, path)
  dt[, (TARGET) := as_num(get(TARGET))]

  observed_dt <- dt[is.finite(get(TARGET))]
  missing_dt <- dt[!is.finite(get(TARGET))]
  if (nrow(missing_dt) == 0L) {
    return(list(predictions = NULL, n_rows = nrow(dt), n_missing = 0L))
  }

  missing_dt[, row_id := .I]
  prediction_values <- predict_all_geostat_methods(observed_dt, missing_dt)
  metadata <- missing_dt[, .(
    row_id,
    file_id,
    date = as.character(date),
    year,
    pass,
    smap_pixel_key = get(KEY),
    source_file = normalizePath(path, mustWork = TRUE)
  )]
  predictions <- merge(
    metadata,
    prediction_values,
    by = "row_id",
    all.x = FALSE,
    all.y = TRUE,
    sort = FALSE
  )
  predictions[, row_id := NULL]
  setcolorder(
    predictions,
    c(
      "file_id", "date", "year", "pass", "smap_pixel_key", "method",
      "prediction", "kriging_variance", "nearest_distance_m",
      "prediction_status", "source_file"
    )
  )
  list(predictions = predictions, n_rows = nrow(dt), n_missing = nrow(missing_dt))
}


main <- function() {
  message("11b: Generate GI predictions for original SMAP gaps")
  message(strrep("=", 78))
  message("Project root: ", PROJECT_ROOT)
  message("Gap-fill years: ", paste(GAPFILL_YEARS, collapse = ", "))
  message("Project seed: ", RANDOM_SEED)
  message(strrep("=", 78))

  files <- list_complete_files()
  initialize_output()
  manifest_rows <- list()
  total_missing <- 0L
  total_predictions <- 0L

  for (i in seq_len(nrow(files))) {
    row <- files[i]
    result <- process_one_file(row$pass, row$path)
    n_written <- 0L
    if (!is.null(result$predictions)) {
      append_predictions(result$predictions)
      n_written <- nrow(result$predictions)
    }

    manifest_rows[[length(manifest_rows) + 1L]] <- data.table(
      file_id = file_id_from_path(row$pass, row$path),
      date = as.character(row$date),
      year = row$year,
      pass = row$pass,
      source_file = normalizePath(row$path, mustWork = TRUE),
      n_rows = result$n_rows,
      n_original_missing = result$n_missing,
      n_methods = length(SELECTED_INTERPOLATION_METHODS),
      n_prediction_rows = n_written,
      project_seed = RANDOM_SEED
    )

    total_missing <- total_missing + result$n_missing
    total_predictions <- total_predictions + n_written
    if (i %% 100L == 0L) {
      message(
        "  processed ", i, " / ", nrow(files),
        "; original gaps=", total_missing,
        "; prediction rows=", total_predictions
      )
    }
  }

  fwrite(rbindlist(manifest_rows, fill = TRUE), MANIFEST_PATH)
  message("\nSaved:")
  message("  ", PREDICTION_PATH)
  message("  ", MANIFEST_PATH)
  message("\nOriginal missing pixels: ", total_missing)
  message("GI prediction rows:       ", total_predictions)
}


main()
