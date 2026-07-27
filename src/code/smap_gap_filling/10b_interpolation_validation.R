#!/usr/bin/env Rscript

# 10b_interpolation_validation.R
#
# Validate the three GI methods on the exact 2024 target keys created by
# 10_generate_holdout_manifests.py.  This script never creates or samples its
# own gaps.  Metrics are pooled over target pixels so they are directly
# comparable with the ML metrics.

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
VALIDATION_DIR <- file.path(
  PROJECT_ROOT,
  "src/data/processed/smap_gap_filling/05_gapfill_model_validation"
)
HOLDOUT_PATH <- file.path(
  VALIDATION_DIR,
  "holdouts/validation_holdouts_2024.csv"
)
OUT_DIR <- file.path(VALIDATION_DIR, "interpolation")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

PREDICTION_PATH <- file.path(OUT_DIR, "interpolation_validation_predictions.csv")
METRICS_PATH <- file.path(OUT_DIR, "interpolation_validation_metrics.csv")
DAILY_METRICS_PATH <- file.path(OUT_DIR, "interpolation_validation_daily_metrics.csv")
COVERAGE_PATH <- file.path(OUT_DIR, "interpolation_validation_coverage.csv")


resolve_source_file <- function(row) {
  candidate <- as.character(row$source_file[1])
  if (file.exists(candidate)) return(candidate)
  fallback <- file.path(
    INPUT_DIR,
    as.character(row$pass[1]),
    "complete",
    basename(candidate)
  )
  if (file.exists(fallback)) return(fallback)
  stop("Could not locate source complete file: ", candidate)
}


read_holdout_manifest <- function() {
  if (!file.exists(HOLDOUT_PATH)) {
    stop(
      "Missing shared holdout manifest: ", HOLDOUT_PATH,
      "\nRun 10_generate_holdout_manifests.py first."
    )
  }
  manifest <- fread(HOLDOUT_PATH)
  required <- c(
    "split", "holdout_mode", "date", "year", "pass", "file_id",
    "source_file", KEY, "observed"
  )
  missing <- setdiff(required, names(manifest))
  if (length(missing) > 0L) {
    stop("Holdout manifest missing columns: ", paste(missing, collapse = ", "))
  }
  manifest[, date := as.IDate(date)]
  manifest[, pass := tolower(as.character(pass))]
  manifest[, (KEY) := as.character(get(KEY))]
  manifest[, observed := as_num(observed)]
  manifest
}


process_group <- function(group_manifest) {
  path <- resolve_source_file(group_manifest)
  pass_name <- as.character(group_manifest$pass[1])
  holdout_mode <- as.character(group_manifest$holdout_mode[1])

  dt <- fread(path)
  dt <- add_basic_columns(dt, pass_name, path)
  dt[, (TARGET) := as_num(get(TARGET))]

  target_keys <- unique(as.character(group_manifest[[KEY]]))
  observed_dt <- dt[is.finite(get(TARGET)) & !(get(KEY) %in% target_keys)]
  target_dt <- dt[get(KEY) %in% target_keys]

  if (nrow(target_dt) != nrow(group_manifest)) {
    missing_keys <- setdiff(target_keys, target_dt[[KEY]])
    stop(
      "Not all manifest targets were found in ", path,
      ". Missing examples: ", paste(head(missing_keys, 10L), collapse = ", ")
    )
  }

  target_dt <- merge(
    target_dt,
    group_manifest[, .(manifest_observed = observed), by = c(KEY)],
    by = KEY,
    all.x = TRUE,
    sort = FALSE
  )
  if (any(!is.finite(target_dt$manifest_observed))) {
    stop("Manifest observed values are missing for ", path)
  }
  if (any(abs(target_dt[[TARGET]] - target_dt$manifest_observed) > 1e-12, na.rm = TRUE)) {
    stop("Manifest observed values do not match the complete file: ", path)
  }

  target_dt[, row_id := .I]
  predictions <- predict_all_geostat_methods(observed_dt, target_dt)
  metadata <- target_dt[, .(
    row_id,
    split = "validation",
    holdout_mode = holdout_mode,
    date,
    year,
    pass,
    file_id,
    source_file = path,
    smap_pixel_key = get(KEY),
    observed = get(TARGET),
    x = as_num(x),
    y = as_num(y)
  )]
  merge(metadata, predictions, by = "row_id", all.x = FALSE, all.y = TRUE, sort = FALSE)
}


make_metrics <- function(predictions, group_columns) {
  predictions[
    ,
    {
      metric <- compute_metrics(observed, prediction)
      list(
        rmse = metric$rmse,
        mae = metric$mae,
        bias = metric$bias,
        r2 = metric$r2,
        n = metric$n,
        n_targets = .N,
        coverage = metric$n / .N
      )
    },
    by = c(group_columns, "method")
  ][order(holdout_mode, rmse, method)]
}


main <- function() {
  message("10b: GI validation on shared 2024 holdouts")
  message(strrep("=", 78))
  message("Project root:     ", PROJECT_ROOT)
  message("Holdout manifest: ", HOLDOUT_PATH)
  message("Project seed:     ", RANDOM_SEED)
  message(strrep("=", 78))

  manifest <- read_holdout_manifest()
  groups <- split(manifest, interaction(manifest$file_id, manifest$holdout_mode, drop = TRUE))
  prediction_parts <- vector("list", length(groups))

  for (i in seq_along(groups)) {
    prediction_parts[[i]] <- process_group(as.data.table(groups[[i]]))
    if (i %% 100L == 0L) message("  processed ", i, " / ", length(groups), " retrieval/mode groups")
  }

  predictions <- rbindlist(prediction_parts, use.names = TRUE, fill = TRUE)
  setorder(predictions, holdout_mode, date, pass, smap_pixel_key, method)
  metrics <- make_metrics(predictions, c("split", "holdout_mode"))
  daily_metrics <- make_metrics(predictions, c("split", "holdout_mode", "date", "pass"))
  coverage <- predictions[
    ,
    .(
      n_targets = .N,
      n_predictions = sum(is.finite(prediction)),
      coverage = mean(is.finite(prediction)),
      n_failed = sum(!is.finite(prediction))
    ),
    by = .(holdout_mode, method)
  ][order(holdout_mode, method)]

  fwrite(predictions, PREDICTION_PATH)
  fwrite(metrics, METRICS_PATH)
  fwrite(daily_metrics, DAILY_METRICS_PATH)
  fwrite(coverage, COVERAGE_PATH)

  message("\nSaved:")
  message("  ", PREDICTION_PATH)
  message("  ", METRICS_PATH)
  message("  ", DAILY_METRICS_PATH)
  message("  ", COVERAGE_PATH)
  message("\nPooled validation metrics:")
  print(metrics)
}


main()
