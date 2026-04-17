source("src/code/smap/00_config.r")

if (!requireNamespace("sf", quietly = TRUE)) {
  stop("Package 'sf' is required.")
}

library(sf)

required_cfg <- c(
  "PROC_SMAP_AM_CSV",
  "PROC_SMAP_PM_CSV",
  "DTRND_SMAP_AM_RDS",
  "DTRND_SMAP_PM_RDS"
)

missing_cfg <- setdiff(required_cfg, names(cfg))
if (length(missing_cfg) > 0L) {
  stop(
    "Missing required path(s) in cfg: ",
    paste(missing_cfg, collapse = ", "),
    ". Add them to src/config/paths.r first."
  )
}

make_log_row <- function(
  infile,
  pass,
  rows = NA_integer_,
  output_file = NA_character_,
  trend_removed = NA,
  p_trend = NA_real_,
  r2_trend = NA_real_,
  status = "written",
  note = NA_character_
) {
  data.frame(
    source_file = basename(infile),
    pass = pass,
    rows = rows,
    output_file = output_file,
    trend_removed = trend_removed,
    p_trend = p_trend,
    r2_trend = r2_trend,
    status = status,
    note = note,
    stringsAsFactors = FALSE
  )
}

get_input_files <- function(pass = c("am", "pm"), limit = NULL) {
  pass <- match.arg(pass)

  in_dir <- if (pass == "am") cfg$PROC_SMAP_AM_CSV else cfg$PROC_SMAP_PM_CSV

  files <- list.files(
    path = in_dir,
    pattern = "\\.csv$",
    full.names = TRUE
  )

  files <- sort(files)

  if (length(files) == 0L) {
    stop("No extracted ", pass, " CSV files found in: ", in_dir)
  }

  if (!is.null(limit)) {
    limit <- as.integer(limit)
    files <- files[seq_len(min(limit, length(files)))]
  }

  files
}

get_output_dir <- function(pass = c("am", "pm")) {
  pass <- match.arg(pass)
  if (pass == "am") cfg$DTRND_SMAP_AM_RDS else cfg$DTRND_SMAP_PM_RDS
}

build_output_file <- function(infile, pass = c("am", "pm")) {
  pass <- match.arg(pass)

  file.path(
    get_output_dir(pass),
    paste0(tools::file_path_sans_ext(basename(infile)), ".rds")
  )
}

make_square <- function(x, y, h) {
  st_polygon(list(matrix(
    c(
      x - h, y - h,
      x + h, y - h,
      x + h, y + h,
      x - h, y + h,
      x - h, y - h
    ),
    ncol = 2,
    byrow = TRUE
  )))
}

process_one_detrend_file <- function(infile, overwrite = smap_cfg$overwrite) {
  sm <- read.csv(infile, stringsAsFactors = FALSE)

  required_cols <- c("source_file", "date_tag", "pass", "lon", "lat", "soil_moisture")
  missing_cols <- setdiff(required_cols, names(sm))
  if (length(missing_cols) > 0L) {
    stop(
      "Missing required columns in ", basename(infile), ": ",
      paste(missing_cols, collapse = ", ")
    )
  }

  pass_guess <- if (grepl("_pm\\.csv$", infile, ignore.case = TRUE)) "pm" else "am"
  out_file <- build_output_file(infile, pass = pass_guess)

  if (file.exists(out_file) && !isTRUE(overwrite)) {
    return(make_log_row(
      infile = infile,
      pass = pass_guess,
      rows = NA_integer_,
      output_file = out_file,
      status = "skipped_existing"
    ))
  }

  # Handle empty extracted CSVs before checking pass values
  if (nrow(sm) == 0L) {
    empty_obj <- st_sf(
      data.frame(
        source_file = character(0),
        date_tag = character(0),
        pass = character(0),
        lon = numeric(0),
        lat = numeric(0),
        x = numeric(0),
        y = numeric(0),
        soil_moisture = numeric(0),
        trend_hat = numeric(0),
        resid = numeric(0),
        sm_for_kriging = numeric(0),
        trend_removed = logical(0),
        p_trend = numeric(0),
        r2_trend = numeric(0),
        trend_ind = integer(0),
        pixel_id = integer(0),
        stringsAsFactors = FALSE
      ),
      geometry = st_sfc(crs = smap_cfg$crs_ease)
    )

    saveRDS(empty_obj, out_file)

    return(make_log_row(
      infile = infile,
      pass = pass_guess,
      rows = 0L,
      output_file = out_file,
      trend_removed = NA,
      p_trend = NA_real_,
      r2_trend = NA_real_,
      status = "written_empty",
      note = "empty_input_csv"
    ))
  }

  pass_values <- unique(tolower(sm$pass))
  if (length(pass_values) != 1L) {
    stop(
      "Expected exactly one pass in file ", basename(infile),
      ", found: ", paste(pass_values, collapse = ", ")
    )
  }
  pass <- pass_values[1]

  if (!identical(pass, pass_guess)) {
    warning(
      "Pass inferred from filename (", pass_guess,
      ") does not match pass column (", pass,
      ") in ", basename(infile), ". Using pass column."
    )
    out_file <- build_output_file(infile, pass = pass)
  }

  sm_sf <- st_as_sf(
    sm,
    coords = c("lon", "lat"),
    crs = smap_cfg$crs_wgs84,
    remove = FALSE
  )

  sm_proj <- st_transform(sm_sf, crs = smap_cfg$crs_ease)

  xy <- st_coordinates(sm_proj)
  sm_proj$x <- xy[, 1]
  sm_proj$y <- xy[, 2]

  sm_df <- st_drop_geometry(sm_proj)
  observed <- sm_df$soil_moisture

  if (nrow(sm_df) < 6L || length(unique(observed)) < 2L) {
    m0 <- lm(soil_moisture ~ 1, data = sm_df)

    trend_hat <- as.numeric(predict(m0, newdata = sm_df))
    resid <- observed - trend_hat
    sm_for_kriging <- observed

    trend_removed <- FALSE
    p_trend <- NA_real_
    r2_trend <- NA_real_
    note <- "small_n_or_constant"
  } else {
    m0 <- lm(soil_moisture ~ 1, data = sm_df)

    m_start <- lm(
      soil_moisture ~ x + y + I(x^2) + I(y^2) + I(x * y),
      data = sm_df
    )

    m_sel <- step(
      m_start,
      direction = "backward",
      scope = list(
        lower = ~ x + y,
        upper = ~ x + y + I(x^2) + I(y^2) + I(x * y)
      ),
      trace = 0
    )

    trend_test <- anova(m0, m_sel)

    p_trend <- as.numeric(trend_test[2, "Pr(>F)"])
    r2_trend <- summary(m_sel)$r.squared

    trend_removed <- !is.na(p_trend) && p_trend < 0.05 && r2_trend > 0.01

    if (trend_removed) {
      trend_hat <- as.numeric(predict(m_sel, newdata = sm_df))
      resid <- observed - trend_hat
      sm_for_kriging <- resid
    } else {
      trend_hat <- as.numeric(predict(m0, newdata = sm_df))
      resid <- observed - trend_hat
      sm_for_kriging <- observed
    }

    note <- "model_fitted"
  }

  recon_error <- max(abs(observed - (trend_hat + resid)))
  if (!isTRUE(all.equal(recon_error, 0, tolerance = 1e-10))) {
    warning(
      "Reconstruction check is not zero for ",
      basename(infile),
      " (max error = ", recon_error, ")"
    )
  }

  sm_proj$trend_hat <- trend_hat
  sm_proj$resid <- resid
  sm_proj$sm_for_kriging <- sm_for_kriging
  sm_proj$trend_removed <- trend_removed
  sm_proj$p_trend <- p_trend
  sm_proj$r2_trend <- r2_trend
  sm_proj$trend_ind <- as.integer(trend_removed)

  half <- smap_cfg$smap_cellsize / 2

  polys <- st_sfc(
    lapply(seq_len(nrow(sm_proj)), function(i) {
      make_square(xy[i, 1], xy[i, 2], half)
    }),
    crs = smap_cfg$crs_ease
  )

  cells_ease <- sm_proj
  cells_ease$pixel_id <- seq_len(nrow(cells_ease))
  st_geometry(cells_ease) <- polys

  saveRDS(cells_ease, out_file)

  make_log_row(
    infile = infile,
    pass = pass,
    rows = nrow(cells_ease),
    output_file = out_file,
    trend_removed = trend_removed,
    p_trend = p_trend,
    r2_trend = r2_trend,
    status = "written",
    note = note
  )
}

safe_process_one_detrend_file <- function(infile, overwrite = smap_cfg$overwrite) {
  pass_guess <- if (grepl("_pm\\.csv$", infile, ignore.case = TRUE)) "pm" else "am"

  tryCatch(
    process_one_detrend_file(infile, overwrite = overwrite),
    error = function(e) {
      make_log_row(
        infile = infile,
        pass = pass_guess,
        rows = NA_integer_,
        output_file = NA_character_,
        trend_removed = NA,
        p_trend = NA_real_,
        r2_trend = NA_real_,
        status = "error",
        note = conditionMessage(e)
      )
    }
  )
}

run_detrend_serial <- function(pass = c("am", "pm"), limit = NULL, overwrite = smap_cfg$overwrite) {
  pass <- match.arg(pass)

  files <- get_input_files(pass = pass, limit = limit)

  message("Found ", length(files), " ", pass, " file(s).")

  logs <- lapply(files, safe_process_one_detrend_file, overwrite = overwrite)
  logs <- do.call(rbind, logs)
  rownames(logs) <- NULL

  print(logs, row.names = FALSE)
  invisible(logs)
}

run_detrend_parallel <- function(pass = c("am", "pm"),
                                 cores = smap_cfg$parallel_cores,
                                 limit = NULL,
                                 overwrite = smap_cfg$overwrite) {
  pass <- match.arg(pass)

  if (.Platform$OS.type == "windows") {
    stop("This parallel runner uses mclapply() and is intended for Linux/macOS.")
  }

  files <- get_input_files(pass = pass, limit = limit)

  cores <- as.integer(cores)
  if (is.na(cores) || cores < 1L) {
    stop("`cores` must be a positive integer.")
  }

  cores <- min(cores, length(files))

  if (cores == 1L) {
    message("cores = 1, falling back to serial execution.")
    return(run_detrend_serial(pass = pass, limit = limit, overwrite = overwrite))
  }

  message("Found ", length(files), " ", pass, " file(s).")
  message("Running in parallel with ", cores, " core(s).")

  logs <- parallel::mclapply(
    X = files,
    FUN = safe_process_one_detrend_file,
    overwrite = overwrite,
    mc.cores = cores,
    mc.preschedule = FALSE
  )

  logs <- do.call(rbind, logs)
  rownames(logs) <- NULL

  print(logs, row.names = FALSE)
  invisible(logs)
}

run_detrend_both_serial <- function(limit = NULL, overwrite = smap_cfg$overwrite) {
  am_log <- run_detrend_serial(pass = "am", limit = limit, overwrite = overwrite)
  pm_log <- run_detrend_serial(pass = "pm", limit = limit, overwrite = overwrite)

  logs <- rbind(am_log, pm_log)
  rownames(logs) <- NULL
  invisible(logs)
}

run_detrend_both_parallel <- function(cores = smap_cfg$parallel_cores,
                                      limit = NULL,
                                      overwrite = smap_cfg$overwrite) {
  am_log <- run_detrend_parallel(pass = "am", cores = cores, limit = limit, overwrite = overwrite)
  pm_log <- run_detrend_parallel(pass = "pm", cores = cores, limit = limit, overwrite = overwrite)

  logs <- rbind(am_log, pm_log)
  rownames(logs) <- NULL
  invisible(logs)
}

system.time(run_detrend_both_parallel(cores = 4, limit = 100, overwrite = TRUE))
system.time(run_detrend_both_parallel(cores = 8, limit = 100, overwrite = TRUE))
log4 <- run_detrend_both_parallel(cores = 4, limit = 100, overwrite = TRUE)
table(log4$status)


run_detrend_both_parallel(cores = 8)

length(list.files(cfg$PROC_SMAP_AM_CSV, pattern = "\\.csv$"))
length(list.files(cfg$DTRND_SMAP_AM_RDS, pattern = "\\.rds$"))

length(list.files(cfg$PROC_SMAP_PM_CSV, pattern = "\\.csv$"))
length(list.files(cfg$DTRND_SMAP_PM_RDS, pattern = "\\.rds$"))