find_project_root_bootstrap <- function(start = NULL) {
  if (is.null(start)) {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("^--file=", args, value = TRUE)

    if (length(file_arg) > 0) {
      script_path <- sub("^--file=", "", file_arg[1])
      start <- normalizePath(script_path, winslash = "/", mustWork = TRUE)
    } else {
      start <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
    }
  } else {
    start <- normalizePath(start, winslash = "/", mustWork = TRUE)
  }

  path <- if (dir.exists(start)) start else dirname(start)

  repeat {
    if (dir.exists(file.path(path, ".git"))) {
      return(path)
    }

    if (file.exists(file.path(path, "environment.yml")) &&
        dir.exists(file.path(path, "src"))) {
      return(path)
    }

    parent <- dirname(path)
    if (identical(parent, path)) {
      stop("Could not find project root while bootstrapping 02_run_extract_parallel.r")
    }

    path <- parent
  }
}

PROJECT_ROOT_BOOTSTRAP <- find_project_root_bootstrap()
source(file.path(PROJECT_ROOT_BOOTSTRAP, "src", "code", "smap", "01_smap_nc4_extract.r"), local = FALSE)

run_parallel_extract <- function(cores = smap_cfg$parallel_cores, limit = NULL) {
  if (.Platform$OS.type == "windows") {
    stop("This parallel runner uses mclapply() and is intended for Linux/macOS.")
  }

  nc_files <- get_nc_files(limit = limit)

  cores <- as.integer(cores)
  if (is.na(cores) || cores < 1L) {
    stop("`cores` must be a positive integer.")
  }

  cores <- min(cores, length(nc_files))

  if (cores == 1L) {
    message("cores = 1, falling back to serial execution.")
    return(run_serial_extract(limit = limit))
  }

  message("Found ", length(nc_files), " nc file(s).")
  message("Running in parallel with ", cores, " core(s).")

  results <- parallel::mclapply(
    X = nc_files,
    FUN = safe_process_one_nc4,
    mc.cores = cores,
    mc.preschedule = FALSE
  )

  results <- do.call(rbind, results)
  rownames(results) <- NULL

  message("\nProcessing summary:")
  print(results, row.names = FALSE)

  invisible(results)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)

  cores <- if (length(args) >= 1L) as.integer(args[1]) else smap_cfg$parallel_cores
  limit <- if (length(args) >= 2L) as.integer(args[2]) else NULL

  results <- run_parallel_extract(cores = cores, limit = limit)
}