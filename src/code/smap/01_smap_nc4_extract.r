if (!exists("cfg", inherits = FALSE) || !exists("smap_cfg", inherits = FALSE)) {
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
        stop("Could not find project root while bootstrapping 01_smap_nc4_extract.r")
      }

      path <- parent
    }
  }

  PROJECT_ROOT_BOOTSTRAP <- find_project_root_bootstrap()
  source(file.path(PROJECT_ROOT_BOOTSTRAP, "src", "code", "smap", "00_config.r"), local = FALSE)
}

if (!requireNamespace("ncdf4", quietly = TRUE)) {
  stop("Package 'ncdf4' is required. Install it with renv::install('ncdf4').")
}

ensure_dirs()

available_var_names <- function(nc) {
  vars <- vapply(nc$var, function(x) x$name, character(1))
  sort(unique(vars))
}

get_nc_files <- function(limit = NULL) {
  nc_files <- list.files(
    path = cfg$RAW_SMAP_OBS,
    pattern = smap_cfg$file_pattern,
    full.names = TRUE,
    recursive = isTRUE(smap_cfg$recursive),
    ignore.case = TRUE
  )

  nc_files <- sort(nc_files)

  if (length(nc_files) == 0L) {
    stop("No .nc4 or .nc files were found in: ", cfg$RAW_SMAP_OBS)
  }

  if (!is.null(limit)) {
    limit <- as.integer(limit)
    nc_files <- nc_files[seq_len(min(limit, length(nc_files)))]
  }

  nc_files
}

show_nc4_variables <- function(nc_file = NULL) {
  if (is.null(nc_file)) {
    nc_file <- get_nc_files(limit = 1)[1]
  }

  nc <- ncdf4::nc_open(nc_file)
  on.exit(ncdf4::nc_close(nc), add = TRUE)

  vars <- available_var_names(nc)

  message("File: ", basename(nc_file))
  message("Variables found:")
  cat(paste(vars, collapse = "\n"), "\n")

  invisible(vars)
}

find_first_name <- function(available, candidates, label, required = TRUE) {
  exact_match <- candidates[candidates %in% available]
  if (length(exact_match) > 0) {
    return(exact_match[1])
  }

  if (!required) {
    return(NA_character_)
  }

  stop(
    paste0(
      "Could not find ", label, ".\n\nAvailable variables:\n",
      paste(sort(available), collapse = "\n")
    )
  )
}

extract_date_tag <- function(path, pattern = smap_cfg$date_regex) {
  hit <- regexpr(pattern, basename(path), perl = TRUE)
  if (hit[1] == -1) {
    return("")
  }

  regmatches(basename(path), hit)
}

replace_fill_values <- function(values, nc, var_name) {
  fill_att <- ncdf4::ncatt_get(nc, var_name, "_FillValue")
  if (isTRUE(fill_att$hasatt)) {
    values[values %in% as.numeric(fill_att$value)] <- NA_real_
  }

  miss_att <- ncdf4::ncatt_get(nc, var_name, "missing_value")
  if (isTRUE(miss_att$hasatt)) {
    values[values %in% as.numeric(miss_att$value)] <- NA_real_
  }

  values
}

build_coord_frame <- function(sm_dims, lon, lat) {
  if (!is.null(dim(lon)) && !is.null(dim(lat))) {
    if (identical(dim(lon), sm_dims) && identical(dim(lat), sm_dims)) {
      return(data.frame(
        lon = as.numeric(as.vector(lon)),
        lat = as.numeric(as.vector(lat))
      ))
    }

    if (identical(rev(dim(lon)), sm_dims) && identical(rev(dim(lat)), sm_dims)) {
      return(data.frame(
        lon = as.numeric(as.vector(t(lon))),
        lat = as.numeric(as.vector(t(lat)))
      ))
    }

    stop(
      "2D longitude/latitude dimensions do not match the soil-moisture dimensions.\n",
      "Soil moisture dims: ", paste(sm_dims, collapse = " x "), "\n",
      "Longitude dims: ", paste(dim(lon), collapse = " x "), "\n",
      "Latitude dims: ", paste(dim(lat), collapse = " x ")
    )
  }

  if (is.null(dim(lon)) && is.null(dim(lat))) {
    if (length(lon) == sm_dims[1] && length(lat) == sm_dims[2]) {
      return(data.frame(
        lon = rep(as.numeric(lon), times = sm_dims[2]),
        lat = rep(as.numeric(lat), each = sm_dims[1])
      ))
    }

    if (length(lat) == sm_dims[1] && length(lon) == sm_dims[2]) {
      return(data.frame(
        lon = rep(as.numeric(lon), each = sm_dims[1]),
        lat = rep(as.numeric(lat), times = sm_dims[2])
      ))
    }

    stop(
      "1D longitude/latitude lengths do not align with the soil-moisture dimensions.\n",
      "Soil moisture dims: ", paste(sm_dims, collapse = " x "), "\n",
      "Length(lon): ", length(lon), "\n",
      "Length(lat): ", length(lat)
    )
  }

  stop("Mixed lon/lat shapes are not supported. Both must be 1D or both must be 2D.")
}

extract_pass_df <- function(nc, sm_var, lon_var, lat_var, pass_label, source_file) {
  sm_array <- drop(ncdf4::ncvar_get(nc, sm_var))
  sm_dims <- dim(sm_array)

  if (is.null(sm_dims) || length(sm_dims) != 2L) {
    found_dims <- if (is.null(sm_dims)) "vector/scalar" else paste(sm_dims, collapse = " x ")
    stop("Variable '", sm_var, "' must be 2D after drop(). Found: ", found_dims)
  }

  lon <- ncdf4::ncvar_get(nc, lon_var)
  lat <- ncdf4::ncvar_get(nc, lat_var)

  coords <- build_coord_frame(sm_dims = sm_dims, lon = lon, lat = lat)

  values <- as.numeric(as.vector(sm_array))
  values <- replace_fill_values(values, nc, sm_var)

  out_of_range <- !is.na(values) & (
    values < smap_cfg$value_min | values > smap_cfg$value_max
  )
  values[out_of_range] <- NA_real_

  df <- data.frame(
    source_file = basename(source_file),
    date_tag = extract_date_tag(source_file),
    pass = pass_label,
    lon = coords$lon,
    lat = coords$lat,
    soil_moisture = values,
    stringsAsFactors = FALSE
  )

  if (isTRUE(smap_cfg$drop_na)) {
    keep <- !is.na(df$lon) & !is.na(df$lat) & !is.na(df$soil_moisture)
    df <- df[keep, , drop = FALSE]
  }

  df
}

build_output_file <- function(input_file, pass_label) {
  stem <- tools::file_path_sans_ext(basename(input_file))
  out_name <- paste0(stem, "_", tolower(pass_label), ".csv")

  if (tolower(pass_label) == "am") {
    return(file.path(cfg$PROC_SMAP_AM_CSV, out_name))
  }

  if (tolower(pass_label) == "pm") {
    return(file.path(cfg$PROC_SMAP_PM_CSV, out_name))
  }

  stop("Unknown pass label: ", pass_label)
}

write_pass_csv <- function(df, input_file, pass_label) {
  out_file <- build_output_file(input_file, pass_label)

  if (file.exists(out_file) && !isTRUE(smap_cfg$overwrite)) {
    message("Skipping existing file: ", basename(out_file))
    return(data.frame(
      source_file = basename(input_file),
      pass = pass_label,
      rows = NA_integer_,
      output_file = out_file,
      status = "skipped_existing",
      stringsAsFactors = FALSE
    ))
  }

  utils::write.csv(df, out_file, row.names = FALSE)

  data.frame(
    source_file = basename(input_file),
    pass = pass_label,
    rows = nrow(df),
    output_file = out_file,
    status = "written",
    stringsAsFactors = FALSE
  )
}

process_one_nc4 <- function(nc_file) {
  message("Processing: ", basename(nc_file))

  nc <- ncdf4::nc_open(nc_file)
  on.exit(ncdf4::nc_close(nc), add = TRUE)

  available <- available_var_names(nc)

  am_sm_var <- find_first_name(
    available = available,
    candidates = smap_cfg$am_sm_candidates,
    label = "AM soil moisture variable",
    required = FALSE
  )

  pm_sm_var <- find_first_name(
    available = available,
    candidates = smap_cfg$pm_sm_candidates,
    label = "PM soil moisture variable",
    required = FALSE
  )

  if (is.na(am_sm_var) && is.na(pm_sm_var)) {
    stop(
      "No AM or PM soil-moisture variable was found in: ", basename(nc_file),
      "\nAvailable variables:\n", paste(available, collapse = "\n")
    )
  }

  results <- list()

  if (!is.na(am_sm_var)) {
    am_lon_var <- find_first_name(
      available = available,
      candidates = smap_cfg$am_lon_candidates,
      label = "AM longitude variable"
    )

    am_lat_var <- find_first_name(
      available = available,
      candidates = smap_cfg$am_lat_candidates,
      label = "AM latitude variable"
    )

    am_df <- extract_pass_df(
      nc = nc,
      sm_var = am_sm_var,
      lon_var = am_lon_var,
      lat_var = am_lat_var,
      pass_label = "am",
      source_file = nc_file
    )

    results[[length(results) + 1L]] <- write_pass_csv(
      df = am_df,
      input_file = nc_file,
      pass_label = "am"
    )
  }

  if (!is.na(pm_sm_var)) {
    pm_lon_var <- find_first_name(
      available = available,
      candidates = smap_cfg$pm_lon_candidates,
      label = "PM longitude variable"
    )

    pm_lat_var <- find_first_name(
      available = available,
      candidates = smap_cfg$pm_lat_candidates,
      label = "PM latitude variable"
    )

    pm_df <- extract_pass_df(
      nc = nc,
      sm_var = pm_sm_var,
      lon_var = pm_lon_var,
      lat_var = pm_lat_var,
      pass_label = "pm",
      source_file = nc_file
    )

    results[[length(results) + 1L]] <- write_pass_csv(
      df = pm_df,
      input_file = nc_file,
      pass_label = "pm"
    )
  }

  do.call(rbind, results)
}

safe_process_one_nc4 <- function(nc_file) {
  tryCatch(
    process_one_nc4(nc_file),
    error = function(e) {
      data.frame(
        source_file = basename(nc_file),
        pass = NA_character_,
        rows = NA_integer_,
        output_file = NA_character_,
        status = paste("error:", conditionMessage(e)),
        stringsAsFactors = FALSE
      )
    }
  )
}

run_serial_extract <- function(limit = NULL) {
  nc_files <- get_nc_files(limit = limit)

  message("Found ", length(nc_files), " nc file(s).")

  results <- lapply(nc_files, safe_process_one_nc4)
  results <- do.call(rbind, results)
  rownames(results) <- NULL

  message("\nProcessing summary:")
  print(results, row.names = FALSE)

  invisible(results)
}

if (sys.nframe() == 0L) {
  results <- run_serial_extract()
}