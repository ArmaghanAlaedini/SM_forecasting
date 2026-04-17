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
      stop("Could not find project root while bootstrapping 00_config.r")
    }

    path <- parent
  }
}

PROJECT_ROOT_BOOTSTRAP <- find_project_root_bootstrap()

source(file.path(PROJECT_ROOT_BOOTSTRAP, "src", "config", "paths.r"), local = FALSE)
ensure_dirs()

smap_cfg <- list(
  # File scanning
  file_pattern = "\\.(nc4|nc)$",
  recursive = FALSE,

  # Write behavior
  overwrite = TRUE,
  drop_na = TRUE,

  # Soil moisture validity range
  value_min = 0,
  value_max = 1,

  # Local parallel defaults for Linux
  parallel_cores = max(1L, parallel::detectCores(logical = FALSE) - 1L),

  # Pull first YYYYMMDD-like date from filename
  date_regex = "(19|20)\\d{6}",

  # AM variable candidates
  am_sm_candidates = c(
    "Soil_Moisture_Retrieval_Data_AM/soil_moisture",
    "Soil_Moisture_Retrieval_Data_AM/soil_moisture_dca",
    "Soil_Moisture_Retrieval_Data_AM/soil_moisture_scah",
    "Soil_Moisture_Retrieval_Data_AM/soil_moisture_scav",
    "Geophysical_Data_AM/soil_moisture",
    "soil_moisture_am",
    "sm_am"
  ),

  am_lon_candidates = c(
    "Soil_Moisture_Retrieval_Data_AM/longitude",
    "Soil_Moisture_Retrieval_Data_AM/longitude_centroid",
    "Geolocation_Data_AM/longitude",
    "longitude",
    "lon"
  ),

  am_lat_candidates = c(
    "Soil_Moisture_Retrieval_Data_AM/latitude",
    "Soil_Moisture_Retrieval_Data_AM/latitude_centroid",
    "Geolocation_Data_AM/latitude",
    "latitude",
    "lat"
  ),

  # PM variable candidates
  pm_sm_candidates = c(
    "Soil_Moisture_Retrieval_Data_PM/soil_moisture_pm",
    "Soil_Moisture_Retrieval_Data_PM/soil_moisture_dca_pm",
    "Soil_Moisture_Retrieval_Data_PM/soil_moisture_scah_pm",
    "Soil_Moisture_Retrieval_Data_PM/soil_moisture_scav_pm",
    "Geophysical_Data_PM/soil_moisture_pm",
    "soil_moisture_pm",
    "sm_pm"
  ),

  pm_lon_candidates = c(
    "Soil_Moisture_Retrieval_Data_PM/longitude_pm",
    "Soil_Moisture_Retrieval_Data_PM/longitude_centroid_pm",
    "Geolocation_Data_PM/longitude_pm",
    "longitude_pm",
    "lon_pm"
  ),

  pm_lat_candidates = c(
    "Soil_Moisture_Retrieval_Data_PM/latitude_pm",
    "Soil_Moisture_Retrieval_Data_PM/latitude_centroid_pm",
    "Geolocation_Data_PM/latitude_pm",
    "latitude_pm",
    "lat_pm"
  ),

  # CRS + GRID SIZE
  crs_wgs84 = 4326,
  crs_ease  = 6933,
  smap_cellsize = 9024.31,
  
  # DISCRETIZATION
  disc_obs = 3000,
  disc_twn = 3000,
  
  # PLOT LIMITS
  lims_sm_global = c(0.01, 0.51),
  
  # VARIOGRAM / DECONV CONFIG
  vgm_model = "Exp",
  ngroup = 12,
  rd = 0.4,
  maxIter = 1000,
  maxSampleNum = 1000
  
  )