find_project_root <- function(start = NULL) {
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
      stop(
        "Could not find project root. Start R in the project root or run the script from inside the repository."
      )
    }

    path <- parent
  }
}

ROOT <- find_project_root()

SRC <- file.path(ROOT, "src")
CODE <- file.path(SRC, "code")
CONFIG <- file.path(SRC, "config")
DATA <- file.path(SRC, "data")

RAW <- file.path(DATA, "raw")
PROCESSED <- file.path(DATA, "processed")

RAW_DEP <- file.path(RAW, "dep")
RAW_GFS <- file.path(RAW, "gfs")
RAW_ISU <- file.path(RAW, "isu_stations")
RAW_SMAP_OBS <- file.path(RAW, "smap_observations")
RAW_TOWNSHIPS <- file.path(RAW, "townships")

PROC_DEP <- file.path(PROCESSED, "dep")
PROC_GFS <- file.path(PROCESSED, "gfs")
PROC_ISU <- file.path(PROCESSED, "isu_stations")

# SMAP extracted outputs: CSV only
PROC_SMAP <- file.path(PROCESSED, "smap_processed")
PROC_SMAP_AM <- file.path(PROC_SMAP, "am")
PROC_SMAP_PM <- file.path(PROC_SMAP, "pm")
PROC_SMAP_AM_CSV <- file.path(PROC_SMAP_AM, "csv")
PROC_SMAP_PM_CSV <- file.path(PROC_SMAP_PM, "csv")

# SMAP detrended outputs: RDS only, because I added polygons in the dataset
DTRND_SMAP <- file.path(PROCESSED, "smap_detrended")
DTRND_SMAP_AM <- file.path(DTRND_SMAP, "am")
DTRND_SMAP_PM <- file.path(DTRND_SMAP, "pm")
DTRND_SMAP_AM_RDS <- file.path(DTRND_SMAP_AM, "rds")
DTRND_SMAP_PM_RDS <- file.path(DTRND_SMAP_PM, "rds")

# Kriged outputs: CSV + RDS
PROC_KRIGED <- file.path(PROCESSED, "kriged_predictions")
PROC_KRIGED_AM <- file.path(PROC_KRIGED, "am")
PROC_KRIGED_PM <- file.path(PROC_KRIGED, "pm")
PROC_KRIGED_AM_CSV <- file.path(PROC_KRIGED_AM, "csv")
PROC_KRIGED_AM_RDS <- file.path(PROC_KRIGED_AM, "rds")
PROC_KRIGED_PM_CSV <- file.path(PROC_KRIGED_PM, "csv")
PROC_KRIGED_PM_RDS <- file.path(PROC_KRIGED_PM, "rds")

RAW_DEP_CSV <- file.path(RAW_DEP, "DEP_20260405.csv")
RAW_GFS_CSV <- file.path(RAW_GFS, "gfs.csv")
RAW_ISU_STATIONS <- file.path(RAW_ISU, "stations.csv")
RAW_ISU_META <- file.path(RAW_ISU, "stations_meta.csv")
RAW_TOWNSHIPS_SHP <- file.path(RAW_TOWNSHIPS, "civil_townships_a_ia.shp")

PROC_ISU_STATIONS_FULL <- file.path(PROC_ISU, "stations_full.csv")

cfg <- list(
  ROOT = ROOT,
  SRC = SRC,
  CODE = CODE,
  CONFIG = CONFIG,
  DATA = DATA,

  RAW = RAW,
  PROCESSED = PROCESSED,

  RAW_DEP = RAW_DEP,
  RAW_GFS = RAW_GFS,
  RAW_ISU = RAW_ISU,
  RAW_SMAP_OBS = RAW_SMAP_OBS,
  RAW_TOWNSHIPS = RAW_TOWNSHIPS,

  PROC_DEP = PROC_DEP,
  PROC_GFS = PROC_GFS,
  PROC_ISU = PROC_ISU,

  PROC_SMAP = PROC_SMAP,
  PROC_SMAP_AM = PROC_SMAP_AM,
  PROC_SMAP_PM = PROC_SMAP_PM,
  PROC_SMAP_AM_CSV = PROC_SMAP_AM_CSV,
  PROC_SMAP_PM_CSV = PROC_SMAP_PM_CSV,

  DTRND_SMAP = DTRND_SMAP,
  DTRND_SMAP_AM = DTRND_SMAP_AM,
  DTRND_SMAP_PM = DTRND_SMAP_PM,
  DTRND_SMAP_AM_RDS = DTRND_SMAP_AM_RDS,
  DTRND_SMAP_PM_RDS = DTRND_SMAP_PM_RDS,

  PROC_KRIGED = PROC_KRIGED,
  PROC_KRIGED_AM = PROC_KRIGED_AM,
  PROC_KRIGED_PM = PROC_KRIGED_PM,
  PROC_KRIGED_AM_CSV = PROC_KRIGED_AM_CSV,
  PROC_KRIGED_AM_RDS = PROC_KRIGED_AM_RDS,
  PROC_KRIGED_PM_CSV = PROC_KRIGED_PM_CSV,
  PROC_KRIGED_PM_RDS = PROC_KRIGED_PM_RDS,

  RAW_DEP_CSV = RAW_DEP_CSV,
  RAW_GFS_CSV = RAW_GFS_CSV,
  RAW_ISU_STATIONS = RAW_ISU_STATIONS,
  RAW_ISU_META = RAW_ISU_META,
  RAW_TOWNSHIPS_SHP = RAW_TOWNSHIPS_SHP,

  PROC_ISU_STATIONS_FULL = PROC_ISU_STATIONS_FULL
)

ensure_dirs <- function() {
  dirs <- unique(c(
    RAW,
    PROCESSED,

    RAW_DEP,
    RAW_GFS,
    RAW_ISU,
    RAW_SMAP_OBS,
    RAW_TOWNSHIPS,

    PROC_DEP,
    PROC_GFS,
    PROC_ISU,

    PROC_SMAP,
    PROC_SMAP_AM,
    PROC_SMAP_PM,
    PROC_SMAP_AM_CSV,
    PROC_SMAP_PM_CSV,

    DTRND_SMAP,
    DTRND_SMAP_AM,
    DTRND_SMAP_PM,
    DTRND_SMAP_AM_RDS,
    DTRND_SMAP_PM_RDS,

    PROC_KRIGED,
    PROC_KRIGED_AM,
    PROC_KRIGED_PM,
    PROC_KRIGED_AM_CSV,
    PROC_KRIGED_AM_RDS,
    PROC_KRIGED_PM_CSV,
    PROC_KRIGED_PM_RDS
  ))

  for (p in dirs) {
    if (!dir.exists(p)) {
      dir.create(p, recursive = TRUE, showWarnings = FALSE)
    }
  }

  invisible(dirs)
}