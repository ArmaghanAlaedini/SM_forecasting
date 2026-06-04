from pathlib import Path
import importlib.util
import os
import warnings

try:
    from pyproj import datadir
    os.environ["PROJ_DATA"] = datadir.get_data_dir()
except Exception:
    pass

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from shapely.geometry import Point
from pykrige.ok import OrdinaryKriging


# ============================================================
# 0. Load config
# ============================================================

def load_config():
    config_path = Path(__file__).resolve().with_name("00_config.py")

    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)

    if spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    spec.loader.exec_module(cfg)
    return cfg


cfg = load_config()


# ============================================================
# 1. Input paths
# ============================================================

LATTICE_PATH = cfg.SMAP_LATTICE_DIR / "smap_lattice_iowa.parquet"


# ============================================================
# 2. Load data
# ============================================================

def get_station_file_path() -> Path:
    """
    Use stations_full.csv if available.
    Otherwise, use full_stations.csv.
    """
    if cfg.IEM_STATIONS_FULL_PATH.exists():
        return cfg.IEM_STATIONS_FULL_PATH

    if cfg.IEM_STATIONS_FULL_FALLBACK_PATH.exists():
        return cfg.IEM_STATIONS_FULL_FALLBACK_PATH

    raise FileNotFoundError(
        "Could not find station file at either:\n"
        f"{cfg.IEM_STATIONS_FULL_PATH}\n"
        f"{cfg.IEM_STATIONS_FULL_FALLBACK_PATH}"
    )


def load_lattice() -> gpd.GeoDataFrame:
    """
    Load SMAP lattice polygons.
    """
    if not LATTICE_PATH.exists():
        raise FileNotFoundError(f"SMAP lattice not found:\n{LATTICE_PATH}")

    lattice = gpd.read_parquet(LATTICE_PATH)

    if lattice.crs is None:
        lattice = lattice.set_crs(cfg.CRS_EASE)

    return lattice


def load_stations() -> pd.DataFrame:
    """
    Load processed IEM station data.
    """
    station_path = get_station_file_path()

    stations = pd.read_csv(station_path)

    if "valid" not in stations.columns:
        raise ValueError("Station file must contain a 'valid' date column.")

    if "lat" not in stations.columns or "lon" not in stations.columns:
        raise ValueError("Station file must contain 'lat' and 'lon' columns.")

    stations["valid"] = pd.to_datetime(stations["valid"], errors="coerce")

    stations = stations.dropna(subset=["valid", "lat", "lon"]).copy()

    stations = stations[
        stations["valid"].dt.year.isin(cfg.ALL_YEARS)
    ].copy()

    return stations


# ============================================================
# 3. Point samples inside SMAP polygons
# ============================================================

def make_polygon_sample_points(lattice: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make sample points inside each SMAP polygon.

    five_point mode:
        center, east, west, north, south

    centroid mode:
        center only

    The kriged point predictions are averaged by polygon.
    """
    records = []

    if cfg.IEM_PTA_SAMPLE_MODE == "centroid":
        offsets = [(0.0, 0.0, "center")]
    else:
        d = cfg.SMAP_CELLSIZE_M * 0.30
        offsets = [
            (0.0, 0.0, "center"),
            (d, 0.0, "east"),
            (-d, 0.0, "west"),
            (0.0, d, "north"),
            (0.0, -d, "south"),
        ]

    for row in lattice.itertuples(index=False):
        for dx, dy, label in offsets:
            sx = row.x + dx
            sy = row.y + dy

            records.append({
                "smap_pixel_key": row.smap_pixel_key,
                "sample_label": label,
                "sample_x": sx,
                "sample_y": sy,
                "geometry": Point(sx, sy),
            })

    samples = gpd.GeoDataFrame(records, geometry="geometry", crs=cfg.CRS_EASE)

    return samples


# ============================================================
# 4. Station preparation
# ============================================================

def prepare_station_day(
    stations: pd.DataFrame,
    date_value: pd.Timestamp,
    variable: str,
) -> gpd.GeoDataFrame:
    """
    Prepare station data for one date and one variable.
    """
    if variable not in stations.columns:
        raise ValueError(f"Variable not found in station file: {variable}")

    day = stations.loc[stations["valid"] == date_value].copy()

    day[variable] = pd.to_numeric(day[variable], errors="coerce")
    day[variable] = day[variable].replace(cfg.IEM_MISSING_VALUE, np.nan)

    day = day.dropna(subset=["lat", "lon", variable]).copy()

    day = day[
        day["lon"].between(-180, 180)
        & day["lat"].between(-90, 90)
    ].copy()

    gdf = gpd.GeoDataFrame(
        day,
        geometry=gpd.points_from_xy(day["lon"], day["lat"]),
        crs=cfg.CRS_WGS84,
    ).to_crs(cfg.CRS_EASE)

    gdf["x"] = gdf.geometry.x
    gdf["y"] = gdf.geometry.y
    gdf["z"] = gdf[variable].astype(float)

    return gdf


# ============================================================
# 5. Spatial trend model
# ============================================================

def make_trend_features(
    x: np.ndarray,
    y: np.ndarray,
    x_mean: float,
    y_mean: float,
) -> pd.DataFrame:
    """
    Reduced quadratic spatial trend using centered kilometer coordinates.
    """
    x0 = (x / 1000.0) - x_mean
    y0 = (y / 1000.0) - y_mean

    return pd.DataFrame({
        "x": x0,
        "y": y0,
        "x2": x0 ** 2,
        "y2": y0 ** 2,
        "xy": x0 * y0,
    })


def fit_trend_or_mean(stations_gdf: gpd.GeoDataFrame):
    """
    Fit spatial trend.

    Use trend only if:
        F-test p-value < 0.05
        R2 > 0.01

    Otherwise, use the station mean.
    """
    x = stations_gdf["x"].to_numpy()
    y = stations_gdf["y"].to_numpy()
    z = stations_gdf["z"].to_numpy()

    x_mean = np.nanmean(x / 1000.0)
    y_mean = np.nanmean(y / 1000.0)

    X = make_trend_features(x, y, x_mean, y_mean)
    X = sm.add_constant(X, has_constant="add")

    model = sm.OLS(z, X).fit()

    trend_used = (
        np.isfinite(model.f_pvalue)
        and model.f_pvalue < 0.05
        and model.rsquared > 0.01
    )

    info = {
        "trend_used": bool(trend_used),
        "trend_r2": float(model.rsquared),
        "trend_f_pvalue": float(model.f_pvalue) if np.isfinite(model.f_pvalue) else np.nan,
        "x_mean_km": float(x_mean),
        "y_mean_km": float(y_mean),
        "mean_value": float(np.nanmean(z)),
    }

    return model, trend_used, info


def predict_trend(model, trend_used: bool, info: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Predict trend at new x/y.
    """
    if not trend_used:
        return np.full(len(x), info["mean_value"], dtype=float)

    X_new = make_trend_features(
        x=x,
        y=y,
        x_mean=info["x_mean_km"],
        y_mean=info["y_mean_km"],
    )

    X_new = sm.add_constant(X_new, has_constant="add")

    return np.asarray(model.predict(X_new), dtype=float)


# ============================================================
# 6. Ordinary kriging of residuals
# ============================================================

def krige_residuals(
    station_gdf: gpd.GeoDataFrame,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ordinary krige station residuals to target points.
    """
    OK = OrdinaryKriging(
        station_gdf["x"].to_numpy(),
        station_gdf["y"].to_numpy(),
        station_gdf["residual"].to_numpy(),
        variogram_model=cfg.IEM_VARIIOGRAM_MODEL,
        verbose=False,
        enable_plotting=False,
        coordinates_type="euclidean",
    )

    z_pred, z_var = OK.execute("points", target_x, target_y)

    return np.asarray(z_pred, dtype=float), np.asarray(z_var, dtype=float)


# ============================================================
# 7. One date-variable PTA kriging
# ============================================================

def pta_one_date_variable(
    stations: pd.DataFrame,
    samples: gpd.GeoDataFrame,
    date_value: pd.Timestamp,
    variable: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Point-to-area krige one variable for one date.

    Steps:
        1. prepare station values
        2. fit trend or mean
        3. subtract trend at stations
        4. ordinary krige residuals to SMAP sample points
        5. add trend back at sample points
        6. average sample predictions by SMAP polygon
    """
    summary = {
        "date": date_value.strftime("%Y-%m-%d"),
        "variable": variable,
        "n_stations": 0,
        "status": "ok",
        "message": "",
        "trend_used": pd.NA,
        "trend_r2": pd.NA,
        "trend_f_pvalue": pd.NA,
    }

    if variable not in stations.columns:
        summary["status"] = "skipped_missing_column"
        summary["message"] = f"{variable} not in station file"

        empty = pd.DataFrame({
            "smap_pixel_key": samples["smap_pixel_key"].drop_duplicates(),
            f"{variable}_pta": np.nan,
            f"{variable}_pta_var": np.nan,
            f"{variable}_n_samples": 0,
        })

        return empty, summary

    station_day = prepare_station_day(stations, date_value, variable)
    n_stations = len(station_day)
    summary["n_stations"] = n_stations

    if n_stations < cfg.MIN_STATIONS_FOR_KRIGING:
        summary["status"] = "skipped_too_few_stations"
        summary["message"] = f"Only {n_stations} valid stations"

        empty = pd.DataFrame({
            "smap_pixel_key": samples["smap_pixel_key"].drop_duplicates(),
            f"{variable}_pta": np.nan,
            f"{variable}_pta_var": np.nan,
            f"{variable}_n_samples": 0,
        })

        return empty, summary

    try:
        model, trend_used, trend_info = fit_trend_or_mean(station_day)

        summary["trend_used"] = trend_info["trend_used"]
        summary["trend_r2"] = trend_info["trend_r2"]
        summary["trend_f_pvalue"] = trend_info["trend_f_pvalue"]

        station_trend = predict_trend(
            model=model,
            trend_used=trend_used,
            info=trend_info,
            x=station_day["x"].to_numpy(),
            y=station_day["y"].to_numpy(),
        )

        station_day = station_day.copy()
        station_day["trend"] = station_trend
        station_day["residual"] = station_day["z"] - station_day["trend"]

        target_x = samples["sample_x"].to_numpy()
        target_y = samples["sample_y"].to_numpy()

        target_trend = predict_trend(
            model=model,
            trend_used=trend_used,
            info=trend_info,
            x=target_x,
            y=target_y,
        )

        residual_pred, residual_var = krige_residuals(
            station_gdf=station_day,
            target_x=target_x,
            target_y=target_y,
        )

        sample_pred = samples[["smap_pixel_key", "sample_label"]].copy()
        sample_pred["pred"] = target_trend + residual_pred
        sample_pred["var"] = residual_var

        area_pred = (
            sample_pred
            .groupby("smap_pixel_key", as_index=False)
            .agg(
                **{
                    f"{variable}_pta": ("pred", "mean"),
                    f"{variable}_pta_var": ("var", "mean"),
                    f"{variable}_n_samples": ("pred", "count"),
                }
            )
        )

        return area_pred, summary

    except Exception as exc:
        summary["status"] = "failed"
        summary["message"] = str(exc)

        failed = pd.DataFrame({
            "smap_pixel_key": samples["smap_pixel_key"].drop_duplicates(),
            f"{variable}_pta": np.nan,
            f"{variable}_pta_var": np.nan,
            f"{variable}_n_samples": 0,
        })

        return failed, summary


# ============================================================
# 8. One date processing
# ============================================================

def process_one_date(
    stations: pd.DataFrame,
    lattice: gpd.GeoDataFrame,
    samples: gpd.GeoDataFrame,
    date_value: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Process all configured variables for one date.
    """
    date_str = date_value.strftime("%Y-%m-%d")

    out = lattice[["smap_pixel_key", "lon", "lat", "x", "y", "geometry"]].copy()
    out["date"] = date_str

    summaries = []

    for variable in cfg.IEM_PTA_VARIABLES:
        print(f"    variable: {variable}")

        pred, summary = pta_one_date_variable(
            stations=stations,
            samples=samples,
            date_value=date_value,
            variable=variable,
        )

        out = out.merge(pred, on="smap_pixel_key", how="left")
        summaries.append(summary)

    out["geometry_wkt"] = out.geometry.to_wkt()
    out = pd.DataFrame(out.drop(columns="geometry"))

    return out, summaries


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    cfg.print_config_summary()

    print("\nLoading data")
    print("-" * 60)

    lattice = load_lattice()
    stations = load_stations()
    samples = make_polygon_sample_points(lattice)

    print(f"SMAP polygons:        {len(lattice)}")
    print(f"Polygon samples:      {len(samples)}")
    print(f"IEM station records:  {len(stations)}")

    dates = sorted(stations["valid"].dropna().unique())

    if cfg.MAX_DAYS is not None:
        dates = dates[:cfg.MAX_DAYS]

    print(f"Dates to process:     {len(dates)}")
    print("-" * 60)

    cfg.IEM_PTA_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for i, date_value in enumerate(dates, start=1):
        date_value = pd.Timestamp(date_value)
        date_yyyymmdd = date_value.strftime("%Y%m%d")

        print(f"\n[{i}/{len(dates)}] Processing {date_value.strftime('%Y-%m-%d')}")

        day_df, summaries = process_one_date(
            stations=stations,
            lattice=lattice,
            samples=samples,
            date_value=date_value,
        )

        out_path = cfg.get_iem_pta_daily_csv_path(date_yyyymmdd)
        day_df.to_csv(out_path, index=False)

        all_summaries.extend(summaries)

        print(f"  saved: {out_path}")

    summary_df = pd.DataFrame(all_summaries)
    summary_path = cfg.IEM_PTA_DIR / "iem_pta_kriging_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Daily CSV files saved in: {cfg.IEM_PTA_DIR}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()