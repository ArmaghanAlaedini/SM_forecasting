from pathlib import Path
import gc

import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# ============================================================
# 1. SETTINGS
# ============================================================

YEAR = 2025
DATE_TO_PLOT = "2025-06-15"

# HRRR leads from your config:
# 0  = analysis_f00 / nowcast-style
# 24 = forecast_1_day
LEAD_HOUR = 24

VARIABLES_TO_MAP = [
    "temperature_c",
    "wind_speed_10m_mps",
    "precip_accum_mm",
    "moisture_availability_percent",
    "soil_temperature_c",
]

VARIABLE_FOR_TIME_SERIES = "temperature_c"

SHOW_TOWNSHIPS = True


# ============================================================
# 2. PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

YEARLY_DIR = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "processed"
    / "archived_weather"
    / "yearly"
)

HRRR_FILE = YEARLY_DIR / f"hrrr_weather_iowa_pixels_{YEAR}.parquet"

TOWNSHIP_SHP = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "raw"
    / "townships"
    / "civil_townships_a_ia.shp"
)


# ============================================================
# 3. BASIC CHECKS
# ============================================================

print("HRRR file:")
print(HRRR_FILE)

if not HRRR_FILE.exists():
    raise FileNotFoundError(f"File not found:\n{HRRR_FILE}")

print("\nAvailable columns:")
print(pq.read_schema(HRRR_FILE).names)


# ============================================================
# 4. LOAD ONLY ONE DAY + ONE LEAD
# ============================================================

def load_one_hrrr_day(path, date, lead_hour, variables):
    base_cols = [
        "valid_date",
        "lead_hour",
        "forecast_type",
        "latitude",
        "longitude",
        "grid_id",
    ]

    available_cols = pq.read_schema(path).names
    columns = [c for c in base_cols + variables if c in available_cols]

    print("\nReading only these columns:")
    print(columns)

    df = pd.read_parquet(
        path,
        columns=columns,
        filters=[
            ("valid_date", "=", date),
            ("lead_hour", "=", lead_hour),
        ],
    )

    df["valid_date"] = pd.to_datetime(df["valid_date"])
    df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")

    print("\nLoaded rows:")
    print(len(df))

    print("\nCoordinate range:")
    print("latitude :", df["latitude"].min(), "to", df["latitude"].max())
    print("longitude:", df["longitude"].min(), "to", df["longitude"].max())

    return df


df_day = load_one_hrrr_day(
    path=HRRR_FILE,
    date=DATE_TO_PLOT,
    lead_hour=LEAD_HOUR,
    variables=VARIABLES_TO_MAP,
)

if df_day.empty:
    raise ValueError("No rows loaded. Try another DATE_TO_PLOT or LEAD_HOUR.")


# ============================================================
# 5. OPTIONAL TOWNSHIP BOUNDARIES
# ============================================================

townships = None

if SHOW_TOWNSHIPS and TOWNSHIP_SHP.exists():
    import geopandas as gpd

    townships = gpd.read_file(TOWNSHIP_SHP)

    if townships.crs is None:
        townships = townships.set_crs(epsg=4326)
    elif townships.crs.to_epsg() != 4326:
        townships = townships.to_crs(epsg=4326)

    print("\nLoaded townships:")
    print(len(townships))
else:
    print("\nTownship shapefile not loaded.")


# ============================================================
# 6. QUICK SPATIAL MAPS, NO SAVING
# ============================================================

def plot_hrrr_square_map(df, var, townships=None):
    if var not in df.columns:
        print(f"Skipping {var}: column does not exist.")
        return

    temp = df.copy()
    temp[var] = pd.to_numeric(temp[var], errors="coerce")
    temp = temp.dropna(subset=[var, "latitude", "longitude"])

    if temp.empty:
        print(f"Skipping {var}: all values are NA.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # Big square centroid markers.
    sc = ax.scatter(
        temp["longitude"],
        temp["latitude"],
        c=temp[var],
        s=55,
        marker="s",
        cmap="viridis",
        alpha=0.9,
    )

    if townships is not None:
        townships.boundary.plot(
            ax=ax,
            color="black",
            linewidth=0.25,
            alpha=0.5,
        )

    plt.colorbar(sc, ax=ax, label=var)

    ax.set_title(
        f"HRRR {var}\n{DATE_TO_PLOT}, lead hour {LEAD_HOUR}",
        pad=12,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xlim(-97.2, -88.8)
    ax.set_ylim(39.8, 44.2)

    plt.tight_layout()
    plt.show()


for var in VARIABLES_TO_MAP:
    plot_hrrr_square_map(df_day, var, townships=townships)


# ============================================================
# 7. MEMORY-SAFE DAILY MEAN TIME SERIES, NO SAVING
# ============================================================

def daily_mean_from_row_groups(path, variable, lead_hour):
    """
    Read one variable row-group by row-group instead of loading the full year.
    This is much safer for large yearly HRRR parquet files.
    """
    available_cols = pq.read_schema(path).names

    needed = ["valid_date", "lead_hour", variable]
    missing = [c for c in needed if c not in available_cols]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    pf = pq.ParquetFile(path)

    pieces = []

    print(f"\nBuilding daily mean for {variable}")
    print(f"Number of row groups: {pf.num_row_groups}")

    for i in range(pf.num_row_groups):
        small = pf.read_row_group(i, columns=needed).to_pandas()

        small["lead_hour"] = pd.to_numeric(small["lead_hour"], errors="coerce")
        small = small[small["lead_hour"] == lead_hour].copy()

        if small.empty:
            del small
            continue

        small[variable] = pd.to_numeric(small[variable], errors="coerce")
        small = small.dropna(subset=[variable])

        if small.empty:
            del small
            continue

        grouped = (
            small.groupby("valid_date")[variable]
            .agg(["sum", "count"])
            .reset_index()
        )

        pieces.append(grouped)

        del small, grouped
        gc.collect()

    if not pieces:
        raise ValueError("No data found for this variable and lead hour.")

    out = pd.concat(pieces, ignore_index=True)

    out = (
        out.groupby("valid_date", as_index=False)[["sum", "count"]]
        .sum()
    )

    out["mean"] = out["sum"] / out["count"]
    out["valid_date"] = pd.to_datetime(out["valid_date"])

    return out.sort_values("valid_date")


ts = daily_mean_from_row_groups(
    path=HRRR_FILE,
    variable=VARIABLE_FOR_TIME_SERIES,
    lead_hour=LEAD_HOUR,
)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    ts["valid_date"],
    ts["mean"],
    linewidth=2,
)

ax.set_title(
    f"HRRR Iowa-wide daily mean {VARIABLE_FOR_TIME_SERIES}\n"
    f"{YEAR}, lead hour {LEAD_HOUR}",
    pad=12,
)
ax.set_xlabel("Date")
ax.set_ylabel(VARIABLE_FOR_TIME_SERIES)
ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()

# from pathlib import Path

# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import pyarrow.parquet as pq


# # ============================================================
# # Settings
# # ============================================================

# YEAR = 2020
# MODEL = "hrrr"

# # HRRR available lead hours in your setup:
# # 0  = analysis / nowcast
# # 24 = one-day forecast
# LEAD_HOURS_TO_PLOT = [0, 24]

# VARIABLES_TO_PLOT = [
#     "temperature_c",
#     "wind_speed_10m_mps",
#     "precip_accum_mm",
#     "moisture_availability_percent",
# ]

# # Bigger square markers make centroids look more like pixels.
# # Increase this if the grid still looks too sparse.
# PIXEL_MARKER_SIZE = 14

# BATCH_SIZE = 250_000
# DPI = 300

# PROJECT_ROOT = Path.cwd()

# YEARLY_DIR = (
#     PROJECT_ROOT
#     / "src"
#     / "data"
#     / "processed"
#     / "archived_weather"
#     / "yearly"
# )

# FIGURE_DIR = (
#     PROJECT_ROOT
#     / "src"
#     / "data"
#     / "processed"
#     / "archived_weather"
#     / "figures"
#     / "hrrr_pixel_maps"
#     / str(YEAR)
# )

# PIXEL_FILE = YEARLY_DIR / f"{MODEL}_weather_iowa_pixels_{YEAR}.parquet"

# TOWNSHIP_FILE = (
#     PROJECT_ROOT
#     / "src"
#     / "data"
#     / "raw"
#     / "townships"
#     / "civil_townships_a_ia.shp"
# )


# # ============================================================
# # Helpers
# # ============================================================

# def safe_filename(text: str) -> str:
#     return (
#         str(text)
#         .replace("/", "_")
#         .replace("\\", "_")
#         .replace(" ", "_")
#         .replace("|", "_")
#         .replace(":", "_")
#     )


# def load_townships() -> gpd.GeoDataFrame:
#     townships = gpd.read_file(TOWNSHIP_FILE)

#     if townships.crs is None:
#         raise ValueError("Township shapefile has no CRS. Please define its CRS first.")

#     townships = townships.to_crs("EPSG:4326")

#     return townships


# def load_iowa_boundary(townships: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     return gpd.GeoDataFrame(
#         geometry=[townships.union_all()],
#         crs=townships.crs,
#     )


# def get_parquet_columns(path: Path) -> list[str]:
#     pf = pq.ParquetFile(path)
#     return pf.schema_arrow.names


# def iter_pixel_batches(path: Path, read_cols: list[str]):
#     pf = pq.ParquetFile(path)

#     for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=read_cols):
#         df = batch.to_pandas()

#         if "model" in df.columns:
#             df = df[df["model"].astype(str) == MODEL].copy()

#         if "lead_hour" in df.columns:
#             df["lead_hour"] = pd.to_numeric(df["lead_hour"], errors="coerce")
#             df = df[df["lead_hour"].isin(LEAD_HOURS_TO_PLOT)].copy()

#         if df.empty:
#             continue

#         yield df


# def aggregate_yearly_grid_mean(
#     year: int,
#     variables: list[str],
# ) -> pd.DataFrame:
#     """
#     Aggregate the yearly HRRR data to one value per:
#         grid_id, lead_hour, variable

#     This reads the parquet in batches, so it is safer for memory.
#     """
#     path = YEARLY_DIR / f"{MODEL}_weather_iowa_pixels_{year}.parquet"

#     if not path.exists():
#         raise FileNotFoundError(f"Missing pixel file:\n{path}")

#     available_cols = get_parquet_columns(path)

#     id_cols = [
#         "model",
#         "product",
#         "lead_hour",
#         "forecast_type",
#         "grid_id",
#         "latitude",
#         "longitude",
#     ]

#     variables = [v for v in variables if v in available_cols]

#     if not variables:
#         raise ValueError("None of the requested variables exist in the parquet file.")

#     read_cols = [c for c in id_cols + variables if c in available_cols]

#     required = ["model", "lead_hour", "grid_id", "latitude", "longitude"]
#     missing = [c for c in required if c not in read_cols]

#     if missing:
#         raise ValueError(f"Missing required columns in parquet file:\n{missing}")

#     batch_summaries = []

#     for df in iter_pixel_batches(path, read_cols):
#         group_cols = [c for c in id_cols if c in df.columns]
#         value_cols = [v for v in variables if v in df.columns]

#         for col in value_cols:
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#         sum_df = (
#             df.groupby(group_cols, dropna=False)[value_cols]
#             .sum(min_count=1)
#             .add_suffix("__sum")
#             .reset_index()
#         )

#         count_df = (
#             df.groupby(group_cols, dropna=False)[value_cols]
#             .count()
#             .add_suffix("__count")
#             .reset_index()
#         )

#         batch_summary = sum_df.merge(count_df, on=group_cols, how="outer")
#         batch_summaries.append(batch_summary)

#     if not batch_summaries:
#         return pd.DataFrame()

#     all_batches = pd.concat(batch_summaries, ignore_index=True, sort=False)

#     group_cols = [
#         c for c in id_cols
#         if c in all_batches.columns
#     ]

#     sum_count_cols = [
#         c for c in all_batches.columns
#         if c not in group_cols
#     ]

#     combined = (
#         all_batches.groupby(group_cols, dropna=False)[sum_count_cols]
#         .sum(min_count=1)
#         .reset_index()
#     )

#     for var in variables:
#         sum_col = f"{var}__sum"
#         count_col = f"{var}__count"

#         if sum_col in combined.columns and count_col in combined.columns:
#             combined[var] = combined[sum_col] / combined[count_col].replace(0, np.nan)

#     keep_cols = group_cols + variables
#     keep_cols = [c for c in keep_cols if c in combined.columns]

#     return combined[keep_cols].reset_index(drop=True)


# def make_centroid_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
#     gdf = gpd.GeoDataFrame(
#         df,
#         geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
#         crs="EPSG:4326",
#     )

#     return gdf


# def plot_variable_map(
#     gdf: gpd.GeoDataFrame,
#     townships: gpd.GeoDataFrame,
#     iowa_boundary: gpd.GeoDataFrame,
#     variable: str,
#     lead_hour: int,
#     outdir: Path,
# ) -> None:
#     sub = gdf[gdf["lead_hour"] == lead_hour].copy()

#     if sub.empty:
#         print(f"No data for lead hour {lead_hour}. Skipping {variable}.")
#         return

#     if variable not in sub.columns or sub[variable].isna().all():
#         print(f"No values for {variable}. Skipping.")
#         return

#     forecast_type = "unknown"
#     if "forecast_type" in sub.columns:
#         vals = sub["forecast_type"].dropna().astype(str).unique()
#         if len(vals) > 0:
#             forecast_type = vals[0]

#     fig, ax = plt.subplots(figsize=(10, 8))

#     # Plot township lines first, very light.
#     townships.boundary.plot(
#         ax=ax,
#         color="lightgray",
#         linewidth=0.25,
#         alpha=0.6,
#         zorder=1,
#     )

#     # Plot HRRR centroids as large square pixels.
#     sub.plot(
#         ax=ax,
#         column=variable,
#         cmap="viridis",
#         marker="s",
#         markersize=PIXEL_MARKER_SIZE,
#         alpha=0.95,
#         legend=True,
#         zorder=2,
#     )

#     # Plot Iowa boundary on top.
#     iowa_boundary.boundary.plot(
#         ax=ax,
#         color="#0066b3",
#         linewidth=1.2,
#         zorder=3,
#     )

#     ax.set_title(
#         f"{variable} | HRRR yearly mean | {forecast_type} | F{lead_hour} | {YEAR}",
#         fontsize=13,
#         pad=12,
#     )

#     ax.set_xlabel("Longitude")
#     ax.set_ylabel("Latitude")

#     minx, miny, maxx, maxy = iowa_boundary.total_bounds
#     ax.set_xlim(minx - 0.15, maxx + 0.15)
#     ax.set_ylim(miny - 0.15, maxy + 0.15)

#     ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

#     plt.tight_layout()

#     outdir.mkdir(parents=True, exist_ok=True)

#     out_path = (
#         outdir
#         / f"hrrr_centroid_pixels_{safe_filename(variable)}_F{lead_hour}_{YEAR}.png"
#     )

#     plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
#     plt.show()
#     plt.close()

#     print(f"Saved: {out_path}")


# # ============================================================
# # Main
# # ============================================================

# def main() -> None:
#     print("=" * 70)
#     print(f"Plotting HRRR centroid-pixel maps for {YEAR}")
#     print(f"Pixel file: {PIXEL_FILE}")
#     print("=" * 70)

#     townships = load_townships()
#     iowa_boundary = load_iowa_boundary(townships)

#     print("Aggregating yearly grid means...")
#     yearly_grid = aggregate_yearly_grid_mean(
#         year=YEAR,
#         variables=VARIABLES_TO_PLOT,
#     )

#     print("Yearly grid shape:", yearly_grid.shape)
#     print("Columns:", yearly_grid.columns.tolist())

#     gdf = make_centroid_gdf(yearly_grid)

#     # Clip centroids to Iowa boundary.
#     gdf = gpd.clip(gdf, iowa_boundary)

#     print("Clipped grid shape:", gdf.shape)

#     for variable in VARIABLES_TO_PLOT:
#         if variable not in gdf.columns:
#             print(f"Skipping missing variable: {variable}")
#             continue

#         for lead_hour in LEAD_HOURS_TO_PLOT:
#             plot_variable_map(
#                 gdf=gdf,
#                 townships=townships,
#                 iowa_boundary=iowa_boundary,
#                 variable=variable,
#                 lead_hour=lead_hour,
#                 outdir=FIGURE_DIR,
#             )

#     print("=" * 70)
#     print("Done.")
#     print(f"Figures saved in:\n{FIGURE_DIR}")
#     print("=" * 70)


# if __name__ == "__main__":
#     main()