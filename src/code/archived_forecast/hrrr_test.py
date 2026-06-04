# Herbie package needs python 11 or higher
# activate py312 for this script

# Iowa surrounding (W,S,E,N) -> (-97, 40,-89, 44)

from herbie import Herbie

# ------------------------------------------------------------
# 1. Define one HRRR forecast file
# ------------------------------------------------------------
# This means:
# Forecast initialized at 2020-01-01 00:00 UTC
# Forecast lead time = 24 hours
# So the forecast is valid around 2020-01-02 00:00 UTC

H = Herbie(
    "2020-01-01 00:00",
    model="hrrr",
    product="sfc",
    fxx=24,
)

# ------------------------------------------------------------
# 2. Read only one variable from the HRRR file
# ------------------------------------------------------------
# TMP = temperature
# 2 m above ground = near-surface air temperature
# This avoids downloading/reading all variables.

ds = H.xarray(":TMP:2 m above ground")

print("Full HRRR dataset:")
print(ds)

# ------------------------------------------------------------
# 3. Define your Iowa-area bounding box
# ------------------------------------------------------------
# Your desired box:
# W, S, E, N = (-97, 40, -89, 44)
#
# west  = minimum longitude
# south = minimum latitude
# east  = maximum longitude
# north = maximum latitude

west = -97
south = 40
east = -89
north = 44

# ------------------------------------------------------------
# 4. Convert longitude convention
# ------------------------------------------------------------
# HRRR uses longitudes from 0 to 360.
# Your bounding box uses longitudes from -180 to 180.
# % here in python for negative dividend has different convention
# Example:
# -97 degrees = 263 degrees in 0-360 notation
# -89 degrees = 271 degrees in 0-360 notation

west_360 = west % 360
east_360 = east % 360

print("Converted longitude bounds:")
print("west_360 =", west_360)
print("east_360 =", east_360)

# ------------------------------------------------------------
# 5. Create a spatial mask for the Iowa-area box
# ------------------------------------------------------------
# ds["latitude"] and ds["longitude"] already exist inside HRRR.
# We are selecting only grid cells whose lat/lon fall inside the box.

bbox_mask = (
    (ds["latitude"] >= south)
    & (ds["latitude"] <= north)
    & (ds["longitude"] >= west_360)
    & (ds["longitude"] <= east_360)
)

# ------------------------------------------------------------
# 6. Apply the mask
# ------------------------------------------------------------
# drop=True removes grid cells outside the box.

ds_iowa = ds.where(bbox_mask, drop=True)

print("Subset HRRR dataset over Iowa-area box:")
print(ds_iowa)

# ------------------------------------------------------------
# 7. Check the coordinate range after subsetting
# ------------------------------------------------------------
# This helps confirm that the subset is actually around Iowa.

print("Subset latitude range:")
print(float(ds_iowa["latitude"].min()), "to", float(ds_iowa["latitude"].max()))

print("Subset longitude range in 0-360 system:")
print(float(ds_iowa["longitude"].min()), "to", float(ds_iowa["longitude"].max()))

print("Subset longitude range in -180 to 180 system:")
print(
    float(ds_iowa["longitude"].min()) - 360,
    "to",
    float(ds_iowa["longitude"].max()) - 360,
)

# ------------------------------------------------------------
# 8. Check temperature values
# ------------------------------------------------------------
# HRRR temperature is in Kelvin.
# Convert it to Celsius.

temp_k_min = float(ds_iowa["t2m"].min())
temp_k_max = float(ds_iowa["t2m"].max())

temp_c_min = temp_k_min - 273.15
temp_c_max = temp_k_max - 273.15

print("2-meter temperature range over Iowa box:")
print("Kelvin:", temp_k_min, "to", temp_k_max)
print("Celsius:", temp_c_min, "to", temp_c_max)

# ------------------------------------------------------------
# 9. Convert the Iowa subset to a table
# ------------------------------------------------------------

df_iowa = ds_iowa[["t2m"]].to_dataframe().reset_index()

# Remove rows outside the exact bounding box
df_iowa = df_iowa.dropna(subset=["t2m"]).copy()

# Convert longitude from 0-360 to -180 to 180
df_iowa["longitude_180"] = ((df_iowa["longitude"] + 180) % 360) - 180

# Convert temperature from Kelvin to Celsius
df_iowa["t2m_c"] = df_iowa["t2m"] - 273.15

# Add metadata
df_iowa["init_time"] = "2020-01-01 00:00"
df_iowa["valid_time"] = "2020-01-02 00:00"
df_iowa["lead_hour"] = 24
df_iowa["forecast_type"] = "forecast_1_day"

# Keep useful columns
df_iowa = df_iowa[
    [
        "init_time",
        "valid_time",
        "lead_hour",
        "forecast_type",
        "latitude",
        "longitude_180",
        "t2m",
        "t2m_c",
    ]
]

print(df_iowa.head())
print("Number of valid grid cells:", len(df_iowa))
print("Latitude range:", df_iowa["latitude"].min(), "to", df_iowa["latitude"].max())
print("Longitude range:", df_iowa["longitude_180"].min(), "to", df_iowa["longitude_180"].max())