
from pathlib import Path
import os
import sys
import pandas as pd
import geopandas as gpd
# import pyogrio  # doens't work (figure out why) used fiona instead
import matplotlib.pyplot as plt
from adjustText import adjust_text


# print(gpd.__version__)
# print(pyogrio.__version__)
# twnshp_map = gpd.read_file(RAW_TOWNSHIPS / "civil_townships_a_ia.shp")
# twnshp_map = gpd.read_file(
#     RAW_TOWNSHIPS / "civil_townships_a_ia.shp",
#     engine="fiona")


cwd = os.getcwd()
print(cwd) 

twnshp = "/home/armaghan/projects/SM_forecasting/src/data/raw/townships/civil_townships_a_ia.shp"
iem = "/home/armaghan/projects/SM_forecasting/src/data/processed/isu_stations/stations_full.csv"
smap_lattice = "/home/armaghan/projects/SM_forecasting/src/data/processed/smap_gap_filling/support/smap_lattice/smap_lattice_iowa.parquet"
# 2. Extract the CSV directory and build the PDF output path
out_path = os.path.dirname(os.path.abspath(iem))
output_pdf_path = os.path.join(out_path, "spatial_map.pdf")
output_png_path = os.path.join(out_path, "spatial_map.png")


# Load data
gdf_shapes = gpd.read_file(twnshp, engine="fiona")
df_csv = pd.read_csv(iem)
gdf_points = gpd.GeoDataFrame(
    df_csv, geometry=gpd.points_from_xy(df_csv['lon'], df_csv['lat']), crs="EPSG:4326"
).to_crs(gdf_shapes.crs)

# Dissolve townships into a single outer state boundary
iowa_poly = gdf_shapes.dissolve()

# Boundary line, for drawing the outline 
iowa_outline = iowa_poly.boundary

# SMAP pixel lattice — read straight from parquet, no .rds/R interop needed
gdf_pixels = gpd.read_parquet(smap_lattice)
if gdf_pixels.crs != gdf_shapes.crs:
    gdf_pixels = gdf_pixels.to_crs(gdf_shapes.crs)

gdf_pixels_clipped = gpd.clip(gdf_pixels, iowa_poly)


fig, ax = plt.subplots(figsize=(10, 10))

gdf_pixels_clipped.plot(ax=ax, facecolor='none', edgecolor='#1a5470', linewidth=0.6, alpha=0.75, zorder=1)
iowa_outline.plot(ax=ax, color='#2d0954', linewidth=1.4, zorder=2)
gdf_points.plot(ax=ax, color='#29af7f', marker='o', markersize=45,
                edgecolor='white', linewidth=0.6, zorder=3, label='IEM stations')

# ax.set_title("IEM Stations over SMAP Pixel Grid — Iowa", fontsize=14, fontweight='bold', pad=15)
ax.set_axis_off()
ax.legend(loc='lower center', frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()
