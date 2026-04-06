# !/usr/bin/env python3
import h5py
import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset



file_path = "SMAP_L3_SM_P_E_20200711_R19240_001_subsetted.nc4"

if os.path.isfile(file_path):
    print(f"'{file_path}' exists.")
else:
    print(f"'{file_path}' doesn't exist.")



nc = Dataset(file_path, "r")

print(nc.groups.keys())

with h5py.File(file_path, "r") as f: # navigating the file using h5py
    def navigate_file(name, obj):
        print(name)
    f.visititems(navigate_file)

ds_am = xr.open_dataset(file_path, group="Soil_Moisture_Retrieval_Data_AM", engine="netcdf4")
print(ds_am)