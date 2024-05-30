import numpy as np
import xarray as xr
import glob

max_Defo = 1 # maximum deforestation according to Hansen
last_year_HLU = 2020

lats = ["10N", "00N", "10S", "20S"]
lons = ["080W", "070W", "060W", "050W", "040W"]

path_amazon_basin = 'data/amazon_basin_mask.nc'
path_hansen       = "path_to_hansen/"
path_hansen_save  = 'data/hansen_deforestation.nc'
path_tmp          = "data/hansen_tmp/"

for p in glob(path_hansen+'*.tif'):
    f = p.split("/")[-1]
    lat, lon = f.split("_")[-2:]
    lon = lon.split(".")[0]
    if lon in lons and lat in lats:
        h = xr.open_dataarray(p)
        h = h.rename("forestLoss").isel(band=0, drop=True).rename({"x":"lon", "y":"lat"})
        h = h.isin(range(1,last_year_HLU+1-2000)) * 100
        # temporarily save
        h.to_netcdf(path_tmp + f.replace("_lossyear", "").replace("tif", "nc"))

h = xr.open_mfdataset(path_tmp + "*.nc")

mask = xr.open_dataarray(path_amazon_basin)
lats = mask.lat.values
lons = mask.lon.values
grid_res = 0.25

lon_slice = slice(mask.lon.min() - grid_res/2, mask.lon.max() + grid_res/2)
lat_slice = slice(mask.lat.min() - grid_res/2, mask.lat.max() + grid_res/2)

if len(h.sel(lat=lat_slice).lat.values) == 0: h = h.reindex(lat=list(reversed(h.lat)))
h = h.sel(lon=lon_slice, lat=lat_slice)

def take_closest(fineGrid, coarseList): return min(coarseList, key=lambda x:abs(x-fineGrid))

lat_old = h.lat.values
lat_new = [take_closest(l,lats) for l in lat_old]

lon_old = h.lon.values
lon_new = [take_closest(l,lons) for l in lon_old]

h["lat"] = lat_new
h["lon"] = lon_new

h = h.groupby("lat").mean()
h = h.groupby("lon").mean()
h = h.where(mask) <= max_Defo
h.to_netcdf(path_hansen_save)
