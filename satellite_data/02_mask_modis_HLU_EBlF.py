import numpy as np
import xarray as xr

# min EBlF percentage
min_EBlF = 80 # minimal amount of Evergreen Broadleaf Forest
max_HLU  =  0 # maximum amount of Human Land Use

path_modis = 'path_to_modis/MCD12C1.*.hdf'
path_modis_HLU    = 'data/modis_HLU.nc'
path_amazon_basin = 'data/amazon_basin_mask.nc'
path_hansen       = 'data/hansen_deforestation.nc'
path_cells_to_use = 'data/cells_to_include.nc'


years_HLU = slice(np.datetime64("2001-01"), np.datetime64("2020-12"))


# Land Cover Types in the data (right order)
lct_values = [
    "Water bodies",
    "Evergreen Needleleaf Forests",
    "Evergreen Broadleaf Forests",
    "Deciduous Needleleaf Forests",
    "Deciduous Broadleaf Forests",
    "Mixed Forests",
    "Closed Shrublands",
    "Open Shrublands",
    "Woody Savannas",
    "Savannas",
    "Grasslands",
    "Permanent Wetlands",
    "Croplands",
    "Urban and Built-up Lands",
    "Cropland/Natural Vegetation Mosaics",
    "Permanent Snow and Ice",
    "Barren"
]

HLU  = ['Croplands', 'Urban and Built-up Lands', 'Cropland/Natural Vegetation Mosaics']
EBlF = ['Evergreen Broadleaf Forests']

def preprocess_add_date(ds):
    mydate = np.datetime64(ds.attrs['RANGEBEGINNINGDATE'])
    return (ds
            .expand_dims(dict(time=[mydate]))
            .rename(x = "lon", y = "lat")
            .drop('spatial_ref')
            .assign_coords(dict(band = lct_values))
        )

# Function to load and shape the data
ds = xr.open_mfdataset(
        path_modis, 
        engine = 'rasterio',
        chunks='auto',
        preprocess=preprocess_add_date).Land_Cover_Type_1_Percent
print("ds opened")
ds.attrs['units'] = ds.attrs['units'][0]

# Coarsening to 0.25 grid (like AMSR-E and AMSR2) by taking the mean of the 5*5 cells
ds = ds.sel(time=years_HLU, band=HLU+EBlF).reindex(lat=list(reversed(ds.lat)))
mask = xr.open_dataarray(path_amazon_basin)
grid_res = 0.25
coarse_f = 5
lon_slice = slice(mask.lon.min() - grid_res/2, mask.lon.max() + grid_res/2)
lat_slice = slice(mask.lat.min() - grid_res/2, mask.lat.max() + grid_res/2)
ds = ds.sel(lon=lon_slice, lat=lat_slice)

ds = ds.coarsen(lon=coarse_f, lat=coarse_f, boundary="exact").mean()
if np.allclose(mask.lon.values, ds.lon.values): ds["lon"] = mask.lon
if np.allclose(mask.lat.values, ds.lat.values): ds["lat"] = mask.lat
ds = ds.where(mask)

# Selecting HLU and EBlF
hlu  = ds.sel(band=HLU).sum(dim="band").max("time")  <= max_HLU  # only keep cells with maximum that amount of HLU at any point in time
eblf = ds.sel(band=EBlF).sum(dim="band").min("time") >= min_EBlF # only keep cells that have at least min_eblf EBlF at any point in time
xr.Dataset({"EBlF": eblf, "HLU": hlu}).to_netcdf(path_modis_HLU)

hansen = xr.open_dataarray(path_hansen)
(hansen & eblf & hlu).where(mask).to_netcdf(path_cells_to_use)
