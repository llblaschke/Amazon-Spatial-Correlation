
import numpy as np
import xarray as xr

from model_params_functions import *
from data_analysis_functions import *

path_data_orig  = 'data/simulations.nc'
path_pvalues    = 'data/surrogate_pvalues_1000.nc'

data_orig = xr.open_dataset(path_data_orig)



nSurrogates = 1000


B = np.array([1000.])
# get equilibria
[Vstar, Pstar] = get_equilibria(B)
Vstar = np.ones((nT, ncells)) * Vstar
Pstar = np.ones((nT, ncells)) * Pstar
Vstar = xr.DataArray(Vstar, dims=("time", "cell"))
Pstar = xr.DataArray(Pstar, dims=("time", "cell"))

B = np.ones(nT)*1000.

V0 = Vstar.isel(time=0).values

# SIMULATION
surrogates = model_nruns_with_white_noise(nSurrogates, B, V0).assign_coords({"time": range(nT)})
print("nullmodel runs created")

# i where first time in any cell P drops below Pcrit
i_max = int(data_orig.time.where(data_orig.Precipitation <= Pcrit).min()) 

# EWS
[ac1, var, spC] = calculate_EWS(surrogates.Vegetation - Vstar, nSurrogates=nSurrogates, nT=i_max)
surrogates["AC1"]      = xr.DataArray(ac1, dims=("run", "time", "cell")).compute()
surrogates["Variance"] = xr.DataArray(var, dims=("run", "time", "cell")).compute()
surrogates["SpCorr"]   = xr.DataArray(spC, dims=("run", "time", "cell")).compute()
surrogates["P*"] = xr.DataArray(Pstar, dims = ("time", "cell"))
surrogates["V*"] = xr.DataArray(Vstar, dims = ("time", "cell"))
surrogates = surrogates.assign_coords({"time": range(i_max)})
surrogates = surrogates.assign_coords({"B": ("time", data_orig.isel(time=slice(None, i_max).B))})
print("ews calculated")

surrogates = calculate_trends_all_t(surrogates.to_array("indicator").sel(indicator=["AC1", "Variance", "SpCorr"]))
print("all trends calculated")



# SIGNIFICANCE 
# calculate trends for all time steps of original simulation data
data_orig = data_orig.assign_coords({"time": range(len(data_orig.time)), "cell": range(len(data_orig.cell))})
data = data_orig.to_array("indicator").sel(indicator=["AC1", "Variance", "SpCorr"])
data = calculate_trends_all_t(data).compute()


time_nonan = surrogates.dropna("time", "all").time.values
ps = calculate_p_values(data.sel(time=time_nonan), surrogates.sel(time=time_nonan))
ps = ps.to_dataset("indicator").rename({"AC1": "pAC1", "Variance": "pVariance", "SpCorr": "pSpCorr"}) 

data_orig = data_orig.merge(ps)
data_orig.to_netcdf(path_pvalues, 'w')
