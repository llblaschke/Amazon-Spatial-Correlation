#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np

from model_params_functions import *
from data_analysis_functions import *

path_simulation = "data/simulations.nc"

# get equilibria
[Vstar, Pstar] = get_equilibria(B)
Vstar = xr.DataArray(Vstar, dims=("time", "cell"))
Pstar = xr.DataArray(Pstar, dims=("time", "cell"))

V0 = Vstar.isel(time=0).values

# SIMULATION
data = model_nruns_with_white_noise(nruns, B, V0)

# i where first time in any cell P drops below Pcrit (take a few (10) steps before)
i_max = np.where((data.Precipitation - Pcrit).min(("cell", "run")) < 0)[0][0] - 10

# EWS
[ac1, var, spC] = calculate_EWS(data.Vegetation - Vstar, nruns=nruns, nT=i_max)
data["AC1"]      = xr.DataArray(ac1, dims=("run", "time", "cell")).compute()
data["Variance"] = xr.DataArray(var, dims=("run", "time", "cell")).compute()
data["SpCorr"]   = xr.DataArray(spC, dims=("run", "time", "cell")).compute()
data["P*"] = xr.DataArray(Pstar, dims = ("time", "cell"))
data["V*"] = xr.DataArray(Vstar, dims = ("time", "cell"))
data = data.assign_coords(B = ("time", B))

# save
data.to_netcdf(path_simulation)
