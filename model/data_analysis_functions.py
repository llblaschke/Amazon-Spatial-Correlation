import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, percentileofscore

from model_params_functions import *


# minimum length of CSD indicator time series to calculate first trend
n_first_trend   = 10



# Definition of "neighborhood": 1 cell to east and 1 cell to west
distance_matrix = np.zeros((ncells, ncells))
distance_matrix += np.diag(np.ones(ncells-1), k= 1)
distance_matrix += np.diag(np.ones(ncells-1), k=-1)
distance_matrix = distance_matrix == 1


########################################################
# Sliding Windows indicators

def sliding_window_indicators(d):
    # inds = np.zeros((3, nT, d.shape[-1]))
    inds = np.zeros((3, nT, d.shape[-1]))
    inds[:,:,:] = np.nan
    T = d.shape[0]
    for itime in range(sliding_window_size, T):
        dtmp = d[itime-sliding_window_size:itime, :]
        inds[0, itime-1, :] = pd.DataFrame(dtmp[1:, :]).corrwith(pd.DataFrame(dtmp[:-1, :])).to_numpy() # AC1
        inds[1, itime-1, :] = np.var(dtmp, axis=0)                                                     # Var
        inds[2, itime-1, :] = pd.DataFrame(dtmp).corr().where(distance_matrix).mean().to_xarray()       # SpCorr

    return inds

# Apply Sliding Windows indicators
def calculate_indicators(d, time_dim='time'):
    d.load()
    s =  xr.apply_ufunc(
        sliding_window_indicators, 
        d,
        input_core_dims=[[time_dim, "cell"]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[['indicator', "time", "cell"]],
        vectorize=True                 # vectorize to apply it on single lat-lon-combinations
        )
    return s

# Apply Sliding Windows indicators on several runs
def calculate_EWS(data, nruns, nT=nT):

    ac1 = np.zeros((nruns, nT, ncells))
    var = np.zeros((nruns, nT, ncells))
    spC = np.zeros((nruns, nT, ncells))
    ac1[:,:,:] = np.nan
    var[:,:,:] = np.nan
    spC[:,:,:] = np.nan
    for nrun in range(nruns):    
        ac1tmp, vartmp, spCtmp = sliding_window_indicators(data.isel(run=nrun))
        ac1[nrun, :, :] = ac1tmp
        var[nrun, :, :] = vartmp
        spC[nrun, :, :] = spCtmp
    
    return [ac1, var, spC]


########################################################
# Signficance analysis
########################################################
# Fourier surrogates
def fourrier_surrogates(ts, ns):
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    # random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0])) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts


# Trends for all time perios
def all_trends(d):
    # print(d.shape)
    ts = np.zeros(nT)
    ts[:] = np.nan
    T = np.max(np.where(np.invert(np.isnan(d)))[0])
    for itime in range(sliding_window_size+n_first_trend, T+1):
        dtmp = d[sliding_window_size-1:itime]
        ts[itime] = linregress(range(itime-sliding_window_size+1), dtmp)[0]

    return ts

# apply all trends
def calculate_trends_all_t(d, time_dim='time'):
    # d.load()
    s =  xr.apply_ufunc(
        all_trends, 
        d,
        input_core_dims=[[time_dim]],   # dimension not to "broadcast" (=loop) over
        output_core_dims=[[time_dim]],
        vectorize=True                 # vectorize to apply it on single lat-lon-combinations
        )
    return s

# p-values
def p_value(d, s):
    return (1 - percentileofscore(s,d)/100.)

def calculate_p_values(d, s, surrogates_dim="surrogate"):
    s.load()
    s =  xr.apply_ufunc(
        p_value, 
        d,
        s, 
        input_core_dims=[[], [surrogates_dim]],   # dimension not to "broadcast" (=loop) over
        vectorize=True                 # vectorize to apply it on single lat-lon-combinations
        )
    return s



