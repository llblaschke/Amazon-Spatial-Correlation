import numpy as np
import xarray as xr

nruns             = 1000
ncells            = 10
sigma_precip      = 20.
startValB         = 1010
endValB           = 990
Pcrit             = 1079.41 # according to wolfram alpha

# number of surrogates
ns = 1000

# general parameters
tau       = 100
deltaT    = 1.
dt        = deltaT/tau
nT        = 1000
B         = np.linspace(startValB, endValB, nT)

neqRuns      =  1000
neqRunsSetup = 10000


# s_i (equilibrium precipitation in the absence of any vegetation: P_d = local_sensitivity * B)
minSensitivity    = 0.2
local_sensitivity = np.linspace(.1, 1, ncells)
local_sensitivity[local_sensitivity < minSensitivity] = minSensitivity

# Interaction matrix Precipiptation
interactionScale     = 600

# Spatial correlation parameters
sliding_window_size = 100

# Precipitation-Vegetation-Interaction parameters for Equlibrium Vegetation Cover V*
# minimum precipitation P1 and saturation precipitation P2 as functions of
alpha = 0.0011
beta  = 28 * 10 # 10-times the value in Sebastians paper
phi = 2.45      # phi = gamma * delta

# minimum precipitation P1 and saturation precipitation P2 as functions of
expphi = np.exp(phi)
P1 = beta * np.exp(phi / 2.)
P2 = beta * np.exp(phi / 2.) + expphi / np.sqrt(0.03 * alpha)





########################################################
## FUNCTIONS
########################################################
# Coupling MATRIX
coupling_matrix = np.diag(np.ones(ncells), 0) 
for i in range(ncells-1): coupling_matrix += np.diag(np.ones(ncells-i-1), k = i+1) * (1/float(i+2)) # last term is weighting by 1/distance)
coupling_matrix = coupling_matrix * interactionScale
coupling_matrix_east = coupling_matrix - np.diag(np.ones(ncells),0)*interactionScale # only interaction to the east, i.e. subtract "self-interaction"


########################################################
# Vegetation 
def equilibrium_vegetation(P):
    Vnew = 1.03 - 1.03 / (1 + alpha * ((P-P1)/expphi)**2)
    Vnew[P<P1] = 0
    Vnew[P>P2] = 1
    return Vnew

def equilibrium_vegetation_scalar(P):
    if P <= P1: return 0
    if P >= P2: return 1
    return 1.03 - 1.03 / (1 + alpha * ((P-P1)/expphi)**2)

# Distance of V from V*
def V_diff(B, Pstar, Vstar, sensitivity, index):
    return (Pstar - sensitivity*B - np.matmul(coupling_matrix_east, Vstar)[index])/interactionScale - equilibrium_vegetation_scalar(Pstar)

########################################################
# Precipitation
def total_equilibrium_precipitation(V, B_of_t):
    return local_sensitivity * B_of_t + np.matmul(coupling_matrix, V)

def total_precipitation(V, B_of_t):
    return total_equilibrium_precipitation(V, B_of_t) + sigma_precip * np.random.randn(ncells)
  

########################################################
# Equilibria
# calculate equilibria for all cells for one specific B
def equilibrium_vegetation_and_precipitation(B):
    Vstartmp = np.ones(ncells)
    Pstartmp_vector = np.ones(ncells)

    for index in range(ncells-1,-1,-1):
        sensitivity = local_sensitivity[index]
        Pstartmp = total_equilibrium_precipitation(Vstartmp, B)[index]
        Verr = 1
        Perr = 1
        while  np.abs(Verr) > 1e-14 or np.abs(Perr) > 1e-3:
            Verr = V_diff(B, Pstartmp, Vstartmp, sensitivity, index)
            Vstartmp[index] = Vstartmp[index] - Verr
            Perr = Pstartmp
            Pstartmp = total_equilibrium_precipitation(Vstartmp, B)[index]
            Perr = Perr - Pstartmp 
        Pstartmp_vector[index] = Pstartmp
    return [Vstartmp, Pstartmp_vector]

# equilibria for all B-values
def get_equilibria(Bs):
    Vstar = np.ones((len(Bs), ncells))
    Pstar = np.ones((len(Bs), ncells))
    for i in range(len(Bs)):
        [Vstar[i,:], Pstar[i,:]] = equilibrium_vegetation_and_precipitation(Bs[i])
    return [Vstar, Pstar]


########################################################
# EULER-MARUYAMA EQUILIBRIUM SOLVE
def euler_maruyama_solve_white_noise_equilibrium_run(V0, Bs, nT=nT):#, tau, dt, nT, B, sigma, ncells):
    Vs = np.zeros((nT, ncells))
    Ps = np.zeros((nT, ncells))
    # start from equilibrium
    Vtmp = V0 + 0.01 * np.random.randn(ncells)
    Btmp = Bs[0]
    for i in range(neqRunsSetup-neqRuns):
        Ptmp = total_precipitation(Vtmp, Btmp)
        Vtmp = Vtmp + (equilibrium_vegetation(Ptmp) - Vtmp) * dt

    for t in range(nT):
        Btmp = Bs[t]
        for i in range(neqRuns):
            Ptmp = total_precipitation(Vtmp, Btmp)
            Vtmp = Vtmp + (equilibrium_vegetation(Ptmp) - Vtmp) * dt
        Vs[t, :] = Vtmp
        Ps[t, :] = Ptmp

    Ps  = xr.DataArray(Ps, dims = ("time", "cell"))
    Vs  = xr.DataArray(Vs, dims = ("time", "cell"))
    return xr.Dataset({"Precipitation": Ps, "Vegetation": Vs}).to_array("variable")


########################################################
# Several runs of model
def model_nruns_with_white_noise(nruns, Bs, V0, nT=nT):
    data = xr.Dataset()
    for nrun in range(nruns):
        data[nrun] = euler_maruyama_solve_white_noise_equilibrium_run(V0, Bs, nT=nT)
    return data.to_array("run").to_dataset("variable")



