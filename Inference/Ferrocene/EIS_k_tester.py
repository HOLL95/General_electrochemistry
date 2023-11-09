import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_class import EIS
from heuristic_class import Laviron_EIS
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)
truncate=10
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])
EIS().bode(np.column_stack((real, imag)), frequencies)
plt.show()


param_list={
        "E_0":0.24,
        'E_start':  0.24-10e-3, #(starting dc voltage - V)
        'E_reverse':0.24+10e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "Cfarad":0,
        "E0_mean":0,
        "E0_std": 0.025,
        "k0_shape":0.4,
        "k0_scale":75,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[100],
    "GH_quadrature":False,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "EIS_Cf":"CPE",
    "EIS_Cdl":"CPE",
    "DC_pot":240e-3,
    "Rct_only":False,
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0.15, 0.35],
    "E0_mean":[0.15, 0.35],
    "E0_std":[0.001, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-7],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "k0_shape":[0,10],
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
}
import copy

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["E_0","k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic","phase"])

spectra=np.column_stack((real, imag))
#EIS().bode(spectra, frequencies)
#plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)

"""sim_data=laviron.simulate([cdl_only[x] for x in laviron.optim_list], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
ax.legend()
ax.set_title("C fit")
plt.show()

"""
params={'E_0': 0.20225617873352192, 'k_0': 57.306676595213354, 'gamma': 4.6405804443127187e-11, 'Cdl': 8.75161453231382e-06, 'alpha': 0.3774297334729768, 'Ru': 80.92159706659638, 'cpe_alpha_cdl': 0.756284207176267, 'cpe_alpha_faradaic': 0.8447336542387816, 'phase': -0.7048512714370077}
crap_eo={'E0_mean': 0.35, 'E0_std': 0.04597377187031984, 'k_0': 0.9170295784159767, 'gamma': 5.4039057494043345e-09, 'Cdl': 9.233041679008419e-06, 'alpha': 0.6499999999999999, 'Ru': 80.3618065939186, 'cpe_alpha_cdl': 0.7495703342169231, 'cpe_alpha_faradaic': 0.9965534124836868, 'phase': 0.08174799951007117}


laviron.simulation_options["label"]="MCMC"
laviron.simulation_options["data_representation"]="bode"
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx)
for k in [10,params["k_0"], 75, 150, 200,1000]:
    params={'E_0': 0.20225617873352192, 'k_0': k, 'gamma': 4.6405804443127187e-11, 'Cdl': 8.75161453231382e-06, 'alpha': 0.3774297334729768, 'Ru': 80.92159706659638, 'cpe_alpha_cdl': 0.756284207176267, 'cpe_alpha_faradaic': 0.8447336542387816, 'phase': -0.7048512714370077}

    vals=[params[x] for x in laviron.optim_list]
    sim_data=laviron.simulate(vals, fitting_frequencies)
    
    
    EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, data_type="phase_mag")
    ax.set_title("No Cdl_f fit")
plt.show()
