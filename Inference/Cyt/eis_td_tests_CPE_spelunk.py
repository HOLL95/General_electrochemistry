
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
from EIS_optimiser import EIS_genetics
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
import numpy as np
import pints
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=0
frequencies=np.flip(data[:,2])
param_list={
       "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':1,  #    (frequency Hz)
        "original_omega":1,
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 1e-6, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "k0_scale":100, 
        "k0_shape":0.1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "E0_mean":DC_val,
        "E0_std":0.02,
        "cap_phase":0,
        "num_peaks":20,
        "Cdl_logm":1e-5,
        "Cdl_logc":1e-5,
        "cap_phase_logm":0,
        "cap_phase_logc":0,
        "cap_phase_m":0,
        "cap_phase_c":10,
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
 
    "data_representation":"bode",
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'E0_mean':[-0.4, -0.1],
    'E0_std':[1e-3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[0, 2*math.pi],
    "cap_phase":[0, 2*math.pi],
    "k0_shape":[0,100],
    "k0_scale":[0,2],
    "Cdl_logm":[-100, 100],
    "Cdl_logc":[-100, 100],
    "cap_phase_logc":[-100, 100],
    "cap_phase_c":[-100,100]
    
}
for key in param_list.keys():
    if key not in param_bounds:
        param_bounds[key]=[0.1*param_list[key], 10*param_list[key]]
num_peaks=[20]





td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
freqs=td.define_frequencies(-1,6)
td.def_optim_list(["E_0","gamma","k_0" , "Cdl", "alpha", "Ru", "phase", "cap_phase"])
sim_vals=[0, 1e-10*0, 100, 1.000000000000003e-05, 0.55, 250, 0, 0]


sim=td.simulate(sim_vals, freqs)
fig, ax=plt.subplots(1,1)
twinx=ax.twinx()
EIS().bode(sim, freqs, ax=ax, twinx=twinx, data_type="phase_mag")
td.def_optim_list(["E_0","gamma","k_0" , "Cdl_logm", "Cdl_logc","alpha", "Ru", "phase", "cap_phase_logm", "cap_phase_logc"])
sim_vals=[0, 1e-10*0, 100, 1.000000000000003e-06, 1e-6, 0.55, 250, 0,0.1, 0.1]


sim2=td.simulate(sim_vals, freqs)

EIS().bode(sim2, freqs, ax=ax, twinx=twinx, data_type="phase_mag")


plt.show()
    

