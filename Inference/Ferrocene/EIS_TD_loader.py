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
from EIS_TD import EIS_TD
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

plt.show()

DC_val=0
param_list={
       "E_0":0,
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
        "num_peaks":10,
        "Cdl_std":1e-5,
        "Cdl_mean":1e-5
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
    
}
import copy

td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
freqs=np.multiply(frequencies, 2*math.pi)
fig, ax=plt.subplots(1,1)
twinx=ax.twinx()
EIS().bode(np.column_stack((real, imag)), frequencies,ax=ax, twinx=twinx,label="Data", lw=2, compact_labels=True)
labels=["EC fit","PSV/100", "PSV/10", "PSV"]

cdl_vals=[1.4936235822043384e-06,0.000342081409583126*0.01,0.000342081409583126*0.1,0.000342081409583126]
for i in range(0, len(cdl_vals)):
    cdl_val=cdl_vals[i]
    ramped_params=[0.2591910307724134-0.24, 0.0674086382052161, 177.04633092062943, 88.31972285297374,cdl_val , 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, 8.794196510802587, 0, 0, 0.5999998004431891]
    ramped_params=[0.2515085054963522-0.24, 0.0637810584632682, 62.915075289229755, 109.99988420501067, cdl_val, 0, 0, 0,6.334048572909808e-11, 8.799273223827607,0,0, 0.6]
    ramped_param_list=["E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"]
    r_dict=dict(zip(ramped_param_list, ramped_params))
    td.def_optim_list(["E0_mean","E0_std","gamma","k_0" , "Cdl", "alpha", "Ru", "phase", "cap_phase"])
    sim_vals=[r_dict[x] for x in td.optim_list]


    sim=td.simulate(sim_vals, freqs)
  

    EIS().bode(sim, frequencies, ax=ax, twinx=twinx, data_type="phase_mag", label=labels[i], compact_labels=True)
ax.legend()
plt.show()
