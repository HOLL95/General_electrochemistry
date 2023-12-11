
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
from harmonics_plotter import harmonics
from heuristic_class import Laviron_EIS
import numpy as np
import pints
from EIS_TD import EIS_TD

param_list={
    "E_0":0,
    'E_start': -10e-3, #(starting dc voltage - V)
    'E_reverse':10e-3,
    'omega':10, #8.88480830076,  #    (frequency Hz)
    "original_omega":10,
    'd_E': 10*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 5e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 75, #(reaction rate s-1)
    'alpha': 0.45,
    "E0_mean":0,
    "E0_std": 0.025,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :0,
    "time_end": -1,
    "Upper_lambda":0.64,
    'num_peaks': 10,
    "k0_shape":0.4,
    "k0_scale":0.01,
    "cpe_alpha_cdl":1
    
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["original_omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[32],
    "GH_quadrature":True,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "invert_imaginary":False,
    "Marcus_kinetics":False,
    "optim_list":[],
    "DC_pot":0,
    
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),

    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.2,0.2],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "Upper_lambda":[0.02, 10],
}
import copy
import time
scale_vals=[0.125, 0.25, 0.5, 1, 2]
std_vals=[0.01, 0.02, 0.03, 0.04, 0.05]
laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
lav_cdl_val= param_list["Cdl"]*param_list["area"]


simulation_options["data_representation"]="bode"
td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
frequencies=td.define_frequencies(-2, 6, points_per_decade=10)
common_params_1=["k0_scale", "k0_shape","Cdl"]
common_params_2=["E0_mean", "E0_std", "Cdl", "k_0"]
label_vals=["$k_{0}$ shape=","$E^0\\sigma=$"]
classes=[td, laviron]
optim=[common_params_1, common_params_2]
vals=[scale_vals, std_vals]
fig, ax=plt.subplots(2,2)
for i in range(0,1):
    for j in range(1, len(classes)):
        for m in [True, False]:
            classes[j].simulation_options["C_sim"]=m
            axes=ax[i,j]
            twinx=axes.twinx()
            if i==0:
                classes[j].simulation_options["GH_quadrature"]=False
            else:
                classes[j].simulation_options["GH_quadrature"]=True
            classes[j].def_optim_list(optim[i])
            for z in range(0, 1):
                sim_params=[param_list[x] for x in classes[j].optim_list]
                sim_params[1]=vals[i][z]
                if j==1:
                    sim_params[2]=lav_cdl_val
                    bode_vals=classes[j].simulate(sim_params, frequencies*2*math.pi)
                else:
                    bode_vals=classes[j].simulate(sim_params, frequencies)
                print(sim_params)
                EIS().bode(bode_vals, frequencies, ax=axes, twinx=twinx, compact_labels=True, data_type="phase_mag", label=label_vals[i]+str(vals[i][z]))
ax[0,0].set_title("Time domain")
ax[0,1].set_title("Equivalent circuit")
ax[0,1].legend(loc="lower left")
ax[1,1].legend(loc="lower left")
plt.show()
