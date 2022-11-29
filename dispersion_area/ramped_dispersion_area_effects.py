import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
import matplotlib.pyplot as plt
import sys

harm_range=list(range(4, 10))
param_list={
    "E_0":-0.2,
    'E_start':  -500e-3, #(starting dc voltage - V)
    'E_reverse':100e-3,
    'omega':8.88,#8.88480830076,  #    (frequency Hz)
    "v":    22.35174e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :0.1,
    "time_end": None,
    'num_peaks': 30,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[param_list['E_start'],param_list['E_reverse']],
    "E0_std": [1e-5,  0.1],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
}





amplitude_vals=[50e-3, 100e-3, 150e-3, 200e-3, 250e-3, 300e-3]
fig, ax=plt.subplots(len(harm_range), len(amplitude_vals))

std_dev_vals=[0.01, 0.025, 0.05]
#plt.show()

for i in range(0, len(amplitude_vals)):
    updated_param_list=copy.deepcopy(param_list)
    ax[0, i].set_title("{0} mV".format(amplitude_vals[i]*1000))
    updated_param_list["d_E"]=amplitude_vals[i]
    sim=single_electron(None, updated_param_list, simulation_options, other_values, param_bounds)
    h_class=harmonics(sim.other_values["harmonic_range"], sim.dim_dict["omega"], 0.05)
    time=sim.t_nondim(sim.time_vec)
    sim.def_optim_list(["E_0"])
    non_dispersed=sim.i_nondim(sim.test_vals([-0.2], "timeseries"))
    plot_dict=dict(non_dispersed_time_series=non_dispersed, hanning=False, plot_func=abs, axes_list=ax[:,i])#
    dispersion_class=single_electron(None, updated_param_list, simulation_options, other_values, param_bounds)
    dispersion_class.def_optim_list(["E0_mean", "E0_std"])
    for j in range(0, len(std_dev_vals)):
        dispersed=dispersion_class.i_nondim(dispersion_class.test_vals([-0.2, std_dev_vals[j]], "timeseries"))
        plot_dict["{0}V_time_series".format(std_dev_vals[j])]=dispersed
    if i==2:
        plot_dict["legend"]={"loc":"center", "bbox_to_anchor":[1.8, 1.65], "ncol":2}
    else:
        plot_dict["legend"]=None
    h_class.plot_harmonics(time, **plot_dict)
plt.show()
