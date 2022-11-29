import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from pints import plot
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
harm_range=list(range(4, 6))
from scipy import interpolate
param_list={
    "E_0":-0.2,
    'E_start':  -600e-3, #(starting dc voltage - V)
    'E_reverse':-100e-3,
    'omega':8.88480830076,  #    (frequency Hz)
    "v":200e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0.000653657774506,
    'CdlE2': 0.000245772700637,
    "CdlE3":-1e-6,
    'gamma': 2e-11,
    "original_gamma":2e-11,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 5,
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
    "likelihood":likelihood_options[0],
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
    'k_0': [0.1, 1e4], #(reaction rate s-1)
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
    "all_freqs":[1e-5, 2000],
    "all_amps":[1e-5, 0.5],
    "all_phases":[0, 2*math.pi],
}

sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
rpotential=sim.e_nondim(sim.define_voltages())
rcurrent=sim.test_vals([], "timeseries")
plt.plot(rpotential)
plt.show()
sim.simulation_options["method"]="sum_of_sinusoids"
from FT_potential_tests import FT_potential_osc
sines=FT_potential_osc()
keys=list(sines.sinusoid_dict.keys())
values=[sines.sinusoid_dict[key] for key in keys]
sim.def_optim_list(keys)
z=sim.test_vals(values, "timeseries")
plt.plot(sim.t_nondim(sim.time_vec), sim.e_nondim(sim.define_voltages()))
plt.plot(sim.t_nondim(sim.time_vec), rpotential, alpha=0.5)
#plt.plot(sim.t_nondim(sim.time_vec), 0.8*sim.nd_param.c_E0*np.sin(sim.nd_param.nd_param_dict["freq_array"][0]*sim.t_nondim(sim.time_vec)))
#plt.plot(sim.t_nondim(sim.time_vec), 0.8*sim.nd_param.c_E0*np.sin(sim.dim_dict["freq_array"][0]*sim.t_nondim(sim.time_vec)))

plt.show()
plt.plot(sim.t_nondim(sim.time_vec), z)
plt.show()
simulation_options["method"]="sinusoidal"
sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
sim.def_optim_list(["omega", "E_0"])
z=sim.test_vals([1, 0.1], "timeseries")
print(sim.nd_param.nd_param_dict["nd_omega"], "freq_2")
print(sim.nd_param.c_T0)
