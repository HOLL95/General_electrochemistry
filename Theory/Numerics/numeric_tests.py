
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
import math
import copy
import pints
from single_e_class_unified import single_electron
from single_electron_sensitivities import Sensitivity
import numpy as np
import matplotlib.pyplot as plt
srs=np.arange(50, 1000, 10)
diffs=np.zeros(len(srs))
for z in range(0, len(srs)):
    harm_range=list(range(4, 6))
    param_list={
        "E_0":-0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
        'omega':10,  #    (frequency Hz)
        "original_omega":10,
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 5e-5, #(capacitance parameters)
        'CdlE1': 0.000653657774506,
        'CdlE2': 0.000245772700637,
        "CdlE3":-1e-6,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 10, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/srs[z]),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 5,
    }
    solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
    likelihood_options=["timeseries", "fourier"]
    time_start=1/(param_list["omega"])
    simulation_options={
        "no_transient":time_start,
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
        'phase' : [0, 2*math.pi],
        "all_freqs":[1e-3, 100],
        "all_amps":[1e-5, 0.5],
        "all_phases":[0, 2*math.pi],
    }

    sim=Sensitivity(param_list, simulation_options, other_values, param_bounds)
    """
    fixed_current=sim.test_vals([param_list[x] for x in sim.sens_params], "timeseries")
    sim.update_params([param_list[x] for x in sim.sens_params])
    adaptive_current, _, _=sim.simulate_S1()
    adaptive_current=adaptive_current[sim.time_idx]
    diffs[z]=np.mean(np.abs(np.divide(np.subtract(fixed_current, adaptive_current),adaptive_current)))"""
    if z!=0:
        prev_current=fixed_current
    fixed_current=sim.test_vals([param_list[x] for x in sim.sens_params], "timeseries")
    if z==0:
        original_time=sim.time_vec[sim.time_idx]
        interped=fixed_current
    if z!=0:
        interped=np.interp(time, sim.time_vec[sim.time_idx], fixed_current)
        diffs[z]=np.linalg.norm(np.divide(np.subtract(prev_current, interped), 1))/np.linalg.norm(interped)
    time=sim.time_vec[sim.time_idx]
    #plt.plot(potential, current)
    #plt.plot(sim.define_voltages(), fixed_current)
    #plt.title(100*sim.RMSE(fixed_current, current)/np.mean(current)
plt.plot(srs[1:], np.log10(100*diffs[1:]))
plt.show()
