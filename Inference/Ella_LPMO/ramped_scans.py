import matplotlib.pyplot as plt
import math
import os
import sys
import re
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
harm_range=list(range(1,7,1))
import numpy as np
frequencies=np.flip([0.5, 1, 2.5, 5])
scan_rates=[5e-3,7e-3, 10e-3, 20e-3]
amplitudes=[150e-3, 200e-3, 250e-3]
import copy
fig, ax=plt.subplots(len(harm_range), 3)
param_list={
    "E_0":0.1,
    'E_start': -0.2, #(starting dc voltage - V)
    'E_reverse':0.45,
    'omega':1, #8.88480830076,  #    (frequency Hz)
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    "v":10e-3,
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.0000245772700637,
    "CdlE3":0,#-1e-6,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1.5, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0,
    "E0_std": 0.025,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 10,
    "k0_shape":0.4,
    "k0_scale":75,
    "dcv_sep":0,
    
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
    "dispersion_bins":[32],
    "GH_quadrature":True,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "invert_imaginary":False,
    "Marcus_kinetics":False,
    "optim_list":[],
    
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,

    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.5, 0.5],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.2,0.2],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [1, 200], #(reaction rate s-1)
    'alpha': [0.4, 0.8],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "dcv_sep":[0, 0.2]

}
change_dict={"omega":frequencies,"d_E":amplitudes, "v":scan_rates, }
unit_dict={"d_E":"V", "v":"mV/s", "omega":"Hz"}
scale_dict={"d_E":1, "v":1000, "omega":1}
change_keys=list(change_dict.keys())
for i in range(0, len(change_keys)):
    key=change_keys[i]
    
    for j in range(0, len(change_dict[key])):
        copy_params=copy.deepcopy(param_list)
        copy_params[key]=change_dict[key][j]
        h_class=harmonics(other_values["harmonic_range"], copy_params["omega"], 0.25)
        ramped=single_electron("",copy_params, simulation_options, other_values, param_bounds)
        copy_options=copy.deepcopy(simulation_options)
        copy_options["method"]="dcv"
        dcv=single_electron("",copy_params, copy_options, other_values, param_bounds)
        DC_pot=dcv.e_nondim(dcv.define_voltages())
        clean_sim=ramped.i_nondim(ramped.test_vals([], "timeseries"))
        sim=ramped.add_noise(clean_sim, 0.01*max(clean_sim))
        times=ramped.t_nondim(ramped.time_vec)
        plot_dict={"plot_func":abs, "hanning":True, "axes_list":ax[:,i], "xaxis":DC_pot}
        name="%.2f%s_time_series" % (change_dict[key][j]*scale_dict[key], unit_dict[key])
        plot_dict[name]=sim
        h_class.plot_harmonics(times, **plot_dict)
plt.show()