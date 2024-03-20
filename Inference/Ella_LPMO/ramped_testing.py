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
import numpy as np
import pints
from scipy.optimize import minimize
from scipy.signal import decimate

data_loc="/home/henryll/Documents/Experimental_data/Ella/LPMOComplete/"

file_name="PGE+CfAA10_FTV_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")

#sblank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
dec_amount=16
volt_data=voltage_data_file[0::dec_amount, 1]


plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}

curr_dict=plot_dict
#for key in curr_dict:
#    curr_dict[key]=decimate(curr_dict[key], 16)

param_list={
    "E_0":-0.3,
    'E_start':  -400e-3, #(starting dc voltage - V)
    'E_reverse': 400e-3,
    'omega':9.349514676883269, #8.88480830076,  #    (frequency Hz)
    "v":0.014498423004208266,#0.03348950985573435,
    'd_E': 300*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 7e-11,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 30,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[32],
    "GH_quadrature":True,
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
    "harmonic_range":list(range(1,9,1)),
    "experiment_time": curr_dict["time"],
    "experiment_current": curr_dict["current"],
    "experiment_voltage":curr_dict["potential"],
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.3,0.3],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],2.5*param_list["original_gamma"]],
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
}
import copy
copied_other=copy.deepcopy(other_values)
copied_sim=copy.deepcopy(simulation_options)
copied_params=copy.deepcopy(param_list)
ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)





time_results=ferro.t_nondim(ferro.other_values["experiment_time"])
current_results=ferro.i_nondim(ferro.other_values["experiment_current"])*1e6
voltage_results=ferro.e_nondim(ferro.other_values["experiment_voltage"])
#plt.plot(ferro.other_values["experiment_time"], current_results)
interval_dict={"interval_1":[0,8.1], "interval_2":[26.9, 34.9] ,"interval_3":[52.4, 61]}
#reduced=ferro.exclude_Ramped_Faradaic(interval_dict, ferro.other_values["experiment_time"], current_results)
#plt.plot(ferro.other_values["experiment_time"], reduced)
#plt.show()
max_freq=ferro.get_input_freq(curr_dict["time"], curr_dict["current"])
dcv_voltage=ferro.e_nondim(ferro.calc_DCV())
h_class=harmonics(list(range(1, 9)),max_freq, 0.5)
fig, ax=plt.subplots(h_class.num_harmonics, 1)
half_way=len(time_results)//2
h_class.plot_harmonics(time_results[:half_way], current_time_series=current_results[:half_way], axes_list=ax, hanning=True, plot_func=abs, one_sided=True, xaxis=dcv_voltage[:half_way])
h_class.plot_harmonics(time_results[half_way:], current_time_series=current_results[half_way:], axes_list=ax, hanning=True, plot_func=abs, one_sided=True, xaxis=dcv_voltage[half_way:])
plt.show()
#ferro.get_input_params(ferro.e_nondim(voltage_results), ferro.t_nondim(time_results))
#plt.plot(time_results, voltage_results)
#plt.plot(time_results, ferro.define_voltages(no_transient=True))
#plt.show()
