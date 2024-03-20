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
    'omega':0.4284083843231201, #8.88480830076,  #    (frequency Hz)
    "v":0.015,#0.03348950985573435,
    'd_E': 250*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 5e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 0,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 0, #(reaction rate s-1)
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
    "likelihood":likelihood_options[0],
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
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.9*param_list['omega'],1.1*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-4], #(capacitance parameters)
    'CdlE1': [-1,1],#0.000653657774506,
    'CdlE2': [-0.5,0.5],#0.000245772700637,
    'CdlE3': [-0.1,0.1],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],2.5*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0, 2*math.pi],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0, 2*math.pi],
}
import copy
copied_other=copy.deepcopy(other_values)
copied_sim=copy.deepcopy(simulation_options)
copied_params=copy.deepcopy(param_list)
ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)





time_results=ferro.t_nondim(ferro.other_values["experiment_time"])
current_results=ferro.other_values["experiment_current"]
voltage_results=ferro.e_nondim(ferro.other_values["experiment_voltage"])
plt.plot(voltage_results, label="Data")
plt.plot(ferro.e_nondim(ferro.define_voltages(transient=False)), label="Sim")
plt.legend()

plt.show()
plt.plot(ferro.other_values["experiment_time"], current_results)
interval_dict={"interval_1":[0,8.1], "interval_2":[26.9, 34.9] ,"interval_3":[52.4, 61]}
reduced=ferro.exclude_Ramped_Faradaic(interval_dict, ferro.other_values["experiment_time"], current_results)
plt.plot(ferro.other_values["experiment_time"], reduced)
plt.show()
ferro.def_optim_list(["Cdl", "CdlE1", "CdlE2", "CdlE3", "omega", "phase", "cap_phase"])
params=[1.826994586925516e-05, -0.9999999999999991, -0.00786493265843724, 0.001430102343984524, 0.41788176451764303, 5.523902646477277, 6.283185307179585]
pure_cdl=ferro.test_vals(params, "timeseries")
plt.plot(time_results, current_results)
plt.plot(time_results, pure_cdl)
plt.show()



cmaes_problem=pints.SingleOutputProblem(ferro,time_results,reduced)
score = pints.GaussianLogLikelihood(cmaes_problem)
ferro.simulation_options["Ramped_Cdl_only"]=interval_dict
ferro.simulation_options["label"]="cmaes"
ferro.secret_data_time_series=reduced
print(ferro.optim_list)
lower_bound=np.append(np.zeros(len(ferro.optim_list)), [0]*ferro.n_outputs())

upper_bound=np.append(np.ones(len(ferro.optim_list)), [50]*ferro.n_outputs())
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
x0=list(np.random.rand(len(ferro.optim_list)))+[5]*ferro.n_outputs()
sigma=[0.075 for x in range(0, ferro.n_parameters()+ferro.n_outputs())]
print(len(x0), ferro.n_parameters(), ferro.n_outputs(), len(sigma))
cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-6)

cmaes_fitting.set_parallel(True)
ferro.simulation_options["test"]=False
found_parameters, found_value=cmaes_fitting.run()   
real_params=ferro.change_norm_group(found_parameters[:-ferro.n_outputs()], "un_norm")
print(list(real_params))
sim_current=ferro.simulate(found_parameters[:-ferro.n_outputs()], "timeseries")
plt.plot(ferro.other_values["experiment_time"], reduced)
plt.plot(ferro.other_values["experiment_time"], sim_current)
plt.show()