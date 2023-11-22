

import matplotlib.pyplot as plt
import pints

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
from heuristic_class import DCVTrumpet



param_list={
    "E_0":0.2,
    'E_start':  -0.2, #(starting dc voltage - V)
    'E_reverse': 0.6665871839451643,
    'omega':0, #8.88480830076,  #    (frequency Hz)
    "v":0.0338951299038171,#0.03348950985573435,
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 90,  #     (uncompensated resistance ohms)
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
    'sampling_freq' : (1.0/50),
    'phase' :0,
    "time_end": -1,
    "dcv_sep":0.1,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=param_list["sampling_freq"]*2
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[20],
    "GH_quadrature":True,
    "test": False,
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "record_exps":False,
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0, 0.4],
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
    "dcv_sep":[0, 0.2],
}

trumpets=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
trumpets.def_optim_list(["E_0", "k_0", "dcv_sep"])

filename="fc_dc_positions.txt"
file=np.loadtxt(filename, delimiter=",")
scan_rates=file[:,0]
data=np.column_stack((file[:,1], file[:,2]))
vals={'E_0': 0.18401745010050377, 'k_0': 74.4544133828206, 'dcv_sep': 0.0069150204440147165}

sim=trumpets.simulate([vals[x] for x in trumpets.optim_list], np.divide(scan_rates, 1000))
fig, ax=plt.subplots()
fitting_scan_rates=np.divide(scan_rates, 1000)
trumpets.trumpet_plot(scan_rates, data,ax=ax, label="Data")
trumpets.trumpet_plot(scan_rates, trumpets.e_nondim(sim),ax=ax, label="Simulation")
ax.legend()
plt.show()
fitting_data=data/trumpets.nd_param.c_E0
cmaes_problem=pints.MultiOutputProblem(trumpets,fitting_scan_rates,fitting_data)
score = pints.GaussianLogLikelihood(cmaes_problem)
trumpets.simulation_options["label"]="cmaes"
trumpets.simulation_options["trumpet_test"]=False
trumpets.secret_data_trumpet=fitting_data
lower_bound=np.append(np.zeros(len(trumpets.optim_list)), [0]*trumpets.n_outputs())
upper_bound=np.append(np.ones(len(trumpets.optim_list)), [50]*trumpets.n_outputs())
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
for i in range(0, 1):
    x0=list(np.random.rand(len(trumpets.optim_list)))+[5]*trumpets.n_outputs()
    print(len(x0), len(trumpets.optim_list), cmaes_problem.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, trumpets.n_parameters()+trumpets.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
    trumpets.simulation_options["test"]=False
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()   
    real_params=trumpets.change_norm_group(found_parameters[:-trumpets.n_outputs()], "un_norm")

    print(dict(zip(trumpets.optim_list, list(real_params))))