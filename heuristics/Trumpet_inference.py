import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(source_loc)
import math
import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from heuristic_class import DCVTrumpet
import time
import pints
from pints import plot
fig, ax=plt.subplots()
harm_range=list(range(1, 9))
cc=0

param_list={
        "E_0":-0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':200e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 0, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        "dcv_sep":0.0,
        'sampling_freq' : (1.0/50),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 30,
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
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":["dcv_sep"]
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
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
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
    "dcv_sep":[0, 0.5]
}
#psv=single_electron("", param_list, simulation_options, other_values, param_bounds)
trumpet=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
scan_rates=[x*1e-3 for x in np.logspace(1, 3.5, 20)]
start=time.time()
trumpet.simulation_options["synthetic_noise"]=0.01
start=time.time()
noisy_trumpet_result=trumpet.e_nondim(trumpet.simulate([param_list["dcv_sep"]], scan_rates, optimise_flag=False))
trumpet.secret_data_trumpet=noisy_trumpet_result
print(time.time()-start, "external_time_1")
trumpet.simulation_options["synthetic_noise"]=0
for i in range(0, 3):
    start=time.time()
    trumpet_result=trumpet.e_nondim(trumpet.simulate([param_list["dcv_sep"]], scan_rates))
    print(time.time()-start, "external_time_2")
cc=0
labels=["Simulation", "Simulation + noise"]
for data in [trumpet_result, noisy_trumpet_result]:
    trumpet.trumpet_plot(scan_rates, data, ax=ax, colour_counter=cc, label=labels[cc])
    cc+=1
plt.xlabel("Log(scan rate (V s$^{-1}$))")
plt.ylabel("Peak position (V)")
plt.legend()
plt.show()

trumpet.def_optim_list(["E_0", "k_0", "alpha"])

MCMC_problem=pints.MultiOutputProblem(trumpet, scan_rates, noisy_trumpet_result)
updated_lb=[param_bounds[x][0] for x in trumpet.optim_list]+[0, 0]
updated_ub=[param_bounds[x][1] for x in trumpet.optim_list]+[2, 2]
updated_b=[updated_lb, updated_ub]
updated_b=np.sort(updated_b, axis=0)
log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
log_posterior=pints.LogPosterior(log_liklihood, log_prior)
mcmc_parameters=[param_list[x] for x in trumpet.optim_list]#
mcmc_parameters=np.append([param_list[x] for x in trumpet.optim_list], [0.1, 0.1])#[sim.dim_dict[x] for x in sim.optim_list]+[error]

#mcmc_parameters=np.append(mcmc_parameters,error)
xs=[mcmc_parameters,
    mcmc_parameters,
    mcmc_parameters
    ]

trumpet.simulation_options["trumpet_test"]=False
mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)

mcmc.set_parallel(True)
mcmc.set_max_iterations(5000)
chains=mcmc.run()
save_file="Trumpet_no_Ru_fitting"
f=open(save_file, "wb")
np.save(f, chains)
plot.trace(chains)

plt.show()