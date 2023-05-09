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
from heuristic_class import DCVTrumpet, Laviron_EIS, PSV_harmonic_minimum
import time
import pints
from pints import plot

harm_range=list(range(1, 13))
cc=0

param_list={
       "E_0":0.3,
        'E_start':  0e-3, #(starting dc voltage - V)
        'E_reverse':200e-3,
        'omega':10,
        "original_omega":10,  #    (frequency Hz)
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        "dcv_sep":0.0,
        'sampling_freq' : (1.0/400),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 30,
        "cpe_alpha_faradaic":0.8
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
    "E_step_start":param_list["E_start"]-12.5e-3,
    "num_steps":100,
    "E_step_range":25e-3,

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
    'gamma': [0.1*param_list["original_gamma"],1e-9],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
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
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1]
}
#psv=single_electron("", param_list, simulation_options, other_values, param_bounds)
p_min=PSV_harmonic_minimum(param_list, simulation_options, other_values, param_bounds)
print(p_min.simulation_options["psv_copying"])
plt.plot(p_min.simulation_options["even_harms"],p_min.test_vals([], "timeseries"), label="Simulation", marker="o")

p_min.simulation_options["synthetic_noise"]=0.01
noisy_observations=p_min.test_vals([], "timeseries")
plt.plot(p_min.simulation_options["even_harms"], noisy_observations, label="Simulation + noise", marker="o")
plt.xlabel("Harmonic number")
plt.ylabel("Minimum harmonic potential offset")
plt.legend()
plt.show()
p_min.def_optim_list(["k_0", "alpha", "Ru"])
p_min.simulation_options["synthetic_noise"]=0


p_min.simulation_options["label"]="cmaes"
cmaes_problem=pints.MultiOutputProblem(p_min, p_min.simulation_options["even_harms"], noisy_observations)
score = pints.GaussianLogLikelihood(cmaes_problem)
lower_bound=np.append(np.zeros(len(p_min.optim_list)), [0])
upper_bound=np.append(np.ones(len(p_min.optim_list)), [1])
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
true_params=[param_list[x] for x in p_min.optim_list]
print(true_params)
x0=p_min.change_norm_group(true_params, "norm")+[0.01]
print(x0)
num_runs=1
z=-1
found=False
params=np.zeros((num_runs, p_min.n_parameters()+1))
for i in range(0, num_runs):
    z+=1
    print(len(x0), p_min.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.05 for x in range(0, p_min.n_parameters()+1)], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()
    dim_params=p_min.change_norm_group(found_parameters[:-1], "un_norm")
    print(dim_params, found_parameters[-1])



p_min.simulation_options["label"]="MCMC"





MCMC_problem=pints.MultiOutputProblem(p_min, p_min.simulation_options["even_harms"], noisy_observations)
updated_lb=[param_bounds[x][0] for x in p_min.optim_list]+[0]
updated_ub=[param_bounds[x][1] for x in p_min.optim_list]+[1]
updated_b=[updated_lb, updated_ub]
updated_b=np.sort(updated_b, axis=0)
log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
log_posterior=pints.LogPosterior(log_liklihood, log_prior)
mcmc_parameters=[param_list[x] for x in p_min.optim_list]#
mcmc_parameters=np.append([param_list[x] for x in p_min.optim_list], [found_parameters[-1]])#[sim.dim_dict[x] for x in sim.optim_list]+[error]

#mcmc_parameters=np.append(mcmc_parameters,error)
xs=[mcmc_parameters,
    mcmc_parameters,
    mcmc_parameters
    ]

p_min.simulation_options["trumpet_test"]=False
mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)

mcmc.set_parallel(True)
mcmc.set_max_iterations(10000)
chains=mcmc.run()
save_file="Harm_min_Ru_fitting_real"
f=open(save_file, "wb")
np.save(f, chains)
plot.trace(chains)
plt.show()
