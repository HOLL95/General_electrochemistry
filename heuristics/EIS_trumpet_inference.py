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
from heuristic_class import DCVTrumpet, Laviron_EIS
import time
import pints
from pints import plot

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
        'Cdl': 1e-5, #(capacitance parameters)
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
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "EIS_Cf":"CPE",
    "EIS_Cdl":"C",
    "DC_pot":param_list["E_0"]
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
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1]
}
#psv=single_electron("", param_list, simulation_options, other_values, param_bounds)

from MCMC_plotting import MCMC_plotting


trumpet_simulation_options={
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
    "optim_list":[]
}
trumpet_param_list=copy.deepcopy(param_list)
trumpet_param_list["omega"]=0
trumpet=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
scan_rates=[x*1e-3 for x in np.logspace(1, 4, 20)]
trumpet.simulation_options["synthetic_noise"]=0.01
noisy_trumpet_result=trumpet.simulate([], scan_rates, optimise_flag=False)
trumpet.simulation_options["synthetic_noise"]=0
trumpet.trumpet_plot(scan_rates, noisy_trumpet_result, colour_counter=0)
trumpet.def_optim_list(["E_0", "k_0"])
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
max_iter=1000
mcmc.set_parallel(True)
mcmc.set_max_iterations(max_iter)
chains=mcmc.run()
plot.trace(chains)
plt.show()



mplot=MCMC_plotting(burn=int(max_iter/2))
catted_param_chains=mplot.concatenate_all_chains(chains)[:-2]
chain_dict=dict(zip(trumpet.optim_list, catted_param_chains))

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list([ "k_0", "alpha","gamma", "Cdl", "cpe_alpha_faradaic", "Ru"])

multivariate_params=["k_0"]#list(set(trumpet.optim_list).intersection(set(laviron.optim_list)))
multivariate_chains=[chain_dict[x] for x in multivariate_params]
#multivariate_means=np.mean(multivariate_chains, axis=1)
#multivariate_cov=np.cov(multivariate_chains)



vals=[param_list[x] for x in laviron.optim_list]
frequency_powers=np.arange(1, 5, 0.05)
frequencies=[10**x for x in frequency_powers]
sim=laviron.simulate(vals, frequencies)
noise_val=0.01
noisy_data=np.column_stack((laviron.add_noise(sim[:,0], noise_val, method="proportional"), laviron.add_noise(sim[:,1], noise_val, method="proportional")))
noisy_data=np.column_stack((laviron.add_noise(sim[:,0], noise_val*np.mean(sim[:,0])), laviron.add_noise(sim[:,1], noise_val*np.mean(-sim[:,1]))))
#laviron.dim_dict["gamma"]=5e-11
laviron.secret_data_EIS=noisy_data
fig, ax=plt.subplots()
for data in [sim, noisy_data]:
    laviron.simulator.nyquist(data, ax=ax, orthonormal=False, scatter=1)
plt.show()


laviron.simulation_options["label"]="cmaes"
cmaes_problem=pints.MultiOutputProblem(laviron,frequencies, noisy_data)
score = pints.GaussianLogLikelihood(cmaes_problem)
lower_bound=np.append(np.zeros(len(laviron.optim_list)), [0, 0])
upper_bound=np.append(np.ones(len(laviron.optim_list)), [100, 100])
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
true_params=[param_list[x] for x in laviron.optim_list]
print(true_params)
x0=laviron.change_norm_group(true_params, "norm")+[5, 5]
print(x0)

num_runs=2
z=-1
found=False
best_score=-1e6
params=np.zeros((num_runs, laviron.n_parameters()+1))
for i in range(0, num_runs):
    x0=list(np.random.rand(len(true_params)))+[5,5]
    z+=1
    print(len(x0), laviron.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, laviron.n_parameters()+2)], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()
    if found_value>best_score:
        dim_params=laviron.change_norm_group(found_parameters[:-2], "un_norm")
        
        print(dim_params, found_parameters[-2:])
        

dim_param_dict=dict(zip(laviron.optim_list, dim_params))
laviron.def_optim_list(["k_0","alpha", "Cdl", "Ru","cpe_alpha_faradaic"])
laviron.simulation_options["label"]="MCMC"


MCMC_problem=pints.MultiOutputProblem(laviron, frequencies, noisy_data)



updated_lb=[param_bounds[x][0] for x in laviron.optim_list if x not in multivariate_params]+[0, 0]
updated_ub=[param_bounds[x][1] for x in laviron.optim_list if x not in multivariate_params]+[100, 100]
updated_b=[updated_lb, updated_ub]
updated_b=np.sort(updated_b, axis=0)


log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
#no_corr=np.identity(len(multivariate_params))*np.std(multivariate_chains, axis=1)
#print(multivariate_means, no_corr)
#multi_gaussian_prior=pints.MultivariateGaussianLogPrior(multivariate_means, no_corr)


gaussian_prior=pints.GaussianLogPrior(np.mean(chain_dict["k_0"]), np.std(chain_dict["k_0"]))
#gaussian_prior=pints.GaussianLogPrior(102.37460941534808, 3.8713749570928777)

uniform_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
log_prior=pints.ComposedLogPrior(gaussian_prior, uniform_prior)

log_posterior=pints.LogPosterior(log_liklihood, log_prior)
mcmc_parameters=[param_list[x] for x in laviron.optim_list]#
mcmc_parameters=np.append([dim_param_dict[x] for x in laviron.optim_list], [found_parameters[-2:]])#[laviron.dim_dict[x] for x in laviron.optim_list]+[error]
print(mcmc_parameters)
#mcmc_parameters=np.append(mcmc_parameters,error)
xs=[mcmc_parameters,
    mcmc_parameters,
    mcmc_parameters
    ]


mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)
laviron.simulation_options["test"]=False
mcmc.set_parallel(False)
max_iter=20000
mcmc.set_max_iterations(max_iter)
chains=mcmc.run()
plot.trace(chains)
plt.show()






