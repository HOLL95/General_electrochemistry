import numpy as np
import os
import sys
cwd=os.getcwd()
cwd_list=cwd.split("/")
cwd_idx=([i for i,x in enumerate(cwd_list) if x=="src"][0])+1
src_loc=("/").join(cwd_list[:cwd_idx])
sys.path.append(src_loc)

import matplotlib.pyplot as plt
import math
import time
from single_e_class_unified import single_electron
from scipy.optimize import curve_fit
import pints
from pints import plot
F=96485.3329
R=8.31446261815324
Es=0.15
dE=4e-3
DeltaE=0.3
Esw=80e-3

n=1
T=298
alpha=0.5
sampling_factor=200
param_list={
"E_0":0.0,
'E_start':  Es, #(starting dc voltage - V)
'scan_increment': dE,   #(ac voltage amplitude - V) freq_range[j],#
'area': 0.07, #(electrode surface area cm^2)
'gamma': 1e-10,
"omega":0.5,
"Ru":0,
"original_gamma":1e-10,
"T":273+25,
"n":n,
'k_0': 75, #(reaction rate s-1)
'alpha': 0.5,
"sampling_factor":sampling_factor,
"SW_amplitude":Esw,
"deltaE":DeltaE
}
K=param_list["k_0"]/param_list["omega"]
print((DeltaE/dE)*50)
simulation_options={
"method":"square_wave",
"experimental_fitting":False,
"likelihood":"timeseries",
"square_wave_return":"net",
"optim_list":["E_0", "k_0", "alpha"],
"no_transient":False
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":range(0, 1),
    "experiment_time": None,
    "experiment_current": None,
    "experiment_voltage":None,
    "bounds_val":200,
}
param_bounds={
"E_0":[param_list["E_start"]-param_list["deltaE"], param_list["E_start"]],
"k_0":[0.1, 1e3],
"alpha":[0.4, 0.6]
}
omega_list=[5, 10]
noise_val=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
omega_counter=0
SWV_chain_dict={}
for k0_val in [0.5]:
    param_list["omega"]=omega_list[omega_counter]
    omega_counter+=1
    key="k0={0}".format(k0_val)
    SWV_chain_dict[key]={}
    for i in range(0, len(noise_val)):
        SW=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        plt.plot()
        end=int((DeltaE/dE)*sampling_factor)
        test=SW.test_vals([0.0, k0_val, 0.5], "timeseries")
        noisy_test=SW.add_noise(test, noise_val[i]*max(test))
        print(len(test))
        MCMC_problem=pints.SingleOutputProblem(SW, list(range(0, len(noisy_test))), noisy_test)
        log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
        log_prior=pints.UniformLogPrior([SW.param_bounds[x][0] for x in SW.optim_list]+[0], [SW.param_bounds[x][1] for x in SW.optim_list]+[2*SW.RMSE(noisy_test, test)])
        print(log_prior.n_parameters(), log_liklihood.n_parameters())
        log_posterior=pints.LogPosterior(log_liklihood, log_prior)
        mcmc_parameters=[param_list["E_0"], k0_val, param_list["alpha"]]+[SW.RMSE(noisy_test, test)]
        xs=[mcmc_parameters,
            mcmc_parameters,
            mcmc_parameters
            ]
        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)
        mcmc.set_parallel(True)
        mcmc.set_max_iterations(5000)
        chains=mcmc.run()
        means=np.zeros(len(SW.optim_list))
        for z in range(0, len(SW.optim_list)):

            total_chain=[chains[x, 500:, z] for x in range(0, 3)]
            total_chain=np.concatenate(total_chain)
            means[z]=np.mean(total_chain)
        SWV_chain_dict[key][str(noise_val[i])]=chains
np.save("SWV_sampled_MCMC_errors_irreversible_3", SWV_chain_dict)
