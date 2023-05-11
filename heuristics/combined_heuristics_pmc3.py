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
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from combined_heuristic_inference import combined_heuristics
from MCMC_plotting import MCMC_plotting
import pymc as pm 
import pints
from pints import plot
import time
import pytensor
import pytensor.tensor as pt
harm_range=list(range(1, 9))
cc=0
mplot=MCMC_plotting()
param_list={
       "E_0":-0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':0,
       
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

}
other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[5, 15],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
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

frequency_powers=np.arange(1, 5, 0.05)
frequencies=[10**x for x in frequency_powers]
EIS={"name":"EIS",
    "params":{"cpe_alpha_faradaic":0.8, "v":1, "omega":0, "cpe_alpha_faradaic":0.8},
    "options":{"EIS_Cf":"CPE", "EIS_Cdl":"C", "invert_imaginary":True,
    "DC_pot":param_list["E_0"]+0.05, 
    "optim_list":["k_0", "gamma", "Cdl", "alpha", "cpe_alpha_faradaic", "Ru"]},
    "others":{}, 
    "bounds":{},
    "times":frequencies,
    "noise":0.01, 
    "noise_bounds":[0, 10]
    }
Trumpet={"name":"Trumpet",
    "params":{"v":100e-3, "omega":0},
    "options":{
    "optim_list":["k_0", "E_0"]},
    "others":{}, 
    "bounds":{},
    "times":[x*1e-3 for x in np.logspace(1, 3.5, 20)],
    "noise":0.01
    }
Minima={"name":"Harmonic_min",
    "params":{"omega":10, "original_omega":10},
    "options":{
    "E_step_start":param_list["E_start"]-12.5e-3,
    "method":"sinusoidal",
    "num_steps":75,
    "E_step_range":25e-3,
    "psv_copying":True,
    "optim_list":["k_0", "alpha"]},
    "others":{}, 
    "bounds":{},
    "times":[],
    "noise":0.01
    }

import arviz as az
noise_dist_names=["EIS_real_sigma", "EIS_real_nu", "EIS_imag_sigma", "EIS_imag_nu", "Trumpet_red_sigma", "Trumpet_ox_sigma", "min_sigma"]

global_options={"label":"cmaes", "test":False, "return_arg":"individual", "weights":[1,1,1]}
sim_class=combined_heuristics(param_list, simulation_options, other_values, param_bounds, global_options,
                        EIS, Trumpet, Minima)
print(sim_class.common_optim_list)
def gaussian_log_likelihood(simulation, data, sigma):
    n=len(data)
    log_likelihood = -(n/2)*np.log(2*np.pi) - n*np.log(sigma) - (1/(2*sigma**2))*np.sum(np.power(simulation-data, 2))

    return log_likelihood
def multitiplicative_gaussian_likelihood(simulation, data, sigma, eta):
    n=len(data)
    logn=0.5 * n*  np.log(2 * np.pi)
    log_likelihood = -logn - np.sum(np.sum(np.log(simulation**eta * sigma), axis=0)+ 0.5 / sigma**2 * np.sum((data - simulation)**2/ simulation ** (2 * eta), axis=0))
    return log_likelihood

param_names=sim_class.common_optim_list
true_param_vals=np.zeros(len(param_names))
for i in range(0, len(param_names)):
    if param_names[i] in param_list:
        true_param_vals[i]=param_list[param_names[i]]
    else:
        for dictionary in [EIS, Trumpet, Minima]:
            if param_names[i] in dictionary["params"]:
                true_param_vals[i]=dictionary["params"][param_names[i]]
                break
normed_params=sim_class.change_norm_group(true_param_vals, "norm")
true_param_vals_dict=dict(zip(sim_class.common_optim_list, true_param_vals))

sim_class.test_vals(true_param_vals)

def custom_log_likelihood(parameters, data, sim_class):
    parameters=parameters[0]
    min_sigma=parameters[-1]
    trumpet_sigmas=parameters[-3:-1]
    eis_noise_params=parameters[-7:-3]
    simulation=sim_class.test_vals(parameters[:-7])
    EIS_likelihoods=[0,0]
    trumpet_likelihoods=[0,0]
    for i in range(0, 2):
        EIS_likelihoods[i]=multitiplicative_gaussian_likelihood(simulation[0][:,i], data[0][:,i], eis_noise_params[2*i], eis_noise_params[(2*i)+1])
        trumpet_likelihoods[i]=gaussian_log_likelihood(simulation[1][:,i], data[1][:,i],trumpet_sigmas[i])
    min_likelihood=[gaussian_log_likelihood(simulation[2], data[2], min_sigma)]
    all_likelihoods=EIS_likelihoods+trumpet_likelihoods+min_likelihood

    return np.array(np.sum(all_likelihoods))
class LogLike(pt.Op):
    itypes = [pt.dvector]
    otypes = [pt.dscalar]

    def __init__(self, data, func, sim_class):
        self.data = data
        self.func=func
        self.sim_class=sim_class

    def perform(self, node, inputs, outputs):
        logl=self.func(inputs, self.data, self.sim_class)
        outputs[0][0]=logl

        
c=LogLike(sim_class.total_likelihood, custom_log_likelihood,sim_class)  

for j in range(0, 3):
    with pm.Model() as model:
        
        parameter_distributions=[pm.Uniform(x, lower=param_bounds[x][0], upper=param_bounds[x][1], initval=true_param_vals_dict[x]) for x in sim_class.common_optim_list]
        noise_distributions=[pm.Uniform(x, lower=0, upper=10, initval=1) for x in noise_dist_names]
        composed_dist=parameter_distributions+noise_distributions
        print([type(x) for x in composed_dist])
        theta =pt.as_tensor_variable(composed_dist)
        save_str= "/home/henney/Documents/Oxford/General_electrochemistry/heuristics/Pymc_combined_inference_{0}.nc".format(j)
        pm.Potential("likelihood", c(theta))
        idata_mh = pm.sample(10000, tune=1000, chains=3, cores=4)
        try:
            idata_mh.to_netcdf(save_str)
        except:
            pints_list=mplot.convert_idata_to_pints_array(idata_mh)
            save_file="Pymc_combined_inference"
            f=open(save_file, "wb")
            np.save(f, pints_list)
            f.close()
        
    