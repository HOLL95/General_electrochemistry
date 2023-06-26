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
from combined_heuristic_inference import combined_heuristics
import pints
from pints import plot

harm_range=list(range(1, 9))
cc=0

param_list={
       "E_0":0.3,
        'E_start':  0, #(starting dc voltage - V)
        'E_reverse':500e-3,
       
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
    "options":{
                "EIS_Cf":"CPE", "EIS_Cdl":"C", 
                "DC_pot":param_list["E_0"]+0.05, 
                "optim_list":["k_0", "gamma", "Cdl", "alpha", "cpe_alpha_faradaic", "Ru"],
                "invert_imaginary":True,},
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
for scaling in [1, 10]:
    for j in range(0, 20):
        global_options={"label":"cmaes", "test":False, "return_arg":"multiple_sigma", "weights":[1,1,scaling]}
        test=combined_heuristics(param_list, simulation_options, other_values, param_bounds, global_options,
                                EIS, Trumpet, Minima)
        print(test.common_optim_list)

        param_names=test.common_optim_list
        true_param_values=np.zeros(len(param_names))
        for i in range(0, len(param_names)):
            if param_names[i] in param_list:
                true_param_values[i]=param_list[param_names[i]]
            else:
                for dictionary in [EIS, Trumpet, Minima]:
                    if param_names[i] in dictionary["params"]:
                        true_param_values[i]=dictionary["params"][param_names[i]]
                        break
        normed_params=test.change_norm_group(true_param_values, "norm")
        print(normed_params)
        for i in range(0, 5):
            plt.plot(test.total_likelihood[:,i])
            plt.show()
        z=test.test_vals(true_param_values)
        k0_idx=test.common_optim_list.index("k_0")

        cmaes_problem=pints.SingleOutputProblem(test,np.linspace(0, 1, len(test.total_likelihood)), test.total_likelihood)
        score = pints.GaussianLogLikelihood(cmaes_problem)
        lower_bound=np.append(np.zeros(len(param_names)), [0])
        upper_bound=np.append(np.ones(len(param_names)), [10])
        CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
        best_score=-1e6
        num_runs=1
        for i in range(0, num_runs):
            x0=list(np.random.rand(len(test.common_optim_list)))+[1]
            x0=np.append(normed_params,1)
            print(x0, test.n_parameters())
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, len(param_names)+1)], boundaries=CMAES_boundaries, method=pints.CMAES)
            cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
            cmaes_fitting.set_parallel(True)
            found_parameters, found_value=cmaes_fitting.run()
            if best_score<found_value:
                dim_params=test.change_norm_group(found_parameters[:-1], "un_norm")

                best_score=found_value
                print(param_names)
                print(true_param_values)
                print(list(dim_params))   
                noise=found_parameters[-1]
        if abs(dim_params[k0_idx]-param_list["k_0"])>5:
            print("HARGH")
            continue
                
        test.global_options["label"]="MCMC"




        #noise=3.5
        MCMC_problem=pints.SingleOutputProblem(test,np.linspace(0, 1, len(test.total_likelihood)), test.total_likelihood)
        updated_lb=[test.common_bounds[x][0] for x in test.common_optim_list]+[0]
        updated_ub=[test.common_bounds[x][1] for x in test.common_optim_list]+[10*noise]
        updated_b=[updated_lb, updated_ub]
        updated_b=np.sort(updated_b, axis=0)
        log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
        log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
        log_posterior=pints.LogPosterior(log_liklihood, log_prior)

        mcmc_parameters=np.append(true_param_values, noise)
        xs=[mcmc_parameters,
            mcmc_parameters,
            mcmc_parameters
            ]

        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)

        mcmc.set_parallel(True)
        mcmc.set_max_iterations(20000)
        chains=mcmc.run()
        save_file="Combined_initial_long_{0}".format(j)
        f=open(save_file, "wb")
        np.save(f, chains)
plot.trace(chains)
plt.show()
