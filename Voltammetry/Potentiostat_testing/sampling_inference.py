import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from pints import plot
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
import matplotlib.pyplot as plt
import sys

harm_range=list(range(4, 10))
for samples in ["_normal_sampling", ""]:
    for frequency in [10, 50, 100, 200]:
        param_list={
            "E_0":0.2,
            'E_start':  -200e-3, #(starting dc voltage - V)
            'E_reverse':900e-3,
            'omega':frequency,#8.88480830076,  #    (frequency Hz)
            "original_omega":frequency,
            'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
            'area': 0.07, #(electrode surface area cm^2)
            'Ru': 100.0,  #     (uncompensated resistance ohms)
            'Cdl': 1e-4, #(capacitance parameters)
            'CdlE1': 0.000653657774506,
            'CdlE2': 0.000245772700637,
            "CdlE3":-1e-6,
            'gamma': 2e-11,
            "original_gamma":2e-11,        # (surface coverage per unit area)
            'k_0': 1000, #(reaction rate s-1)
            'alpha': 0.5,
            "E0_mean":0.2,
            "E0_std": 0.09,
            "cap_phase":3*math.pi/2,
            "alpha_mean":0.5,
            "alpha_std":1e-3,
            'sampling_freq' : (1.0/200),
            'phase' :3*math.pi/2,
            "time_end": None,
            'num_peaks': 30,
        }
        solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
        likelihood_options=["timeseries", "fourier"]
        time_start=1/(param_list["omega"])
        simulation_options={
            "no_transient":time_start,
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
            "optim_list":[]
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
            'CdlE1': [-0.05,0.15],#0.000653657774506,
            'CdlE2': [-0.01,0.01],#0.000245772700637,
            'CdlE3': [-0.01,0.01],#1.10053945995e-06,
            'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
            'k_0': [0.1, 1e4], #(reaction rate s-1)
            'alpha': [0.4, 0.6],
            "cap_phase":[math.pi/2, 2*math.pi],
            "E0_mean":[param_list['E_start'],param_list['E_reverse']],
            "E0_std": [1e-5,  0.1],
            "alpha_mean":[0.4, 0.65],
            "alpha_std":[1e-3, 0.3],
            "k0_shape":[0,1],
            "k0_scale":[0,1e4],
            "k0_range":[1e2, 1e4],
            'phase' : [math.pi, 2*math.pi],
        }

        sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        current=sim.test_vals([], "timeseries")
        potential=sim.e_nondim(sim.define_voltages(transient=True))

        time=sim.t_nondim(sim.time_vec[sim.time_idx])
        reduced_time=np.arange(time[0], time[-1], 0.0002)
        reduced_current=np.interp(reduced_time, time, current)
        error=0.02*max(reduced_current)
        sim.def_optim_list(["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha", "phase", "cap_phase"])
        if samples=="_normal_sampling":
            noisy_current=sim.add_noise(current, error)
            MCMC_problem=pints.SingleOutputProblem(sim, time, noisy_current)
        else:
            noisy_current=sim.add_noise(reduced_current, error)
            sim.simulation_options["sample_times"]=reduced_time/sim.nd_param.c_T0
            
            MCMC_problem=pints.SingleOutputProblem(sim, reduced_time, noisy_current)


        updated_lb=[param_bounds[x][0] for x in sim.optim_list]+[0]
        updated_ub=[param_bounds[x][1] for x in sim.optim_list]+[20]
        #updated_lb=[0 for x in sim.optim_list  ]+[0]
        #updated_ub=[1 for x in sim.optim_list ]+[1]
        updated_b=[updated_lb, updated_ub]
        updated_b=np.sort(updated_b, axis=0)
        #log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
        log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
        log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
        log_posterior=pints.LogPosterior(log_liklihood, log_prior)
        #mcmc_parameters=sim.change_norm_group(cmaes_params, "norm")
        mcmc_parameters=[sim.dim_dict[x] for x in sim.optim_list]+[error]
        print(mcmc_parameters)
        #mcmc_parameters=np.append(mcmc_parameters,error)
        xs=[mcmc_parameters,
            mcmc_parameters,
            mcmc_parameters
            ]
        print(mcmc_parameters)
        log_params=["k_0", "Ru"]
        transforms=[pints.IdentityTransformation(n_parameters=1) if x not in log_params else pints.LogTransformation(n_parameters=1) for x in sim.optim_list+["error"]]
        #transforms+=[pints.IdentityTransformation(n_parameters=1)]
        print(len(transforms))
        print(sim.n_parameters())
        MCMC_transform=pints.ComposedTransformation(*transforms)
        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)#, transformation=MCMC_transform)

        mcmc.set_parallel(True)
        mcmc.set_max_iterations(25000)
        chains=mcmc.run()
        save_file="MCMC/{0}_Hz_PSV_2_pc_MCMC_no_transform{1}".format(frequency, samples)
        f=open(save_file, "wb")
        np.save(f, chains)
