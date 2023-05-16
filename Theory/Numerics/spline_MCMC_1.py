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
import time
import pints.plot
harm_range=list(range(1, 8))
from scipy import interpolate
from scipy.interpolate import CubicSpline
def general_interp(desired_times, given_times, given_data, flag):
                if flag=="basis":
                    tck = interpolate.splrep(given_times, given_data, s=1, k=3) 
                    returned_data = interpolate.BSpline(*tck)(desired_times)
                elif flag=="cubic":
                    cs = CubicSpline(given_times, given_data)
                    returned_data=cs(desired_times)
                elif flag=="linear":
                    returned_data=np.interp(desired_times, given_times, given_data)
                return returned_data
dimensions=10
Ru_vals=np.logspace(0, 3, dimensions)
k0_vals=np.logspace(0, 3, dimensions)
frequencies=[10]
SRS=[400]
true_sf=400
for i in range(0, dimensions):
    for j in range(0, dimensions):
        for k in range(0, len(frequencies)):
            param_list={
                "E_0":0.3,
                'E_start':  0e-3, #(starting dc voltage - V)
                'E_reverse':400e-3,
                'omega':frequencies[k],  #    (frequency Hz)
                "original_omega":frequencies[k] ,
                'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
                'area': 0.07, #(electrode surface area cm^2)
                'Ru': Ru_vals[i],  #     (uncompensated resistance ohms)
                'Cdl':5e-5, #(capacitance parameters)
                'CdlE1': 0.000653657774506*0,
                'CdlE2': 0.000245772700637*0,
                "CdlE3":0,
                'gamma': 5e-11,
                "original_gamma":5e-11,        # (surface coverage per unit area)
                'k_0': k0_vals[j], #(reaction rate s-1)
                'alpha': 0.5,
                "E0_mean":0.2,
                "E0_std": 0.09,
                "cap_phase":3*math.pi/2,
                "alpha_mean":0.5,
                "alpha_std":1e-3,
                'sampling_freq' : (1.0/true_sf),
                'phase' :3*math.pi/2,
                "cap_phase":3*math.pi/2,
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
                "test":False,
                "method": "sinusoidal",
                "phase_only":False,
                "likelihood":likelihood_options[0],
                "numerical_method": solver_list[1],
                "label": "MCMC",
                "top_hat_return":"abs",
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
                'Ru': [0, 2e3],  #     (uncompensated resistance ohms)
                'Cdl': [0,1e-3], #(capacitance parameters)
                'CdlE1': [-0.05,0.15],#0.000653657774506,
                'CdlE2': [-0.01,0.01],#0.000245772700637,
                'CdlE3': [-0.01,0.01],#1.10053945995e-06,
                'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
                'k_0': [1e-3, 2e3], #(reaction rate s-1)
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
            
            noise_vals=0.01
            sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
            current=sim.test_vals([], "timeseries")
            potential=sim.e_nondim(sim.define_voltages(transient=True))
            times=sim.time_vec
            #sampled_current=current[::sampling]
            #sampled_times=times[::sampling]
            #sampled_potential=potential[::sampling]
            error=noise_vals*max(current)
            noisy_current=sim.add_noise(current, error)
            #plt.plot(potential, noisy_current)
            #plt.show()
            #    
            #noisy_sampled_current=noisy_current[::sampling]
            #interpolated_current=general_interp(times, sampled_times, noisy_sampled_current, "linear")
            #data=[interpolated_current, noisy_current]
            sim.simulation_options["label"]="cmaes"
            sim.simulation_options["test"]=False
            sim.secret_data_time_series=noisy_current
            sim.def_optim_list(["E_0", "k_0", "Ru","Cdl",  "gamma", "alpha", "phase"])
            log_params=["k_0", "Ru"]
           
            cmaes_problem=pints.SingleOutputProblem(sim, sim.time_vec[sim.time_idx], noisy_current)
            score = pints.GaussianLogLikelihood(cmaes_problem)
            sigma=error
            lower_bound=np.append(np.zeros(len(sim.optim_list)), [0])
            upper_bound=np.append(np.ones(len(sim.optim_list)), [10*sigma])
            CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
            true_params=[param_list[x] for x in sim.optim_list]
            x0=sim.change_norm_group(true_params, "norm")+[sigma]
            print(x0)
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.05 for x in range(0, sim.n_parameters()+1)], boundaries=CMAES_boundaries, method=pints.CMAES)#,  transformation=MCMC_transform)
            #labels=["interpolated", "noisy"]
            cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
            #cmaes_fitting.set_log_to_screen(False)
            cmaes_fitting.set_parallel(True)
            
            found_parameters, found_value=cmaes_fitting.run()
            dim_params=sim.change_norm_group(found_parameters[:-1], "un_norm")
            print(true_params)
            print(dim_params, found_parameters[-1])
            error=found_parameters[-1]
            
            sim.simulation_options["label"]="MCMC"
            interped=noisy_current
            #interval=true_sf//SRS[m]
            #reduced_data=noisy_current[::interval]
            #interped=reduced_data#general_interp(sim.time_vec, sim.time_vec[::interval], reduced_data, "basis")
            if sim.simulation_options["likelihood"]=="fourier":
                error=error/len(interped)
                interped=sim.top_hat_filter(interped)
                times=np.linspace(0, 1, len(interped))
                
            elif sim.simulation_options["likelihood"]=="timeseries":
                times=sim.time_vec
            MCMC_problem=pints.SingleOutputProblem(sim, sim.time_vec[sim.time_idx], interped)
            updated_lb=[param_bounds[x][0] for x in sim.optim_list]+[0]
            updated_ub=[param_bounds[x][1] for x in sim.optim_list]+[10*error]
            updated_b=[updated_lb, updated_ub]
            updated_b=np.sort(updated_b, axis=0)
            log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
            log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
            log_posterior=pints.LogPosterior(log_liklihood, log_prior)
            mcmc_parameters=np.append(dim_params, error)#[sim.dim_dict[x] for x in sim.optim_list]+[error]
            print(list(mcmc_parameters), len(mcmc_parameters), len(updated_lb), error)
            print(log_liklihood.n_parameters())
            #mcmc_parameters=np.append(mcmc_parameters,error)
            xs=[mcmc_parameters,
                mcmc_parameters,
                mcmc_parameters
                ]
            
            log_params=["k_0", "Ru"]
            #transforms=[pints.IdentityTransformation(n_parameters=1) if x not in log_params else pints.LogTransformation(n_parameters=1) for x in sim.optim_list+["error"]]
            #MCMC_transform=pints.ComposedTransformation(*transforms)
            mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioBardenetACMC)#, transformation=MCMC_transform)

            mcmc.set_parallel(True)
            mcmc.set_max_iterations(10000)
            chains=mcmc.run()
            #pints.plot.trace(chains)
            #print([pints.rhat(chains[:,:,x ]) for x in range(0, len(sim.optim_list)+1)])
            #plt.show()
            save_file="MCMC/parameter_scan/Low_cdl/R_{0}_k_{1}_SR_{2}_10_Hz".format(round(param_list["Ru"],2), round(param_list["k_0"],2),400)
            f=open(save_file, "wb")
            np.save(f, chains)
