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
dimensions=20
Ru_vals=np.logspace(0, 3.25, dimensions)
k0_vals=np.logspace(0, 3.25, dimensions)
print(Ru_vals)
frequencies=[10]
likelihood_dim=2000
SRS=[400]
param_names=["E_0", "k_0", "Ru","Cdl",  "gamma", "alpha", "phase"]

chosen_param="Ru"
r_loc=param_names.index(chosen_param)
true_sf=400
s1=time.time()
results_list=np.zeros((likelihood_dim, dimensions, dimensions))
param_values_list=np.zeros((likelihood_dim, dimensions, dimensions))
found_params=np.zeros((len(param_names), dimensions, dimensions))
results_dict={"errors":results_list, "values":param_values_list, "inferred_results":found_params}
for i in range(0   , dimensions):
    #fig, ax=plt.subplots(3, 5)
    for j in range(0, dimensions):
        
       
        param_list={
            "E_0":0.3,
            'E_start':  0e-3, #(starting dc voltage - V)
            'E_reverse':400e-3,
            'omega':frequencies[0],  #    (frequency Hz)
            "original_omega":frequencies[0] ,
            'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
            'area': 0.07, #(electrode surface area cm^2)
            'Ru': Ru_vals[j],  #     (uncompensated resistance ohms)
            'Cdl':5e-5, #(capacitance parameters)
            'CdlE1': 0.000653657774506*0,
            'CdlE2': 0.000245772700637*0,
            "CdlE3":0,
            'gamma': 5e-11,
            "original_gamma":5e-11,        # (surface coverage per unit area)
            'k_0': k0_vals[i], #(reaction rate s-1)
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
            "no_transient":False,
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
            'Ru': [1e-2, 2e3],  #     (uncompensated resistance ohms)
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
        error=noise_vals*max(current)
        noisy_current=sim.add_noise(current, error)
        sim.def_optim_list(param_names)  
        ref_params=[param_list[x] for x in sim.optim_list]
        param_mat=np.array([ref_params for x in range(0, likelihood_dim)])
        sim.simulation_options["label"]="cmaes"
        sim.simulation_options["test"]=False
        cmaes_problem=pints.SingleOutputProblem(sim,sim.time_vec, noisy_current)
        score = pints.GaussianLogLikelihood(cmaes_problem)
        sigma=error
        lower_bound=np.append(np.zeros(len(sim.optim_list)), [0])
        upper_bound=np.append(np.ones(len(sim.optim_list)), [10*sigma])
        CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
        true_params=[param_list[x] for x in sim.optim_list]
        print(true_params)
    
        num_runs=10
        z=-1
        found=False
        params=np.zeros((num_runs, sim.n_parameters()+1))
        while found==False:
            x0=np.random.rand(len(true_params)+1)#sim.change_norm_group(true_params, "norm")+[sigma]
            x0[-1]=sim.un_normalise(x0[-1], [0, 10*sigma])
            z+=1
            
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.25 for x in range(0, sim.n_parameters()+1)], boundaries=CMAES_boundaries, method=pints.CMAES)
            cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
            cmaes_fitting.set_parallel(True)
            found_parameters, found_value=cmaes_fitting.run()
            dim_params=sim.change_norm_group(found_parameters[:-1], "un_norm")
            params[z, :-1]=dim_params
            params[z, -1]=found_parameters[-1]
            error=found_parameters[-1]
            if found_value>-5300:
                error_loc=z
                found=True
            elif z==num_runs-1:
                cmaes_errors=params[:, -1]
                error_loc=np.where(cmaes_errors==max(cmaes_errors))[0][0]
                found=True
            
        dim_params=params[error_loc, :-1]
        print(dim_params)
        distance=1.1*abs(dim_params[r_loc]-param_list[chosen_param])
        if 100*(np.abs(dim_params[r_loc]-param_list[chosen_param])/param_list[chosen_param])>50:
            minima=max(param_list[chosen_param]-distance, 1e-2)
            maxima= param_list[chosen_param]+distance#param_list[chosen_param]+(param_list[chosen_param]-minima)
            r_range=np.logspace(np.log10(minima), np.log10(maxima), likelihood_dim)
        elif 100*(np.abs(dim_params[r_loc]-param_list[chosen_param])/param_list[chosen_param])>0.1:
            r_range=np.linspace(param_list[chosen_param]-distance, param_list[chosen_param]+distance, likelihood_dim)
        else:
            r_range=np.linspace(0.8*param_list[chosen_param], 1.2*param_list[chosen_param], likelihood_dim)
        if np.any(np.isnan(r_range))==True:
            print(distance, dim_params[r_loc], param_list[chosen_param])
            raise ValueError("Still not working")
        print(r_range[::100])
        param_mat[:,r_loc]=r_range
        pot=sim.define_voltages()
        start=time.time()
        ts_array=sim.matrix_simulate(param_mat)
        print(time.time()-start)
        errors=[sim.RMSE(x, noisy_current) for x in ts_array]
        results_dict["errors"][:, i, j]=errors
        results_dict["values"][:, i, j]=r_range
        results_dict["inferred_results"][:, i, j]=dim_params
        np.save("Likelihoods/Low_cdl_profile_likelihoods_symmetrical_2", results_dict)       