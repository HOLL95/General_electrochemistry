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
likelihood_dim=1000
SRS=[400]
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
param_names=["E_0", "k_0", "Ru","Cdl",  "gamma", "alpha", "phase"]
results_1=np.load("Likelihoods/Low_cdl_profile_likelihoods_2.npy", allow_pickle=True).item()
results_2=np.load("Likelihoods/Low_cdl_profile_likelihoods.npy", allow_pickle=True).item()
chosen_param="Ru"
r_loc=param_names.index(chosen_param)
true_sf=400
s1=time.time()
results_list=np.zeros((likelihood_dim, dimensions, dimensions))
param_values_list=np.zeros((likelihood_dim, dimensions, dimensions))
results_dict={"errors":results_list, "values":param_values_list}
fig, ax=plt.subplots(4, 5)
for i in range(0, dimensions):
    if i<11:
        results=results_1
    else:
        results=results_2
    #fig=plt.figure()
    #ax=fig.add_subplot()
    print(i)
    for j in range(0, 10):
       
     
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
        axes=ax[i//5, i%5]
        #print(i//5)
        #axes=ax
        curr_results=results["errors"][:,i,j]
        min_res=min(curr_results)
        max_res=max(curr_results)
        normed_results=[sim.normalise(x, [min_res, max_res]) for x in curr_results]
        param=results["inferred_results"][r_loc,i,j]
        plot_val=np.interp([param, param_list["Ru"]], results["values"][:,i,j],curr_results)
        axes.scatter(param, plot_val[0], s=20, color=colors[j%len(colors)])
        axes.scatter(param_list["Ru"], plot_val[1], s=20, color=colors[j%len(colors)], marker="*")
        axes.loglog(results["values"][:,i,j],curr_results, label=round(param_list["Ru"], 1))
        if i%5==0:
            axes.set_ylabel("Likelihood")
        if i//5==3:
            axes.set_xlabel("$R_u(\\Omega)$")
        #axes.set_title(param_list["Ru"])
        #axes.scatter(np.log10(param_list["Ru"]), 0,  marker="*")
        #axes.axvline(param_list["Ru"], color="black", linestyle="--")
        #axes.plot(np.log10(results["values"][:,i,j]), np.log10(curr_results),zs=param_list["Ru"], zdir="y")
        #plt.tight_layout()
ax[0, 2].legend(ncols=5, bbox_to_anchor=[0.5, 1.7], loc="upper center", frameon=False)
fig.set_size_inches(14, 9)
plt.tight_layout()
fig.subplots_adjust(top=0.895, wspace=0.368)

plt.show()