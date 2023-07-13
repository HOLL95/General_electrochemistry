
import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_class import EIS
from EIS_optimiser import EIS_genetics
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
import numpy as np
import pints
import copy
from pints.plot import trace
from MCMC_plotting import MCMC_plotting
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"
mplot=MCMC_plotting()
data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=0
frequencies=np.flip(data[:,2])
param_list={
       "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':1,
        "original_omega":1,  #    (frequency Hz)
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 2e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "k0_scale":100, 
        "k0_shape":0.1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "E0_mean":DC_val,
        "E0_std":0.02,
        "cap_phase":0,
        "num_peaks":5,
        "Cdl_std":5e-6,
        "Cdl_mean":2e-5,
        "Cdl_skew":0,
        "phase_std":1,
        "cap_phase_std":1,
        "Ru_std":10
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
    "dispersion_test":False,
    "method": "sinusoidal",
    "phase_only":True,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
 
    "data_representation":"bode",
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'E0_mean':[-0.4, -0.1],
    'E0_std':[1e-3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[0, 2*math.pi],
    "cap_phase":[0, 2*math.pi],
    "k0_shape":[0,100],
    "k0_scale":[0,2],
    
}
for key in param_list.keys():
    if key not in param_bounds:
        param_bounds[key]=[0.1*param_list[key], 10*param_list[key]]
td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
freqs=td.define_frequencies(-1,6)

shape_vals=[1, 10, 100]
cdl_vals=[0, 0.5, 1]
shape_vals=cdl_vals
core_params=["E_0","gamma","k_0","alpha"]


change_params=["Cdl", "Ru", "phase"]
pure_params=["E_0","gamma","k_0" , "Cdl", "alpha", "Ru", "phase", "cap_phase"]
td.def_optim_list(pure_params)
vals=[DC_val,1e-10, 10,2e-5, 0.55, 250, 0,0]
print(vals)
nodisp=td.simulate(vals, freqs)
fig, ax=plt.subplots(len(change_params), 2)
val_range={
            "Cdl":{"param":"skew", "range":[0, 0.75, 1.5]},
            "Ru":{"param":"std", "range":[10, 25, 50]},
            "phase":{"param":"skew", "range":[0,0.5,1.5]},
}
for j in range(0, len(change_params)):
    key=change_params[j]
    
    current_optim_list=copy.deepcopy(core_params)
    copy_params=copy.deepcopy(param_list)
    for q in range(0, len(change_params)):
        if change_params[q] == key:
            distribution_params=[key+x for x in ["_std", "_mean", "_skew"]]
            current_optim_list+=distribution_params
            copy_params[key+"_mean"]=param_list[key]
            copy_params[key+"_skew"]=0    
            
        else:
            current_optim_list+=[change_params[q]]
    
    for missing_key in copy_params.keys():
        if missing_key not in param_bounds and "_array" not in missing_key:
            
            param_bounds[missing_key]=[0.1*copy_params[missing_key], 10*copy_params[missing_key]]
    td=EIS_TD(copy_params, simulation_options, other_values, param_bounds)
    td.def_optim_list(current_optim_list)       
    disp_ax=ax[j,0]
    
    plot_ax=ax[j,1]
    plot_twinx=plot_ax.twinx()
    EIS().bode(nodisp, freqs, ax=plot_ax, twinx=plot_twinx, label="Nodisp", data_type="phase_mag", compact_labels=True)
    units=mplot.get_units(current_optim_list)
    full_titles=dict(zip(current_optim_list, mplot.get_titles(current_optim_list)))
    half_titles=dict(zip(current_optim_list, mplot.get_titles(current_optim_list, units=False)))
    #print(units)
    #print(full_titles)
    #print(half_titles)
    print(key)
    vary_param=key+"_"+val_range[key]["param"]
    disp_ax.set_xlabel(full_titles[vary_param])
    disp_ax.set_ylabel("f({0})".format(half_titles[vary_param]))
    for i in range(0, len(val_range[key]["range"])):
        
        copy_params[vary_param]=val_range[key]["range"][i]
        

        sim_vals=[copy_params[x] for x in current_optim_list]
        #print(sim_vals)
        #print(current_optim_list)
        sim=td.simulate(sim_vals, freqs)
        td.update_params(sim_vals)
        values, weights=td.return_distributions(500)
        disp_ax.plot(values, weights)
        label="{0}={1} {2}".format(half_titles[vary_param], mplot.format_values(val_range[key]["range"][i]), units[vary_param])
        EIS().bode(sim, freqs, ax=plot_ax, twinx=plot_twinx, label=label, data_type="phase_mag", compact_labels=True)
    
plt.show()


