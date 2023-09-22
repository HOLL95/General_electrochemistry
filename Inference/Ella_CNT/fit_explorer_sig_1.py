import matplotlib.pyplot as plt
import math
import os
import sys
import re
import copy
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from heuristic_class import DCVTrumpet, DCV_peak_area
import numpy as np
import pints

file_loc="/home/henney/Documents/Oxford/Experimental_data/Ella/MWCNT_scan_sig1"

param_list={
    "E_0":0,
    'E_start': -0.5, #(starting dc voltage - V)
    'E_reverse':0.2,
    'omega':0, #8.88480830076,  #    (frequency Hz)
    'd_E': 0,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    "v":22.5e-3,
    'Ru': 0,  #     (uncompensated resistance ohms)
    'Cdl': 0, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 75, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/50),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 10,
    "k0_shape":0.4,
    "k0_scale":75,
    "dcv_sep":0,
    
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[32],
    "GH_quadrature":True,
    "test": False,
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "invert_imaginary":False,
    "Marcus_kinetics":False,
    "optim_list":[],
    "invert_positions":True
    
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),

    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.3, 0],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.2,0.2],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [1, 100], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "dcv_sep":[0, 0.2]

}
alpha=True
trumpets=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
files=os.listdir(file_loc)
file_dict={}
for file in files:
    get_number=re.match("[0-9]+(?=\smV)", file).group(0)
    key=int(get_number)
    data=np.loadtxt(file_loc+"/"+file, skiprows=1)
    file_dict[key]=data
    #time=data[:,0]
    #potential=data[:,1]
    #current=data[:,2]
    #plt.plot(potential, current)
    #plt.show()
first_regime_end=1000
second_regime_end=6000
key_list=sorted(list(file_dict.keys()))
idx=key_list.index(second_regime_end)

trumpet_positions=np.zeros((len(key_list),2))
for i in range(0, len(key_list)):
    number=key_list[i]
    data=file_dict[number]
    time=data[:,0]
    potential=-1*data[:,1]
    current=data[:,2]
    #plt.plot(potential, current)
    # plt.show()
    if number<first_regime_end:
        bg=DCV_peak_area(time, potential, current, 0.07, func_order="3")
        subtract=bg.background_subtract([-0.3,0, -0.3, 0,min(potential)+0.05, max(potential)-0.05, min(potential)+0.05, max(potential)-0.05])
    elif number<second_regime_end:
        bg=DCV_peak_area(time, potential, current, 0.07, func_order="4")
        subtract=bg.background_subtract([-0.2,0.1, -0.4, -0.1,min(potential)+0.05, max(potential)-0.05, min(potential)+0.05, max(potential)-0.05])
    else:
        bg=DCV_peak_area(time, potential, current, 0.07, func_order="4")
        subtract=bg.background_subtract([-0.1,0.3, -0.6, -0.25,-0.4, max(potential)-0.05, -0.65, 0.1])
    #trumpets.simulation_options["find_in_range"]=[0.05, 0.3]
    
    #
    
    
    full_subtract_potential=np.append(subtract["subtract_0"][0],subtract["subtract_1"][0])
    full_subtract_current=np.append(subtract["subtract_0"][1],subtract["subtract_1"][1])
    trumpet_pos=trumpets.trumpet_positions(full_subtract_current, full_subtract_potential, dim_flag=False)
    
    #plt.title(number)
    

    """ fig, ax=plt.subplots(1,2)
    ax[0].set_title(number)
    ax[0].plot(potential, current)
    ax[0].axvline(trumpet_pos[0])
    ax[0].axvline(trumpet_pos[1])
    for i in range(0, 2):
        ax[1].plot(subtract["subtract_{0}".format(i)][0],subtract["subtract_{0}".format(i)][1])
        #ax[0].plot(subtract["poly_{0}".format(i)][0],subtract["poly_{0}".format(i)][1])
        #ax[0].plot(subtract["bg_{0}".format(i)][0],subtract["bg_{0}".format(i)][1])
        ax[0].plot(subtract["subtract_{0}".format(i)][0],subtract["subtract_{0}".format(i)][1])
    ax[1].axvline(trumpet_pos[0])
    ax[1].axvline(trumpet_pos[1])
    plt.show()"""

    trumpet_positions[i,:]=[trumpet_pos[0][0],trumpet_pos[1][0] ]
in_volts=np.array(key_list)*1e-3
trumpet_positions=trumpet_positions#/trumpets.nd_param.c_E0
trumpets.secret_data_trumpet=trumpet_positions
trumpets.def_optim_list(["E_0", "k_0", "alpha", "dcv_sep"])
sim=trumpets.simulate([-0.16133834 , 4.42019328,  0.59933988,  0.0199163 ], in_volts, optimise_flag=True)
if alpha==True:
    trumpets.def_optim_list(["E_0", "k_0","dcv_sep"])
    trumpets.dim_dict["alpha"]=0.5
from MCMC_plotting import MCMC_plotting
mplot=MCMC_plotting(burn=5000)

chains=np.load("sig_1_MCMC_alpha")
params=["E_0", "k_0", "dcv_sep"]#, "sigma_1", "sigma_2"]
units=mplot.get_units(params)
titles=mplot.get_titles(params, units=False)

fig, ax=plt.subplots(2, 3)
mplot.plot_params(params,chains, axes=ax[0,:])
appended_chain=mplot.concatenate_all_chains(chains)
#print(np.mean(np.mean(chains, axis=1), axis=0))
default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(units, titles)
core_params=chains[0,0,:len(params)]
desired_quantiles=[0.05, 0.5, 0.95]
for i in range(0, len(params)):
    trumpets.trumpet_plot(key_list, trumpet_positions, ax=ax[1,i], label="Data")
    sim_params=copy.deepcopy(core_params)
    for j in range(0, len(desired_quantiles)):
        current_quantile=np.quantile(appended_chain[i], desired_quantiles[j])
        
        ax[0,i].axvline(current_quantile, linestyle="--", color=default_color_cycle[j+1])
        
        sim_params[i]=current_quantile
        sim=trumpets.e_nondim(trumpets.simulate(sim_params, in_volts, optimise_flag=True))
        ax[1,i].plot(np.log10(key_list), sim[:,0], color=default_color_cycle[j+1], label=titles[i]+"="+mplot.format_values(current_quantile)+" "+units[params[i]])
        ax[1,i].plot(np.log10(key_list), sim[:,1], color=default_color_cycle[j+1])
        ax[1,i].set_xlabel("Log(scan rate (mV s$^{-1}$))")
        ax[1,i].set_ylabel("Peak position (V)")
    ax[1, i].legend()
plt.show()
