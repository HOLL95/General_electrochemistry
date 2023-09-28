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
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize

data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Immobilised_Fc/GC-3/Fc/Exported"
file_name="2023-09-13_FTV_GC-3_Fc_gain_2_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")

#sblank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
h_class=harmonics(list(range(1,11)),8.375570778115723, 0.5)
dec_amount=64
volt_data=voltage_data_file[0::dec_amount, 1]


param_list={
    "E_0":-0.3,
    'E_start':  -0.22219524456352024, #(starting dc voltage - V)
    'E_reverse': 0.573,
    'omega':8.375570778115723, #8.88480830076,  #    (frequency Hz)
    "v":0.014891177059531243,
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 7e-11,
    "original_gamma":1e-9,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :0.3,
    "time_end": -1,
    'num_peaks': 30,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[32],
    "GH_quadrature":True,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "experiment_time": current_data_file[0::dec_amount,0],
    "experiment_current": current_data_file[0::dec_amount, 1],
    "experiment_voltage":volt_data,
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.3,0.3],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],2.5*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
}
import copy
copied_other=copy.deepcopy(other_values)
copied_sim=copy.deepcopy(simulation_options)
copied_params=copy.deepcopy(param_list)
ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)
print(ferro.init_freq)




time_results=ferro.other_values["experiment_time"]
current_results=ferro.other_values["experiment_current"]
voltage_results=ferro.other_values["experiment_voltage"]
ferro.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru"])

cpe_both={'E_0': 0.2580873519476527, 'k_0': 400.826795806107176, 'gamma': 1.0819057106697644e-09, 'Cdl': 0.00014209665648665912/param_list["area"], 'alpha': 0.4927399067944382, 'Ru': 116.65937541075522, 'cpe_alpha_cdl': 0.6018831645359057, 'cpe_alpha_faradaic': 0.8387369804518475}
vals=[cpe_both[x] for x in ferro.optim_list]

sim=ferro.i_nondim(ferro.test_vals(vals, "timeseries"))
#plt.plot(sim)
plot_args=dict(EIS_Cdl_time_series=sim, hanning=True, plot_func=abs)
for cdl_val in np.flip([1e-5, 5e-5, 1e-4]):
    cpe_both["Cdl"]=cdl_val
    vals=[cpe_both[x] for x in ferro.optim_list]
    sim=ferro.test_vals(vals, "timeseries")
    key="Cdl={0}_time_series".format(cdl_val)
    plot_args[key]=ferro.i_nondim(sim)
    #plt.plot(plot_args[key])
h_class.plot_harmonics(ferro.t_nondim(time_results), **plot_args)
#h_class.plot_harmonics(ferro.t_nondim(time_results), current_time_series=current_results,simulated_time_series=sim, hanning=True, plot_func=abs)
plt.show()

"""ferro.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])

vals=[-0.3235422429210039, 0.03, 96.66237266990319, 49.4141315707467, 9.999700576833107e-05, -0.1498437469686061*0, -0.007907806379936225*0, -0.00022932119418286184*0, 1e-11, 9.015052044193897, 5.53225110499901, 5.273923188038943, 0.4000000312109022]
vals[ferro.optim_list.index("omega")]=param_list["omega"]


plt.plot(time_results, current_results)
plt.plot(time_results, sim)
plt.show()
"""