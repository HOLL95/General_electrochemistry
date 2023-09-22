
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
from EIS_TD import EIS_TD

param_list={
    "E_0":-0.2,
    'E_start':  -0.5, #(starting dc voltage - V)
    'E_reverse':0.3,
    'omega':10, #8.88480830076,  #    (frequency Hz)
    "original_omega":10,
    'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 5e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 2e-11,
    "original_gamma":2e-11,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :0,
    "time_end": -1,
    "Upper_lambda":0.64,
    'num_peaks': 10,
    
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["original_omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "GH_quadrature":True,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "Marcus_kinetics":True,
    "optim_list":[],
    
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),

    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.2,0.2],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
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
    "Upper_lambda":[0.02, 10],
}
import copy
import time
mode="PSV"
#mode="EIS"
if mode=="PSV":
    cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    k0_plot_vals=[0.5, 10, 25,150]

    lambda_vals=[0.1,0.25, 0.4, 0.65]
    cyt.def_optim_list(["Upper_lambda", "k_0"])
    row=3
    col=2
    fig, ax=plt.subplots(row, col)
    for i in range(0, row):
        for j in range(0, col):
            ax[i,j].set_xlabel("Potential (V)")
            ax[i,j].set_ylabel("Current (A)")
    
    lambda_ax=ax[0,1]
    k0_var_ax=ax[0,0]
    lambda_k=50
    k_lambda=0.2
    potential=cyt.e_nondim(cyt.define_voltages()[cyt.time_idx])
    for j in range(0, len(lambda_vals)):
        current=cyt.i_nondim(cyt.test_vals([lambda_vals[j], lambda_k], "timeseries"))
        lambda_ax.plot(potential, current, label="$\\lambda=$"+str(lambda_vals[j]))
    lambda_ax.legend(loc="lower right")
    
    for i in range(0, len(k0_plot_vals)):
        cyt.def_optim_list(["Upper_lambda", "k_0"])
        cyt.simulation_options["Marcus_kinetics"]=True
        current=cyt.i_nondim(cyt.test_vals([k_lambda, k0_plot_vals[i]], "timeseries"))
        k0_var_ax.plot(potential, current,  label="$k^0=$"+str(k0_plot_vals[i]))
        bv_ax=ax[i//2+1, i%2]
        bv_ax.set_title(label="$k^0=$"+str(k0_plot_vals[i]))
        bv_ax.plot(potential, current, label="MHC")
        cyt.def_optim_list(["k_0"])
        cyt.simulation_options["Marcus_kinetics"]=False
        current=cyt.i_nondim(cyt.test_vals([k0_plot_vals[i]], "timeseries"))
        
        bv_ax.plot(potential, current,  label="BV")
        if i==0:
            bv_ax.legend()    
    k0_var_ax.legend(loc="lower right")
 
elif mode=="EIS":
    param_list["E_0"]=0
    param_list["d_E"]=10e-3
    param_list["E_start"]=param_list["E_0"]-param_list["d_E"]
    simulation_options["no_transient"]=False
    simulation_options["data_representation"]="bode"
    td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
    k0_plot_vals=[0.5, 10, 25,150]

    lambda_vals=[0.1,0.25, 0.4, 0.65]
    td.def_optim_list(["Upper_lambda", "k_0"])
    fig, ax=plt.subplots(3, 2)
    lambda_ax=ax[0,1]
    lambda_twinx=lambda_ax.twinx()
    k0_var_ax=ax[0,0]
    k0_twinx=k0_var_ax.twinx()
    lambda_k=50
    k_lambda=0.2
    frequencies=td.define_frequencies(-2,6, points_per_decade=10)
    for j in range(0, len(lambda_vals)):
        print(j)
        bodes=td.simulate([lambda_vals[j], lambda_k],frequencies)
        EIS().bode(bodes, frequencies, ax=lambda_ax, twinx=lambda_twinx, label="$\\lambda=$"+str(lambda_vals[j]), data_type="phase_mag", compact_labels=True)
    lambda_ax.legend()
    for i in range(0, len(k0_plot_vals)):
        print(i)
        td.def_optim_list(["Upper_lambda", "k_0"])
        td.simulation_options["Marcus_kinetics"]=True
        bodes=td.simulate([k_lambda, k0_plot_vals[i]],frequencies)
        EIS().bode(bodes, frequencies, ax=k0_var_ax, twinx=k0_twinx, label="$k^0=$"+str(k0_plot_vals[i]), data_type="phase_mag", compact_labels=True)
       
        bv_ax=ax[i//2+1, i%2]
        bv_twinx=bv_ax.twinx()
        bv_ax.set_title(label="$k^0=$"+str(k0_plot_vals[i]))
        EIS().bode(bodes, frequencies, ax=bv_ax, twinx=bv_twinx, label="MHC", data_type="phase_mag", compact_labels=True) 
        td.def_optim_list(["k_0"])
        td.simulation_options["Marcus_kinetics"]=False
       
        bodes=td.simulate([k0_plot_vals[i]],frequencies)
        EIS().bode(bodes, frequencies, ax=bv_ax, twinx=bv_twinx, label="BV", data_type="phase_mag", compact_labels=True) 
        if i==0:
            bv_ax.legend()
    k0_var_ax.legend(loc="lower right")
plt.subplots_adjust(top=0.96,
bottom=0.11,
left=0.1,
right=0.9,
hspace=0.42,
wspace=0.425)

fig.set_size_inches(7,8)
#plt.show()
fig.savefig("Marcus_{0}_param_scans.png".format(mode), dpi=500)