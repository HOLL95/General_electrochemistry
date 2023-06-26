import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import copy
import math
dir=os.getcwd()
dir_list=dir.split("/")
print(dir_list)
src_index=[x for x in range(0, len(dir_list)) if dir_list[x]=="General_electrochemistry"][0]
sys.path.append(("/").join(dir_list[:src_index+1]+["src"]))
from EIS_class import EIS
from Time_domain_EIS import Time_domain_EIS
def potential(amp,frequency, time, phase):
    return amp*np.sin(2*np.pi*frequency*time+phase)
def current(cdl, amp, frequency, time, phase):
    return (cdl)*frequency*amp*np.cos(2*np.pi*frequency*time+phase)

param_list={
        "E_0":0.001,
        'E_start':  -5e-3, #(starting dc voltage - V)
        'E_reverse':5e-3,
        'omega':10,  #    (frequency Hz)
        "original_omega":10,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl':1e-3, #(capacitance parameters)
        'CdlE1': 0.000653657774506*0,
        'CdlE2': 0.000245772700637*0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 10, #(reaction rate s-1)
        'alpha': 0.55,
        "E0_mean":0.2,
        "E0_std": 0,
    
        "alpha_mean":0.45,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/2**8),
        'phase' :0,
        "cap_phase":0,
        "time_end": None,
        'num_peaks': 5,
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
#time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test":False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "top_hat_return":"composite",
    "optim_list":[],
    "threshold":0.5,
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(2, 7)),
    "bounds_val":20000,
    
}
param_bounds={
    'E_0':[-10e-3, 10e-3],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 2e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
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
sim_class=Time_domain_EIS(param_list, simulation_options,other_values, param_bounds)
min_f=0
max_f=8
points_per_decade=10
frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
freqs=[10**x for x in frequency_powers]
m,p, z,f=sim_class.td_simulate([], freqs)
equiv_circ=EIS(circuit={ "z1":"C1"})
zsim=equiv_circ.test_vals({"C1":param_list["Cdl"]}, f)
z.real[np.where(z.real<1e-3)]=0
print(z.real)
print(z.imag)
plt.scatter(z.real, -z.imag)
#plt.scatter(zsim[:,0], -zsim[:,1])
plt.show()