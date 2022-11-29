import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
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
param_list={
    "E_0":0.0,
    'E_start':  -5e-3, #(starting dc voltage - V)
    'E_reverse':100e-3,
    'omega':10,#8.88480830076,  #    (frequency Hz)
    "original_omega": 10,
    'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 0.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0.000653657774506*0,
    'CdlE2': 0.000245772700637*0,
    "CdlE3":-1e-6,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 100*0, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2,
    "cap_phase":3*math.pi/2,
    "time_end": None,
    'num_peaks': 5,
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
    "likelihood":likelihood_options[1],
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
    'k_0': [0.1, 1e3], #(reaction rate s-1)
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
from time_domain_simulator_class import time_domain

frequency_powers=np.linspace(2, 7, 20)
frequencies=np.power(10, frequency_powers)
buffer=0.1
for k in [0, 10]:
    pred_impede=np.zeros(len(frequencies), dtype="complex")
    for i in range(0, len(frequencies)):
        param_list["k_0"]=k
        param_list["original_omega"]=frequencies[i]
        param_list["omega"]=frequencies[i]
        simulation_options["no_transient"]=1/frequencies[i]
        sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        current=sim.i_nondim(sim.test_vals([], "timeseries"))
        time=sim.t_nondim(sim.time_vec)
        #current=param_list["Cdl"]*np.cos((2*math.pi*param_list["omega"]*time)+param_list["cap_phase"])
        potential=sim.e_nondim(sim.define_voltages())[sim.time_idx]
        #current=potential/10
        
        fft_freq=np.fft.fftfreq(len(current),time[1]-time[0])
        curr_f=frequencies[i]
        #plt.plot(sim.time_vec, current)
        #plt.plot(time, current2)
        #plt.plot(potential, current)
        #plt.show()
        fft_region=np.where((fft_freq>(curr_f-(curr_f*buffer)))& (fft_freq<(curr_f+(curr_f*buffer))))
        print(fft_region)
        current_fft=np.fft.fft(current)
        potential_fft=np.fft.fft(potential)
        divisor=np.divide(potential, current)
        pred_impede[i]=divisor[fft_region]
        #pred_impede=potential_fft[fft_region]/current_fft[fft_region]
        #impedance[i]=current_fft[fft_region]
        #plt.plot(fft_freq, np.log10(abs(np.divide(potential_fft, current_fft))))
        #plt.axvline(frequencies[i], color="black", linestyle="--")
        #plt.show()
        """ plt.subplot(1,3,1)
        plt.plot(np.log10(fft_freq)[fft_region], current_fft[fft_region].real)
        plt.subplot(1,3,2)
        plt.plot(np.log10(fft_freq)[fft_region], current_fft[fft_region].imag)
        plt.subplot(1,3,3)
        plt.plot(time, current)
        plt.show()"""
    plt.scatter(pred_impede.imag, np.log10(-pred_impede.real))
plt.show()