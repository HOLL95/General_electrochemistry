import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
print(sys.path)
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
harm_range=list(range(4, 8))
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
dimensions=15
Ru_vals=np.logspace(0, 2.5, dimensions)
k0_vals=np.logspace(0, 2.5, dimensions)
frequencies=[10]
SRS=[400]
true_sf=400
def get_even_amplitudes(time, current):
    f=np.fft.fftfreq(len(time), time[1]-time[0])
    Y=np.fft.fft(current)
    abs_Y=abs(Y)
    get_primary_harm=abs(max(f[np.where(abs_Y==max(abs_Y))]))
    box_width=0.05
    even_harms=np.multiply(range(1, 7), 2)
    return_arg=np.zeros(len(even_harms))
    for i in range(0, len(even_harms)):
        return_arg[i]=max(abs_Y[np.where((f>even_harms[i]*(1-box_width)) &(f<even_harms[i]*(1+box_width)))])
    return return_arg

E_start_vals=np.linspace(-50e-3, 50e-3, 100)
even_harms=np.zeros((len(E_start_vals), 6))
#fig, ax=plt.subplots(2, 3)
gamma_vals=np.linspace(3e-11, 7e-11, dimensions)
CdlE3_vals=np.linspace(-1e-6, 1e-6, dimensions)
phase_vals=np.linspace(5*math.pi/4, 7*math.pi/4)
alpha_vals=np.linspace(0.4, 0.6, dimensions)
minima=np.zeros((dimensions, dimensions))
min_potential=np.zeros(len(E_start_vals))
std_vals=np.linspace(0.01, 0.05, dimensions)
#fig=plt.figure()
#fig.add_subplot(projection="3d")
#plt.show()
for j in range(0, dimensions):
    for k in range(0, dimensions):
        for i in range(0, len(E_start_vals)):
                    param_list={
                        "E_0":0.3,
                        'E_start':  E_start_vals[i], #(starting dc voltage - V)
                        'E_reverse':400e-3,
                        'omega':frequencies[0],  #    (frequency Hz)
                        "original_omega":frequencies[0] ,
                        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
                        'area': 0.07, #(electrode surface area cm^2)
                        'Ru': Ru_vals[2],  #     (uncompensated resistance ohms)
                        'Cdl':5e-5, #(capacitance parameters)
                        'CdlE1': 0.000653657774506,
                        'CdlE2': 0.000245772700637,
                        "CdlE3":0,
                        'gamma': gamma_vals[-1],
                        "original_gamma":5e-11,        # (surface coverage per unit area)
                        'k_0': 100, #(reaction rate s-1)
                        'alpha': alpha_vals[k],
                        "E0_mean":0.2,
                        "E0_std": std_vals[j],
                        "cap_phase":3*math.pi/2,
                        "alpha_mean":0.45,
                        "alpha_std":1e-3,
                        'sampling_freq' : (1.0/true_sf),
                        'phase' :3*math.pi/2,
                        "cap_phase":phase_vals[2],
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
                        "test":False,
                        "method": "sinusoidal",
                        "phase_only":False,
                        "likelihood":likelihood_options[1],
                        "numerical_method": solver_list[1],
                        "label": "MCMC",
                        "top_hat_return":"composite",
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
                    
                    noise_vals=0.01
                    sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
                    sim.def_optim_list(["E0_mean", "E0_std"])
                    current=sim.test_vals([param_list["E_0"], param_list["E0_std"]], "timeseries")
                    #current=sim.test_vals([], "timeseries")
                    potential=sim.e_nondim(sim.define_voltages(transient=True))
                    times=sim.time_vec[sim.time_idx]
                    min_potential[i]=get_even_amplitudes(times, current)[2]
        
        print(E_start_vals[np.where(min_potential==min(min_potential))])
        minima[j,k]=E_start_vals[np.where(min_potential==min(min_potential))]
results_dict={"minima":minima, "vals":{"k_0":k0_vals, "alpha":alpha_vals}}
np.save("Minima_location_std.npy", results_dict)
for i in range(0, dimensions):
    plt.plot(alpha_vals, minima[i, :], label=k0_vals[i])
plt.legend()
plt.show()    
 

            