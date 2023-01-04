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
harm_range=list(range(4, 6))
from scipy import interpolate
from scipy.interpolate import CubicSpline
param_list={
    "E_0":0.2,
    'E_start':  -200e-3, #(starting dc voltage - V)
    'E_reverse':400e-3,
    'omega':100,  #    (frequency Hz)
    "original_omega":100 ,
    'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 7e-5, #(capacitance parameters)
    'CdlE1': 0.000653657774506,
    'CdlE2': 0.000245772700637,
    "CdlE3":-1e-6,
    'gamma': 2e-11,
    "original_gamma":2e-11,        # (surface coverage per unit area)
    'k_0': 10000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 2,
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
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
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
    'k_0': [0.1, 1e4], #(reaction rate s-1)
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

sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
current=sim.test_vals([], "timeseries")
potential=sim.e_nondim(sim.define_voltages(transient=True))
end_time=param_list["num_peaks"]/param_list["omega"]
num_bio_points=end_time*5000
sampling=int((param_list["num_peaks"]/param_list["sampling_freq"])/num_bio_points)
times=sim.t_nondim(sim.time_vec)
sampled_current=current[::sampling]
sampled_times=times[::sampling]
sampled_potential=potential[::sampling]
print(len(sampled_current), len(current))
fig, ax=plt.subplots(2,2)
noise_vals=[0.005, 0.01, 0.02, 0.05]
residuals=True
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
methods=[ "cubic", "linear","basis"]
for i in range(0, len(noise_vals)):
    axis=ax[i//2, i%2]
    noisy_sampled_current=sim.add_noise(sampled_current, noise_vals[i]*max(current))
    
    if residuals==True:
        for method in methods:
            start=time.time()
            interpolated_current=general_interp(times, sampled_times, noisy_sampled_current, method)
            residual=np.subtract(current, interpolated_current)
            print(method, time.time()-start)
            axis.plot(times, residual, label=method)
        #axis.plot(times, current)
    else:    
        axis.plot(potential, current, color="red")
        axis.scatter(potential, general_interp(times, sampled_times, noisy_sampled_current, method))
        axis.scatter(sampled_potential, noisy_sampled_current,  marker="x")
    axis.set_title("{0}%".format(noise_vals[i]*100))
if residuals==True:
    ax[0,0].legend()
plt.show()