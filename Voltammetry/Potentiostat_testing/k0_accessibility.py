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
from input_design import Input_optimiser
import numpy as np
import matplotlib.pyplot as plt
import sys
harm_range=list(range(1, 50))
from scipy import interpolate
from SALib.sample import saltelli
from SALib.analyze import sobol
import cma
k_vals=np.arange(1, 3000, 1)
max_accessible_harm=np.zeros(len(k_vals))
min_informative_harm=np.zeros(len(k_vals))
np_harm_range=np.array(harm_range)
max_diff_val=np.zeros(len(k_vals))
z=np.load("k_access.npy")
print(z)
plot_k=z[0,:]
log_k=np.log10(plot_k)
plot_mah=z[1,:]
plot_mih=z[2,:]
pc_diff=z[3,:]
plt.subplot(1,2,1)
plt.scatter(log_k, plot_mah, label="Maximum accessible harmonic")
plt.scatter(log_k[np.where(plot_mih<100)], plot_mih[np.where(plot_mih<100)], label="Minimum information harmonic")
plt.xlabel("Log10(K)")
plt.ylabel("Harmonic")
plt.legend()
plt.subplot(1,2,2)
plt.semilogy(log_k, pc_diff)
plt.xlabel("Log10(K)")
plt.ylabel("Percentage error")
plt.show()


print(z)
for j in range(0, len(k_vals)):
    param_list={
        "E_0":-0.3,
        'E_start':  -600e-3, #(starting dc voltage - V)
        'E_reverse':-100e-3,
        'omega':8.88480830076,  #    (frequency Hz)
        "v":50e-3,
        'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100.0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-4, #(capacitance parameters)
        'CdlE1': 0.000653657774506,
        'CdlE2': 0.000245772700637,
        "CdlE3":-1e-6,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': k_vals[j], #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/200),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 5,
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
        "method": "ramped",
        "phase_only":False,
        "likelihood":likelihood_options[0],
        "numerical_method": solver_list[1],
        "label": "MCMC",
        "optim_list":[]
    }
    other_values={
        "filter_val": 0.5,
        "harmonic_range":harm_range,
        "bounds_val":2000,
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
        'phase' : [0, 2*math.pi],
        "all_freqs":[1e-3, 100],
        "all_amps":[1e-5, 0.5],
        "all_phases":[0, 2*math.pi],
    }

    sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    rpotential=sim.e_nondim(sim.define_voltages())
    sim.def_optim_list(["k_0"])
    t=sim.t_nondim(sim.time_vec)
    h_class=harmonics(harm_range,param_list["omega"], 0.1)


    rcurrent=sim.test_vals([param_list["k_0"]], "timeseries")
    clean_rcurrent=copy.deepcopy(rcurrent)
    clean_orig_harms=h_class.generate_harmonics(t, rcurrent, hanning=True)
    shifted_current=sim.test_vals([param_list["k_0"]+30], "timeseries")
    noise=0.01
    rcurrent=sim.add_noise(rcurrent, noise*max(rcurrent))
    #shifted_current=sim.add_noise(shifted_current, noise*max(shifted_current))


    orig_harms=h_class.generate_harmonics(t, rcurrent, hanning=True)
    shifted_harms=h_class.generate_harmonics(t, shifted_current, hanning=True)

    differences=np.zeros((2,len(harm_range)))
    for i in range(0, len(harm_range)):
        differences[0,i]=100*sim.RMSE(abs(shifted_harms[i,:]), abs(clean_orig_harms[i,:]))/max(abs(clean_orig_harms[i,:]))
        differences[1,i]=100*sim.RMSE(abs(orig_harms[i,:]), abs(clean_orig_harms[i,:]))/max(abs(clean_orig_harms[i,:]))
    #print(list(differences))
    #h_class.plot_harmonics(t, init_time_series=rcurrent, shifted_time_series=clean_rcurrent, hanning=True, plot_func=abs)
    #h_class.plot_harmonics(t, init_time_series=shifted_current, shifted_time_series=clean_rcurrent, hanning=True, plot_func=abs)
    #plt.show()
    #print(k_vals[j])
    max_accessible_harm[j]=np_harm_range[np.where(differences[1,:]<5)][-1]#error between noisy and non-noisy
    mih=np_harm_range[np.where(differences[0,:]>1)]
    if mih.size==0:
        return_mih=100
    else:
        return_mih=mih[0]
    min_informative_harm[j]=return_mih#error between shifted and un-shifted
    max_diff_val[j]=max(differences[0,:])

    #print( differences[1,:][np.where(differences[1,:]<5)])
    print(max_diff_val[j])
np.save("k_access", np.vstack((k_vals, max_accessible_harm, min_informative_harm, max_diff_val)))
plt.plot(k_vals, max_accessible_harm)
plt.plot(k_vals, min_informative_harm)
plt.show()
"""
plt.show()
plt.subplot(1,2,1)
plt.plot(harm_range, differences[0,:], label="Shift error")
plt.plot(harm_range, differences[1,:], label="Noise error")
plt.legend()
plt.subplot(1,2,2)
plt.plot(harm_range, abs(differences[0,:]-differences[1,:]), color="green")
plt.show()"""