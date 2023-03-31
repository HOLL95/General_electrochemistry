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
from MCMC_plotting import MCMC_plotting
from heuristic_class import DCVTrumpet
mplot=MCMC_plotting()
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pints.plot
harm_range=list(range(4, 8))
from scipy import interpolate
from scipy.interpolate import CubicSpline

dimensions=5



fig, ax=plt.subplots(2,3)
param_scan_bounds={
                "E0_mean":np.linspace(-0.25, -0.15, dimensions),
                "E0_std":np.linspace(0.005, 0.05, dimensions),
                #"E_0":np.linspace(-0.25, -0.15, dimensions),
               "Ru":np.logspace(0, 2.75, dimensions),
                "k_0":np.logspace(1, 3.25, dimensions),
                "alpha":np.linspace(0.4, 0.6, dimensions),                
                "gamma":np.linspace(3e-11, 7e-11, dimensions)
                }

param_keys=list(param_scan_bounds.keys())
units=mplot.get_units(param_keys)
titles=mplot.get_titles(param_keys, units=False)
mplot=MCMC_plotting()
for i in range(0, len(param_keys)):
    key=param_keys[i]
    for j in range(0, dimensions):
            param_list={
                "E_0":-0.2,
                'E_start': -0.5, #(starting dc voltage - V)
                'E_reverse':0.1,
                "v":100e-3, 
                "omega":0,
                'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
                'area': 0.07, #(electrode surface area cm^2)
                'Ru': 100,  #     (uncompensated resistance ohms)
                'Cdl':5e-5, #(capacitance parameters)
                'CdlE1': 0.000653657774506*0,
                'CdlE2': 0.000245772700637*0,
                "CdlE3":0,
                'gamma': 5e-11,
                "original_gamma":5e-11,        # (surface coverage per unit area)
                'k_0': 100, #(reaction rate s-1)
                'alpha': 0.55,
                "E0_mean":-0.2,
                "E0_std": 0.025,
                "cap_phase":3*math.pi/2,
                "alpha_mean":0.45,
                "alpha_std":1e-3,
                'sampling_freq' : (1.0/50),
                'phase' :3*math.pi/2,
                "cap_phase":3*math.pi/2,
                "time_end": None,
                'num_peaks': 12,
            }
            param_list[key]=param_scan_bounds[key][j]
            solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
            likelihood_options=["timeseries", "fourier"]
            
            simulation_options={
                "no_transient":False,
                "numerical_debugging": False,
                "experimental_fitting":False,
                "dispersion":False,
                "dispersion_bins":[16],
                "test":False,
                "record_exps":True,
                "method": "dcv",
                "phase_only":False,
                "likelihood":likelihood_options[0],
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
            sim=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
            scan_rates=[x*1e-3 for x in np.logspace(1, 4.5, 20)]
            
            #print(trumpet_vals, "ARSEJH")
            sim.def_optim_list(["E0_mean", "E0_std"])
            trumpet_vals=sim.simulate([param_list["E0_mean"], param_list["E0_std"]], scan_rates, optimise_flag=True)
            #current=sim.test_vals(, "timeseries")
            
            #plt.plot(sim.saved_sims["voltage"][-1], sim.saved_sims["current"][-1])
            if key=="gamma":
                dp=0
            else:
                dp=2
            val=mplot.format_values(param_scan_bounds[key][j], dp=dp)
            sim.trumpet_plot(scan_rates, sim.e_nondim(trumpet_vals),colour_counter=j, ax=ax[i//3, i%3], label=val)
            if units[i]!="":
                ax[i//3, i%3].set_title("{0} ({1})".format(titles[i], units[i]))
            else:
                ax[i//3, i%3].set_title(titles[i])
            ax[i//3, i%3].set_xlabel("log$_{10}$ scan rate")
            ax[i//3, i%3].set_ylabel("Peak position (V)")
            ax[i//3, i%3].legend(loc="upper center", ncols=2, handlelength=0.5, fontsize="medium", columnspacing=0.5)
    #plt.show()
fig.set_size_inches(14, 9)
plt.subplots_adjust(left=0.091,
                        bottom=0.08, 
                        right=0.946, 
                        top=0.973, 
                        wspace=0.3, 
                        hspace=0.2)
fig.savefig("Trumpet_plot_scans.png", dpi=500)
plt.show()

            

            
