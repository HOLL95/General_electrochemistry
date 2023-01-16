import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from matplotlib.ticker import FuncFormatter
from pints import plot
from harmonics_plotter import harmonics
import matplotlib.ticker as mticker
import os
import seaborn
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
from MCMC_plotting import MCMC_plotting
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pints.plot
harm_range=list(range(1, 8))
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
dimensions=10
Ru_vals=np.logspace(0, 3, dimensions)
k0_vals=np.logspace(0, 3, dimensions)
frequencies=[10]
results_array=np.zeros((dimensions, dimensions))

SRS=[25, 50, 100, 200, 400]
keys=["Rhat", "Width", "Divergence"]
results_dict={key:{"k_0":copy.deepcopy(results_array), "Ru":copy.deepcopy(results_array)} for key in keys}
true_sf=400
params=["E_0", "k_0", "Ru", "Cdl", "gamma", "alpha", "phase"]
informative_params=["k_0", "Ru"]
k_loc=1
r_loc=2
labels=["", "k_0", "Ru"]
burn=5000
mplot=MCMC_plotting(burn=burn)
len_params=len(params)
for i in range(0, dimensions):
    for j in range(0, dimensions):
        for k in range(0, len(frequencies)):
            param_list={
                "E_0":0.2,
                'E_start':  -200e-3, #(starting dc voltage - V)
                'E_reverse':400e-3,
                'omega':frequencies[k],  #    (frequency Hz)
                "original_omega":frequencies[k] ,
                'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
                'area': 0.07, #(electrode surface area cm^2)
                'Ru': Ru_vals[-i],  #     (uncompensated resistance ohms)
                'Cdl':1e-4, #(capacitance parameters)
                'CdlE1': 0.000653657774506*0,
                'CdlE2': 0.000245772700637*0,
                "CdlE3":0,
                'gamma': 5e-11,
                "original_gamma":5e-11,        # (surface coverage per unit area)
                'k_0': k0_vals[j], #(reaction rate s-1)
                'alpha': 0.5,
                "E0_mean":0.2,
                "E0_std": 0.09,
                "cap_phase":3*math.pi/2,
                "alpha_mean":0.5,
                "alpha_std":1e-3,
                'sampling_freq' : (1.0/true_sf),
                'phase' :3*math.pi/2,
                "time_end": None,
                'num_peaks': 30,
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
                "test":False,
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
            
            noise_vals=0.01
            
            save_file="MCMC/parameter_scan/Low_cdl/R_{0}_k_{1}_SR_{2}_10_Hz".format(round(param_list["Ru"],2), round(param_list["k_0"],2), 400)
            chains=np.load(save_file)
            
            rhats={labels[i]:pints.rhat(chains[:, burn:, i]) for i in [k_loc, r_loc]}

            #if rhats["Ru"]<1.05:
            #    print(rhats)
            #    print(i, j)
            #    pints.plot.trace(chains)
            #    plt.show()
            widths={labels[i]:round(100*np.std(mplot.chain_appender(chains, i))/param_list[labels[i]],2) for i in [k_loc, r_loc]}
            divergence={labels[i]:np.log10(100*abs(np.mean(chains[:, burn:, i])-param_list[labels[i]])/param_list[labels[i]]) for i in [k_loc, r_loc]}
            metric={"Rhat":rhats, "Width":widths, "Divergence":divergence}
            for key in keys:
                for param in informative_params:
                    #if key=="Rhat":

                        #if metric[key][param]>1.1:
                        #    print(metric[key])
                        #    print(save_file)
                        #    pints.plot.trace(chains)
                        #    plt.show()
                        #    metric[key][param]=2
                   # if key=="Divergence":
                    #    if param=="k_0":
                    #        if metric[key][param]>1:
                    #            print(metric[key])
                    #            print(save_file)
                    #            pints.plot.trace(chains)
                    #            plt.show()
                                
                    results_dict[key][param][i,j]=metric[key][param]

fig, ax=plt.subplots(2,2)
ticklabels=mplot.get_titles(informative_params)
plot_ticks=[round(x, 2) for x in Ru_vals]
labels=["Rhat", "Error (% of true value)"]
heatmap_params=["Rhat", "Divergence"]
def logformat(x, pos):
    'The two args are the value and tick position'
    return '$10^{'+str(int(x))+"}$"
for i in range(0, 2):
    for j in range(0, len(informative_params)):
        axes=ax[i,j]
        if keys[i]=="Rhat":
             annot=None
        else:
            annot=results_dict["Width"][informative_params[j]]
        seaborn.heatmap(results_dict[heatmap_params[i]][informative_params[j]], ax=axes, 
                        yticklabels=np.flip(plot_ticks), xticklabels=plot_ticks, cmap="viridis_r", cbar_kws={"label":labels[i], "format":FuncFormatter(logformat)}, annot=annot, robust=True)
        axes.set_xlabel(ticklabels[0])
        axes.set_ylabel(ticklabels[1])
        axes.set_title(informative_params[j]+" "+heatmap_params[i])
plt.subplots_adjust(hspace=0.38,
                    left=0.05, 
                    right=0.995, 
                    top=0.961, 
                    bottom=0.1)
plt.show()
