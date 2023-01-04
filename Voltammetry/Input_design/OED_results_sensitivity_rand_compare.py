import matplotlib.pyplot as plt
import numpy as np

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
import copy
import math
from single_e_class_unified import single_electron
from MCMC_plotting import MCMC_plotting
harm_range=list(range(4, 6))

param_list={
    "E_0":0.0,
    'E_start':  -600e-3, #(starting dc voltage - V)
    'E_reverse':-100e-3,
    'omega':8.88480830076,  #    (frequency Hz)
    "v":200e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0.000653657774506,
    'CdlE2': 0.000245772700637,
    "CdlE3":-1e-6,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
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
    'k_0': [0.1, 2e3], #(reaction rate s-1)
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
    "all_freqs":[1e-5, 2000],
    "all_amps":[1e-5, 0.5],
    "all_phases":[0, 2*math.pi],
}
mplot=MCMC_plotting()
sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
rpotential=sim.e_nondim(sim.define_voltages())
rcurrent=sim.test_vals([], "timeseries")
sim.simulation_options["method"]="sum_of_sinusoids"
#simulation_params=["E_0", "k_0", "Ru", "gamma", "alpha"]
num_points=10
param_ranges={
            "E_0":[0.25, 0.5, 0.75], 
            "k_0":[0.1, 100, 1000], 
            "gamma":[1e-11, 5e-11, 1e-10],
            "Ru":[0.1, 10, 1000],
            "alpha":[0.4, 0.5, 0.6]}
param_ranges={x:np.linspace(param_bounds[x][0], param_bounds[x][1], num_points) for x in param_ranges.keys()}
files=[x+".npy" for x in ["Sobol_D_1"]]
results_dict=np.load(files[0], allow_pickle=True).item()
freq_params=results_dict["params"]

farad_params=list(param_ranges.keys())
ref_farad_params=[param_ranges[x][1] for x in farad_params]
ref_farad_params[0]=0
rands=np.random.rand(len(freq_params))
rand_freq_values=np.zeros(len(freq_params))
for i in range(0, len(freq_params)):
    appropriate_key="all_{0}s".format(freq_params[i][:freq_params[i].index("_")])
    rand_freq_values[i]=sim.un_normalise(rands[i], [param_bounds[appropriate_key][0], param_bounds[appropriate_key][1]])

f_params=["freq_1", "amp_1", "phase_1"]
sim.def_optim_list(f_params)
freq_vals=[10, 1, 3*math.pi/2]
titles=mplot.get_titles(farad_params, units=True)
units=[mplot.unit_dict[x] for x in farad_params]
units[2]="pmol cm$^{-2}$"
fig, axis=plt.subplots(2,3)
shift_vals=[10e-3, 1, 5, 1, 0.01]
shift=dict(zip(param_ranges.keys(), [10e-3, 1, 0.5e-11, 1, 0.01]))
labels=["PSV", "Rand"]
for file in files:
    results_dict=np.load(file, allow_pickle=True).item()    
    

    for i in range(0, len(results_dict["param_values"])+2 ):
        if i==0:
            freq_params=f_params
            f_vals=freq_vals
        elif i==1:
            freq_params=results_dict["params"]
            f_vals=rand_freq_values
        else:
            freq_params=results_dict["params"]
            f_vals=results_dict["param_values"][i-2]
        sim.def_optim_list(freq_params)
        print(sim.optim_list)
        
        potential=sim.e_nondim(sim.define_voltages())
        all_param_names=farad_params+freq_params
        sim.def_optim_list(all_param_names)
        param_bounds["E_0"]=[min(potential), max(potential)]
        times=sim.t_nondim(sim.time_vec)
        pc_diff=np.zeros(num_points)
        for j in range(0, len(farad_params), 2):
            ax=axis[j//3, j%3]
            sim_farad_params=copy.deepcopy(ref_farad_params)
            for z in range(0, len(param_ranges[farad_params[j]])):
                if farad_params[j]=="E_0":
                    value=sim.un_normalise(param_ranges[farad_params[j]][z], param_bounds["E_0"])
                else:
                    value=param_ranges[farad_params[j]][z]
                sim_farad_params[j]=value
                #print(sim_farad_params, farad_params[j])
                current=sim.test_vals(np.append(sim_farad_params,f_vals), "timeseries")
                sim_farad_params[j]+=shift[farad_params[j]]
                shifted_current=sim.test_vals(np.append(sim_farad_params,f_vals), "timeseries")
                pc_diff[z]=100*(sim.RMSE(shifted_current, current)/np.mean(np.abs(current)))

                
            
            if i<2:
                ax.plot(param_ranges[farad_params[j]], np.log10(pc_diff), linestyle="--")
                
            else:
                ax.plot(param_ranges[farad_params[j]], np.log10(pc_diff), label=i-1)
            ax.set_xlabel(titles[j])
            ax.set_ylabel("log10(%) change after shift of {0} {1}".format(units[j], shift_vals[j]))
ax.legend()
axis[-1, -1].set_axis_off()
plt.show()#
