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
from heuristic_class import Laviron_EIS
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from pints.plot import trace
data_loc="/home/henry/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
#data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)
truncate=10
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])
EIS().nyquist(np.column_stack((real, imag)),orthonormal=False)
plt.title("Ferrocene Nyquist plot")
plt.show()


param_list={
        "E_0":0.24,
        'E_start':  0.24-10e-3, #(starting dc voltage - V)
        'E_reverse':0.24+10e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "Cfarad":0,
        "E0_mean":0,
        "E0_std": 0.025,
        "k0_shape":0.4,
        "k0_scale":75,
        "num_peaks":30,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[300],
    "GH_quadrature":False,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "C_sim":True,
    "optim_list":[],
    "EIS_Cf":"C",
    "EIS_Cdl":"C",
    "DC_pot":240e-3,
    "Rct_only":False,
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0.15, 0.35],
    "E0_mean":[0.15, 0.35],
    "E0_std":[0.001, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-7],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0.65,1],
    "k0_shape":[0,10],
    "k0_scale":[0,200],
    "phase":[-180, 180],
}
import copy

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
#laviron.def_optim_list(["E0_mean", "E0_std", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
laviron.def_optim_list(["k0_shape", "k0_scale", "gamma", "Cdl",  "Ru", "cpe_alpha_cdl"])
#laviron.def_optim_list(["E0_mean","E0_std",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])#"E0_mean","E0_std","k0_scale","k0_shape"
#laviron.def_optim_list([ "Cdl", "alpha", "Ru"])#"E0_mean","E0_std","k0_scale","k0_shape"

spectra=np.column_stack((real, imag))
EIS().bode(spectra, frequencies)
plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)
EIS_params_5={'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}

"""sim_data=laviron.simulate([EIS_params_5[x] for x in laviron.optim_list], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
ax.legend()
ax.set_title("C fit")
plt.show()"""



laviron.simulation_options["label"]="MCMC"
laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from MCMC_plotting import MCMC_plotting
from pints import plot
desired_files=["EIS_k_disp_tests_2"]
mplot=MCMC_plotting(burn=90000)

import statsmodels.api as sm
from itertools import combinations
import _pickle as cPickle





names=["k0_shape", "k0_scale", "gamma", "Cdl", "Ru", "cpe_alpha_cdl", "sigma_1","sigma_2"]
ranges=[[0.1, 1.5], [1, 75], [5e-11, 3e-9],[1e-6, 1e-5], [50, 100], [0.6, 0.9]]


chains=np.load(desired_files[0])
catted_chains=mplot.concatenate_all_chains(chains)
#catted_chains=[chains[2, :, param] for param in range(0, len(names))]
chain_dict=dict(zip(names, catted_chains))
trough_params=["gamma", "k0_scale"]

variable_params=names[:-2]
range_dict=dict(zip(variable_params, ranges))
trough_params_target=[5e-11, 75]
num_vars=100
monster_dict={}
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
from copy import deepcopy
import time
start=time.time()
for i in range(0,len(trough_params)):#
    key=trough_params[i]
    depend_axis=names.index(key)

    independent_var=chain_dict[key]
    line_vals=[np.mean(independent_var), trough_params_target[i]]
    
    min_val=range_dict[key][0]
    max_val=range_dict[key][1]
    plot_line=np.logspace(np.log10(min_val), np.log10(max_val), num_vars)
    nearest_idx=find_nearest(plot_line, EIS_params_5[key])
    plot_line[nearest_idx]=EIS_params_5[key]
    not_that_variable=[x for x in variable_params if x is not key]
    for j in range(0, len(not_that_variable)):#
        save_key=trough_params[i]+"-"+not_that_variable[j]
        independ_axis=names.index(not_that_variable[j])
        monster_dict[save_key]={x:{"params":deepcopy(EIS_params_5), "score":1e6} for x in range(0, num_vars**2)}
        second_key=not_that_variable[j]
        predictions=np.logspace(np.log10(range_dict[second_key][0]), np.log10(range_dict[second_key][1]), num_vars)
        nearest_idx2=find_nearest(predictions, EIS_params_5[second_key])        
        predictions[nearest_idx2]=EIS_params_5[second_key]
        #plt.scatter(predictions, plot_line)
        #plt.show()
        for lcv_1 in range(0, num_vars):
            
            
            current_variable=predictions[lcv_1]
           
            #plot_var_list=np.linspace((predictions[lcv_1]*0.5), (predictions[lcv_1]*1.5))
            for lcv_2 in range(0, num_vars):
                idx=(num_vars*lcv_1)+lcv_2
                monster_dict[save_key][idx]["params"][key]=plot_line[lcv_1]
                monster_dict[save_key][idx]["params"][not_that_variable[j]]=predictions[lcv_2]
                sim_params=[monster_dict[save_key][idx]["params"][x] for x in laviron.optim_list]
                sim_data=laviron.simulate(sim_params, fitting_frequencies)
                score1=laviron.RMSE(sim_data[:,0], data_to_fit[:,0])
                score2=laviron.RMSE(sim_data[:,1], data_to_fit[:,1])
                monster_dict[save_key][idx]["score"]=score1+score2
        #for key1 in monster_dict[save_key].keys():
        #    print(monster_dict[save_key][key1]["params"])
with open(r"simple_profile.pickle", "wb") as output_file:
    cPickle.dump(monster_dict, output_file)
print(time.time()-start)