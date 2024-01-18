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

import matplotlib.pyplot as plt
import math
import time
from single_e_class_unified import single_electron

import pints
from pints import plot
F=96485.3329
R=8.31446261815324
Es=0.15
dE=4e-3
DeltaE=0.3
Esw=80e-3

n=1
T=298
alpha=0.5
sampling_factor=200
param_list={
"E_0":0.0,
'E_start':  Es, #(starting dc voltage - V)
'scan_increment': dE,   #(ac voltage amplitude - V) freq_range[j],#
'area': 0.07, #(electrode surface area cm^2)
'gamma': 1e-10,
"omega":0.5,
"Ru":0,
"original_gamma":1e-10,
"T":273+25,
"n":n,
'k_0': 75, #(reaction rate s-1)
'alpha': 0.5,
"sampling_factor":sampling_factor,
"SW_amplitude":Esw,
"deltaE":DeltaE
}
K=param_list["k_0"]/param_list["omega"]
print((DeltaE/dE)*50)
simulation_options={
"method":"square_wave",
"experimental_fitting":False,
"likelihood":"timeseries",
"square_wave_return":"net",
"optim_list":["E_0", "k_0", "alpha"],
"no_transient":False
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":range(0, 1),
    "experiment_time": None,
    "experiment_current": None,
    "experiment_voltage":None,
    "bounds_val":200,
}
param_bounds={
"E_0":[param_list["E_start"]-param_list["deltaE"], param_list["E_start"]],
"k_0":[0.1, 1e3],
"alpha":[0.4, 0.6]
}
omega_list=[5, 10]
noise_val=[0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
omega_counter=0
SWV_chain_dict={}
for k0_val in [0.5]:
    param_list["omega"]=omega_list[omega_counter]
    omega_counter+=1
    key="k0={0}".format(k0_val)
    SWV_chain_dict[key]={}
    for i in range(0, len(noise_val)):
        SW=single_electron(None, param_list, simulation_options, other_values, param_bounds)
        
        end=int((DeltaE/dE)*sampling_factor)
        test=SW.test_vals([0.0, k0_val, 0.5], "timeseries")
        plt.plot(test)
        plt.show()
        
        
