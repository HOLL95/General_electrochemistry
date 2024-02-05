import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
import numpy as np
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
data_loc="/home/henryll/Documents/Experimental_data/Ella/SWV/SWV_exp/"
files=["CjAA10_red.txt","CjAA10_ox.txt"]
file="CjAA10_red.txt"
data=np.loadtxt(data_loc+file)

import matplotlib.pyplot as plt
import math
import time
from single_e_class_unified import single_electron

plt.plot(data[:,0],data[:,1])


plt.show()
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
scan_direction=-1
estep=2e-3
param_list={
"E_0":0.0,
'E_start': 0.25, #(starting dc voltage - V)
'scan_increment': estep,   #(ac voltage amplitude - V) freq_range[j],#
'area': 0.07, #(electrode surface area cm^2)
'gamma': 1e-11,
"omega":10,
"Ru":0,
"original_gamma":1e-11,
"T":273+25,
"n":n,
'k_0': 75, #(reaction rate s-1)
'alpha': 0.5,
"sampling_factor":sampling_factor,
"SW_amplitude":2e-3,
"deltaE":0.45,
"v":scan_direction,
}
K=param_list["k_0"]/param_list["omega"]
print(param_list["deltaE"])
simulation_options={
"method":"square_wave",
"experimental_fitting":False,
"likelihood":"timeseries",
"square_wave_return":"backwards",
"optim_list":["E_0", "k_0", "alpha","gamma"],
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
noise_val=[0.005]#, 0.01, 0.02, 0.03, 0.04, 0.05]
omega_counter=0
SWV_chain_dict={}

SW=single_electron(None, param_list, simulation_options, other_values, param_bounds)

end=int((DeltaE/dE)*sampling_factor)
volts=SW.e_nondim(SW.define_voltages())
f, b, subtract, E_p=SW.SW_peak_extractor(volts)
experimental_current=data[:-1,1]/SW.nd_param.sw_class.c_I0
sim_current=SW.simulate([0.0, 4.75, 0.5, 1e-11], [])
plt.plot(volts)
plt.scatter(SW.f_idx,E_p)
plt.scatter(SW.f_idx,data[:-1,0], linestyle="--")
plt.show()

plt.plot(E_p, sim_current)
plt.plot(E_p, experimental_current)
plt.show()