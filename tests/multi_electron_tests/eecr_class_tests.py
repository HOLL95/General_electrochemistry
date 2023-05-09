import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:dir_list.index("General_electrochemistry")+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import math
import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
from square_scheme import square_scheme

from harmonics_plotter import harmonics
import time
import pints
from pints import plot

harm_range=list(range(1, 13))
cc=0

param_list={
       "E_0":0.0,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':300e-3,
        'omega':10,
        "v":50e-3,  #    (frequency Hz)
        'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-4, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        'sampling_freq' : (1.0/50),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 30,

    }


simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "disperson":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[],

}
other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  
    'Cdl': [0,1e-3], 
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-9],
    'k_0': [0.1, 1000], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "cap_phase":[math.pi/2, 2*math.pi],
    'phase' : [0, 2*math.pi],
}
linkage_dict=elements=["AoRo", "AoRr", "AiRo", "AiRr", "ArRo", "ArRr"]
linked_list={   
            "AoRo":{"AoRr":{"type":"BV_red", "group":None},"ArRo" :{"type":"Cat", "group":None}},
            "AoRr":{"AoRo":{"type":"BV_ox", "group":None} ,"AiRo":{"type":"Cat", "group":None}, "ArRr":{"type":"Cat", "group":None}},
            "AiRo":{"AoRr":{"type":"Cat", "group":None},"AiRr":{"type":"BV_red", "group":None}},
            "AiRr":{"AiRo":{"type":"BV_ox", "group":None},"ArRo":{"type":"Cat", "group":None}},
            "ArRo":{"AoRo":{"type":"Cat", "group":None},"AiRr":{"type":"Cat", "group":None},"ArRr":{"type":"BV_red", "group":None}},
            "ArRr":{"ArRo":{"type":"BV_ox", "group":None},"AoRr":{"type":"Cat", "group":None}},
}
"""linked_list={   
            "AoRo":{"AoRr":{"type":"BV_red", "group":None},},
            "AoRr":{"AoRo":{"type":"BV_ox", "group":None} ,"AiRo":{"type":"BV_red", "group":None}, },
            "AiRo":{"AoRr":{"type":"BV_ox", "group":None},"AiRr":{"type":"BV_red", "group":None}},
            "AiRr":{"AiRo":{"type":"BV_ox", "group":None},"ArRo":{"type":"BV_red", "group":None}},
            "ArRo":{"AiRr":{"type":"BV_ox", "group":None},"ArRr":{"type":"BV_red", "group":None}},
            "ArRr":{"ArRo":{"type":"BV_ox", "group":None}},
}"""
#linked_list={   
#            "AoRo":{"AoRr":{"type":"BV_red", "group":None},},
#            "AoRr":{"AoRo":{"type":"BV_ox", "group":None} ,"AiRo":{"type":"BV_red", "group":None}, },
#            "AiRo":{"AoRr":{"type":"BV_ox", "group":None},"AiRr":{"type":"BV_red", "group":None}},
#            "AiRr":{"AiRo":{"type":"BV_ox", "group":None},},

#}
specs=list(linked_list.keys())
simulation_options["linkage_dict"]=linked_list
simulation_options["subtracted_species"]="ArRr"
eecr=square_scheme(param_list, simulation_options, other_values, param_bounds)
eecr_farad_params=eecr.farad_params
param_vals=np.random.rand(len(eecr_farad_params))
e0_counter=1
plt.plot(eecr.define_voltages())
plt.show()
e0_vals=[]
for i in range(0, len(eecr_farad_params)):
    if "k" in eecr_farad_params[i]:
        param_vals[i]=eecr.un_normalise(param_vals[i], param_bounds["k_0"])
    elif "E0" in eecr_farad_params[i]:
        param_vals[i]=param_list["E_reverse"]-(e0_counter*0.15)
        e0_vals.append(param_vals[i])
        e0_counter+=1
    elif "alpha" in eecr_farad_params[i]:
        param_vals[i]=eecr.un_normalise(param_vals[i], param_bounds["alpha"])
print(param_vals, eecr.farad_params)
dim_dict=dict(zip(eecr_farad_params, param_vals))
eecr.def_optim_list(eecr_farad_params)
import time
start=time.time()
current=eecr.simulate(param_vals, [])
print(time.time()-start)
plt.plot(eecr.e_nondim(eecr.define_voltages())[5:], current[5:])
plt.show()