import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from collections import deque
from params_class import params
from single_e_class_unified import single_electron
from convolutive_modelling_class import conv_model
import isolver_martin_brent
import mpmath
import math
param_list={
    "E_0":0.25,
    'E_start':  0.0, #(starting dc voltage - V)
    'E_reverse':0.5,
    'omega':10, #8.88480830076,  #    (frequency Hz)
    "original_omega":10,
    "v":0.25,
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "psi":0.5,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10, #(reaction rate s-1)
    'alpha': 0.5,
    "cap_phase":3*math.pi/2*0,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2*0,
    "num_peaks":20
}
import copy
orig_param_list=copy.deepcopy(param_list)
sim_options={
    "method":"ramped",
    "experimental_fitting":False,
    "likelihood":"timeseries"
}

#test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
frequency_powers=np.linspace(-1, 5, 10)
frequencies=[10**float(x) for x in frequency_powers]
z=np.zeros((2, len(frequencies)))
print(frequencies)

test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
current=test_class.simulate_current(CPE=True)
numerical_current=test_class.test_vals([], "timeseries")
potential=test_class.e_nondim(test_class.define_voltages())
plt.plot(test_class.time_vec, current)

plt.plot(test_class.time_vec, numerical_current)
plt.show()
