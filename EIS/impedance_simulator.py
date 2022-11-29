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
import math
from collections import deque
from params_class import params
from single_e_class_unified import single_electron
import isolver_martin_brent
param_list={
    "E_0":0.0,
    'E_start':  0.0, #(starting dc voltage - V)
    'E_reverse':-5*1e-3,
    'omega':10, #8.88480830076,  #    (frequency Hz)
    "original_omega":10,
    'd_E': 0.1*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 10.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "psi":0.5,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 0.001    , #(reaction rate s-1)
    'alpha': 0.5,
    "cap_phase":3*math.pi/2,
    'sampling_freq' : (1.0/400),
    'phase' :3*math.pi/2,
    "num_peaks":20
}
import copy
orig_param_list=copy.deepcopy(param_list)
sim_options={
    "method":"sinusoidal",
    "experimental_fitting":False,
    "likelihood":"timeseries"
}

frequencies=np.logspace(-1, 6, 60)
impedance=np.zeros(len(frequencies), dtype="complex")
for i in range(0, len(frequencies)):
    param_list["original_omega"]=frequencies[i]
    param_list["omega"]=frequencies[i]
    sim_options["no_transient"]=2/param_list["omega"]
    eis_test=single_electron(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
    current=eis_test.i_nondim(eis_test.test_vals([],"timeseries"))
    potential=eis_test.e_nondim(eis_test.define_voltages()[eis_test.time_idx])
    t=eis_test.t_nondim(eis_test.time_vec)
    i_fft=np.fft.fft(current)
    e_fft=np.fft.fft(potential)
    Z_of_omega      = e_fft/i_fft
    fft_freq=np.fft.fftfreq(len(current), t[1]-t[0])
    ft_idx=np.abs(np.subtract(fft_freq, frequencies[i]))
    peak_loc=np.where(ft_idx==min(ft_idx))[0][0]
    impedance[i] = Z_of_omega[peak_loc]
    #plt.plot(fft_freq, i_fft)
    #plt.axvline(fft_freq[peak_loc])
    #plt.show()
plt.scatter(np.real(impedance), -np.imag(impedance))
plt.show()
