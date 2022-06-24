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

param_list={
    "E_0":0.25,
    'E_start':  0.0, #(starting dc voltage - V)
    'E_reverse':0.5,
    'omega':10, #8.88480830076,  #    (frequency Hz)

    #"v":0.25,
    'd_E': 1*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 10.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-5, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "psi":0.5,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10    , #(reaction rate s-1)
    'alpha': 0.5,
    "cap_phase":0,
    'sampling_freq' : (1.0/200),
    'phase' :0,
    "num_peaks":20
}
import copy
orig_param_list=copy.deepcopy(param_list)
sim_options={
    "method":"sinusoidal",
    "experimental_fitting":False,
    "likelihood":"timeseries"
}

#test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
frequency_powers=np.linspace(-1, 5, 10)
frequencies=[10**float(x) for x in frequency_powers]
z=np.zeros((2, len(frequencies)))
print(frequencies)
for i in range(0, len(frequency_powers)):
    print(i)
    param_list["omega"]=frequencies[i]
    param_list["original_omega"]=frequencies[i]
    test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
    time=test_class.t_nondim(test_class.time_vec)
    ss_idx=np.where(time>1/param_list["original_omega"])
    potential=test_class.define_voltages()

    freq=np.fft.fftfreq(len(potential),time[1]-time[0])

    current=test_class.simulate_current(CPE=True)

    fft_pot=np.fft.fft(potential)
    fft_curr=np.fft.fft(current)
    one_tail_pot=fft_pot[1:len(fft_pot)//2+1]
    one_tail_curr=fft_curr[1:len(fft_pot)//2+1]
    freq=np.fft.fftfreq(len(potential), test_class.dt)[1:len(fft_pot)//2+1]
    #plt.subplot(1,2,1)
    #plt.plot(current)
    #plt.subplot(1,2,2)
    #plt.plot(potential)
    #plt.show()

    pred_imped=one_tail_pot/one_tail_curr
    abs_pot=np.abs(one_tail_pot)
    get_max=np.where(abs_pot==max(abs_pot))
    #plt.subplot(1,2,1)
    #plt.plot(freq, np.abs(one_tail_pot))
    #plt.axhline(abs_pot[get_max][0])
    #plt.subplot(1,2,2)
    #plt.title(frequencies[i])
    #plt.plot(freq, np.abs(one_tail_curr))
    #plt.show()
    impede=pred_imped[get_max][0]
    z[0][i]=np.real(impede)
    z[1][i]=-np.imag(impede)

#np.save("DCV_CPE_param_scans", save_dict)
#plt.legend()
#plt.show()
plt.plot(z[0,:], z[1,:])
plt.show()
