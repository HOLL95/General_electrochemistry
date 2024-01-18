
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
from EIS_optimiser import EIS_genetics
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
import numpy as np
import pints
from pints.plot import trace
data_loc="/home/henryll/Documents/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=0
frequencies=np.flip(data[:,2])
param_list={
       "E_0":0.001,
        'E_start':  DC_val-2e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+2e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 2e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "k0_shape":0.4,
        "k0_scale":10,
        "phase":0,
        "E0_mean":DC_val,
        "E0_std":0.02,
        "cap_phase":0,
        "num_peaks":3
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "DC_pot":0,
    "invert_imaginary":False,
    "data_representation":"nyquist",
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'E0_mean':[-0.05, 0.05],
    'E0_std':[1e-3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[0, 2*math.pi],
    "k0_shape":[0,2],
    "k0_scale":[0,1e4],

    "cap_phase":[0, 2*math.pi],
}

td=EIS_TD(param_list, simulation_options, other_values, param_bounds)

lav_cdl_val= 7.246044133637491e-07
fig, axes=plt.subplots(2,2)
ax=axes[0,0]
twinx=ax.twinx()
twinx2=axes[0,1].twinx()
twinx_3=axes[1,0].twinx()
laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)

laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru"])
lav_circuit={"z1":"R0", "z2":{"p1":"C1", "p2":["R1", "C2"]}}
lav_ec=EIS(circuit=lav_circuit)
#save_data=EIS().convert_to_bode(np.column_stack((real, z.imag[index])))
ec_data_1=lav_ec.test_vals({'R0': 100.89130837867, 'R1': 985.4168866804519, 'C1': 7.246044124672741e-07, 'C2': 6.522158185693151e-06}, np.multiply(frequencies,1))
EIS().bode(ec_data_1, frequencies, ax=ax, twinx=twinx, scatter=1, line=False)
#EIS().bode(ec_data_1, frequencies, ax=axes[2], twinx=twinx_3, scatter=1, line=False)

for scale in [60, 75.6999999, 90]:
    params=[1e-10, scale,lav_cdl_val, 0.55, 100]
   
    
    bode_vals=td.simulate(params, frequencies)
    lav_ec_vals=laviron.simulate(params, frequencies)
    EIS().bode(lav_ec_vals, frequencies, ax=ax, twinx=twinx, label=scale)
    EIS().bode(bode_vals, frequencies, ax=ax, twinx=twinx, label=scale)
td.def_optim_list(["k0_shape", "k0_scale","gamma", "Cdl", "alpha", "Ru", "phase", "cap_phase"])
plt.show()
EIS().bode(ec_data_1, frequencies, ax=axes[1,0], twinx=twinx2, scatter=1, line=False)
for shape in [0.25, 0.5, 0.75, 1.0, 1.25]:
    params=[shape, 75.69999, 1e-10, 1e-5, 0.55, 100, 0, 0]
    param_dict=dict(zip(td.optim_list, params))
    
    td_sim=td.simulate(params, frequencies)
    EIS().bode(td_sim, frequencies, ax=axes[1], twinx=twinx2, label=shape)
    print(td.values, "td")
    #EIS().bode(bode_vals, frequencies, ax=ax, twinx=twinx, label=scale)
laviron.def_optim_list(["gamma","k0_shape", "k0_scale",  "Cdl", "alpha", "Ru"])
laviron.simulation_options["dispersion_test"]=True
for shape in [0.25, 0.5, 0.75, 1.0, 1.25]:
    params=[1e-10, shape, 75.69999, lav_cdl_val, 0.55, 100]
    lav_ec_vals=laviron.simulate(params, frequencies*2*math.pi)
    EIS().bode(lav_ec_vals, frequencies, ax=axes[1], twinx=twinx2, label=shape, lw=3, alpha=0.5)

for i in range(0, len(laviron.disp_test),3):
    EIS().bode(laviron.disp_test[i], frequencies, ax=axes[2], twinx=twinx_3)
EIS().bode(lav_ec_vals, frequencies, ax=axes[2], twinx=twinx_3, lw=4)
axes[0,1].legend()
axes[1,0].legend()
ax.legend()

plt.show()

