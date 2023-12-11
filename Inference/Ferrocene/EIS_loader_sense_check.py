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
    "dispersion_bins":[30,300],
    "GH_quadrature":False,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "C_sim":True,
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
    "cpe_alpha_cdl":[0,1],
    "k0_shape":[0,10],
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
}
import copy

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
#laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E0_mean", "E0_std", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E_0",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
laviron.def_optim_list(["E0_mean","E0_std",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#"E0_mean","E0_std","k0_scale","k0_shape"
spectra=np.column_stack((real, imag))
#EIS().bode(spectra, frequencies)
#plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)

"""sim_data=laviron.simulate([cdl_only[x] for x in laviron.optim_list], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
ax.legend()
ax.set_title("C fit")
plt.show()

"""

#laviron.simulation_options["label"]="cmaes"
laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)
#C
EIS_params1={'E_0': 0.23708843969139082, 'k_0': 4.028523388682444, 'gamma': 7.779384163661676e-10, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#CPE
EIS_params2={'E_0': 0.3047451457126534, 'k_0': 39.40663787158313, 'gamma': 1.0829517784499947e-10, 'Cdl': 8.7932554621096e-06, 'alpha': 0.5394294479538084, 'Ru': 80.76397847517714,"omega":8.794196510802587}
#Cfarad
EIS_params3={'E_0': 0.2161051668499098, 'k_0': 106.6602436491309, 'gamma': 2.5360979030661595e-11, 'Cdl': 8.751614540745486e-06, 'alpha': 0.47965820103670564, 'Ru': 80.92159716231082,"omega":8.794196510802587}
#No CPE
EIS_params_4={'E_0': 0.2014214483444881, 'k0_scale': 1.0950956335756536, 'k0_shape': 1.043401547065882, 'gamma': 1.4645920920242938e-09, 'Cdl': 7.945475589121264e-06, 'alpha': 0.359816590101354, 'Ru': 81.69086207153816, 'cpe_alpha_cdl': 0.768033972041215, 'cpe_alpha_faradaic': 0.9119951540999298, 'phase': -0.20113351781378697,"omega":8.794196510802587} 
EIS_params_4={'E_0': 0.19463469159444804, 'k0_shape': 2.392919243565184, 'k0_scale': 0.2064967806796385, 'gamma': 3.924680690469505e-09, 'Cdl': 9.90894008867702e-07, 'alpha': 0.5883138432002848, 'Ru': 90.57829277724034, 'cpe_alpha_cdl': 0.5659794458625567, 'cpe_alpha_faradaic': 0.14317733968270077}


#CPE but no extra phase
EIS_params_5={'E_0': 0.15993573511018355, 'k0_shape': 1.0143738017899193, 'k0_scale': 0.4719318521743649, 'gamma': 5.364902729622656e-09, 'Cdl': 7.636444325985663e-06, 'alpha': 0.364166232234114, 'Ru': 81.92130789951946, 'cpe_alpha_cdl': 0.7725503185958909, 'cpe_alpha_faradaic': 0.008866416527890587}
EIS_params_5={'E_0': 0.19066872485338204, 'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}
#E0_mean with CPE
EIS_params_6={'E0_mean': 0.3499999999999999, 'E0_std': 0.045854752108924646, 'k_0': 0.9166388879743895, 'gamma': 5.405583319246063e-09, 'Cdl': 9.23395426422013e-06, 'alpha': 0.6499999999999999, 'Ru': 80.37330196288727, 'cpe_alpha_cdl': 0.7495664487939422, 'cpe_alpha_faradaic': 0.1477882191366903}
#E0_mean with C
EIS_params_7={'E0_mean': 0.35, 'E0_std': 0.059192130273338424, 'k_0': 1.678514486845095, 'gamma': 4.586111119553072e-09, 'Cdl': 1.3948590417154984e-06, 'alpha': 0.65, 'Ru': 96.39939088176911, 'cpe_alpha_cdl': 0.6612954622562719, 'cpe_alpha_faradaic': 0.9995161877220936}




#EIS_params_5={'E_0': 0.17367485939633537, 'k0_shape': 2.4360726160882744, 'k0_scale': 0.15942388883368433, 'gamma': 8.019805323074907e-09, 'Cdl': 9.844551737915205e-07, 'alpha': 0.6465984705927521, 'Ru': 91.5686733543809, 'cpe_alpha_cdl': 0.9169107302998178, 'cpe_alpha_faradaic': 0.9863374008502952, 'phase': -1.9689345255496846}
#Both no C
EIS_params_9={'E0_mean': 0.2552237543929984, 'E0_std': 0.0010014967867590788, 'k0_shape': 2.4360714665636465, 'k0_scale': 0.22242584355076356, 'gamma': 2.2858275700178405e-09, 'Cdl': 9.844551474481184e-07, 'alpha': 0.35822572276185904, 'Ru': 91.56868145656738, 'cpe_alpha_cdl': 0.5070420269200153, 'cpe_alpha_faradaic': 0.1970959976913189, 'phase': -1.9689465346727104}
fig, ax=plt.subplots()
twinx=ax.twinx()
sim_vals=laviron.simulate([EIS_params_9[x] for x in laviron.optim_list], fitting_frequencies)
EIS().bode(np.column_stack((real, imag)),frequencies,ax=ax, twinx=twinx)
EIS().bode(sim_vals,frequencies,ax=ax, twinx=twinx, data_type="phase_mag")
plt.show()