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
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from pints.plot import trace
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
#data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)
truncate=1
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])


DC_val=0
param_list={
        "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':0,
        "v":100e-3,  #    (frequency Hz)
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        "num_peaks":150,
        "cap_phase":0
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
    "optim_list":[],
    "C_sim":True,
    "EIS_Cf":"C",
    "EIS_Cdl":"C",
    "DC_pot":240e-3,
    "Rct_only":False,
}
DC_pot=240e-3
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
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
    "cap_phase":[-180,190]
}
import copy
td=EIS_TD(copy.deepcopy(param_list), copy.deepcopy(simulation_options), copy.deepcopy(other_values),copy.deepcopy(param_bounds))

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)

td.simulation_options["dispersion_bins"]=[100]
#laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E0_mean", "E0_std", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru"])
#laviron.def_optim_list(["E0_mean","E0_std",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
banned_param={"cpe_alpha_cdl", "cpe_alpha_faradaic"}
get_set=list(set(laviron.optim_list)-banned_param)
td_optim_list=[x for x in laviron.optim_list if x in get_set]+["phase","cap_phase"]
free_params=dict(zip(["phase","cap_phase"], [0,0]))
print(free_params)
td.def_optim_list(td_optim_list)

spectra=np.column_stack((real, imag))

fitting_frequencies=2*np.pi*frequencies

laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)

var_list={"E_0":[0.18, 0.24, 0.28],
            "k_0":[1, 10, 100],
            "gamma":[5e-12, 1e-11, 5e-11],
            "Cdl":[5e-6, 1e-5, 2e-5],
            "alpha":[0.4, 0.5, 0.6],
            "Ru":[10, 100, 1000],
            }
unit_dict={"E_0":"V",
            "k_0":"$s^{-1}$",
            "gamma":"mol cm$^{-2}$",
            "Cdl":"F cm$^{-2}$",
            "alpha":"",
            "Ru":"$\\Omega$"}
title_dict={"E_0":"$E^0$",
            "k_0":"$k_0$",
            "gamma":"$\\Gamma$",
            "Cdl":"$C_{dl}$",
            "alpha":"$\\alpha$",
            "Ru":"$R_u$"}
scale_dict={"E_0":{"num":2, "unit":"f"},
            "k_0":"d",
            "gamma":{"num":0, "unit":"e"},
            "Cdl":{"num":0, "unit":"e"},
            "alpha":{"num":2, "unit":"f"},
            "Ru":"d"}
fig, axis=plt.subplots(2,3)
keys=list(var_list.keys())
for i in range(0, len(keys)):
    ax=axis[i//3, i%3]
    twinx=ax.twinx()
    ax.set_title(title_dict[keys[i]])
    for j in range(0, len(var_list[keys[i]])):
        param_dict={"E_0":0.241, "k_0":50, "Cdl":1e-5, "gamma":1e-11, "alpha":0.55, "Ru":100}
        #param_dict["E_0"]+=240e-3
        param_dict[keys[i]]=var_list[keys[i]][j]
        param_dict["Cdl"]*=param_list["area"]
        circ_params=[param_dict[x] for x in laviron.optim_list]
        sim_vals=laviron.simulate(circ_params, fitting_frequencies)
        param_dict["Cdl"]/=param_list["area"]
        #param_dict["E0_mean"]-=240e-3
        param_dict["E_0"]-=240e-3
        td_param_list=[param_dict[x] if x in laviron.optim_list else free_params[x] for x in td.optim_list]
        td_vals=td.simulate(td_param_list, frequencies)
        value=var_list[keys[i]][j]
        is_integer = scale_dict[keys[i]]=="d"

        if is_integer==True:
            formatted_string = f"{value:{scale_dict[keys[i]]}}"+" "+unit_dict[keys[i]]
        else:
            formatted_string = f"{value:.{scale_dict[keys[i]]["num"]}{scale_dict[keys[i]]["unit"]}}"+" "+unit_dict[keys[i]]
        EIS().bode(sim_vals,frequencies,ax=ax, twinx=twinx, data_type="phase_mag", compact_labels=True, label=formatted_string)
        EIS().bode(td_vals,frequencies,ax=ax, twinx=twinx, data_type="phase_mag", line=False, scatter=1, compact_labels=True, alpha=0.5)
    ax.legend()
fig.set_size_inches(14, 9)
plt.subplots_adjust(top=0.937,
                    bottom=0.091,
                    left=0.028,
                    right=0.945,
                    hspace=0.367,
                    wspace=0.284)
fig.savefig("TD_comparisons_nodispersion.png", dpi=500)#
from PIL import Image
save_path="TD_comparisons_nodispersion.png"
fig.savefig(save_path, dpi=500)
img = Image.open(save_path)
basewidth = float(img.size[0])//2
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((int(basewidth),hsize), Image.LANCZOS)
img.save(save_path, "PNG", quality=95, dpi=(500, 500))

