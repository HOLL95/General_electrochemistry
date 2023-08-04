
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
from heuristic_class import Laviron_EIS
import numpy as np
import pints
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=-0.2850
frequencies=np.flip(data[:,2])*2*np.pi
param_list={
       "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
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
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "EIS_Cf":"CPE",
    "EIS_Cdl":"CPE",
    "DC_pot":DC_val,
    "data_representation":"bode",
    "invert_imaginary":False,
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[-180, 180],
}

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'Q2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}
modified_best={'gamma': 2.04527100556774e-09, 'k_0': 6.445392840569325, 'Cdl': 0.00010679950323186673, 'alpha': 0.45012511579076264, 'Ru': 92.34006406385043, 'cpe_alpha_cdl': 0.5744311931903479, 'cpe_alpha_faradaic': 0.2254001895528145, "Cfarad":1e-5}
modified_best_cfcpe={'gamma': 9.862877626235459e-11, 'k_0': 335.5819847756206, 'Cdl': 3.834939023436789e-06, 'alpha': 0.6462278438069322, 'Ru': 106.15130276664075, 'cpe_alpha_cdl': 0.547032600942984, 'cpe_alpha_faradaic': 0.7912685404285047}
cf_ec={'R0': 106.15130276664075, 'R1': 229.7520961513602, 'C1': 3.834939023436789e-06, 'Q2': 0.0002441733303642724, 'alpha2': 0.7912685404285047}
modified_best_cdlcpe={'gamma': 2.0452710068414738e-09, 'k_0': 6.445392860387815, 'Cdl': 0.00010679950285178704, 'alpha': 0.6455286284473415, 'Ru': 92.34006414011617, 'cpe_alpha_cdl': 0.5744311935370501, 'cpe_alpha_faradaic': 0.901233996346046}
cdl_ec={'R0': 92.34006414011617, 'R1': 576.8481573858269, 'Q1': 0.00010679950285178704, 'alpha1': 0.5744311935370501, 'C2': 0.00013448043367606354}
modified_best_both={'gamma': 3.51921068905404e-10, 'k_0': 52.275222539383904, 'Cdl': 5.643182816623441e-05, 'alpha': 0.5873572734614309, 'Ru': 95.33508461892154, 'cpe_alpha_cdl': 0.6514911462242302, 'cpe_alpha_faradaic': 0.8744399496768777}
both_ec={'R0': 95.33508461892154, 'R1': 413.35253647557875, 'Q1': 5.643182816623441e-05, 'alpha1': 0.6514911462242302, 'Q2': 0.00018832625102362517, 'alpha2': 0.8744399496768777}
modified_best_neither={'gamma': 3.069616091890602e-09, 'k_0': 7.747287787538096, 'Cdl': 5.005944779664158e-06, 'alpha': 0.6095578924032948, 'Ru': 109.74154304677644, 'cpe_alpha_cdl': 0.8384601599285455, 'cpe_alpha_faradaic': 0.141027078168957}
neither_ec={'R0': 109.74154304677644, 'R1': 319.7628554608886, 'C1': 5.005944779664158e-06, 'C2': 0.0002018330587367815}
window_params={
    "1":modified_best_both,
    "5":{'gamma': 1.7525053267940106e-10, 'k_0': 109.27532081864334, 'Cdl': 5.6500820890938345e-05, 'alpha': 0.6086254434046491, 'Ru': 94.81552013641031, 'cpe_alpha_cdl': 0.6488646445451568, 'cpe_alpha_faradaic': 0.8344498347862688},
    "10":{'gamma': 9.329420627295972e-11, 'k_0': 222.2825839979554, 'Cdl': 4.805677069750373e-05, 'alpha': 0.5664056975600678, 'Ru': 96.08847172364777, 'cpe_alpha_cdl': 0.6690873188838039, 'cpe_alpha_faradaic': 0.7992882047764807},
    "15":{'gamma': 2.2493609698466272e-11, 'k_0': 1066.0277550983167, 'Cdl': 3.4399830866899994e-05, 'alpha': 0.5635505378541217, 'Ru': 98.3046029222622, 'cpe_alpha_cdl': 0.7111794124518027, 'cpe_alpha_faradaic': 0.7278632452536045}


}
window_c_params={
    "1":modified_best_cdlcpe,
    "15":{'gamma': 1.0607958802240322e-11, 'k_0': 1999.9999999999986, 'Cdl': 0.0009222863801474816, 'alpha': 0.620979123534349, 'Ru': 25.531898932230494, 'cpe_alpha_cdl': 0.2258700105433444, 'cpe_alpha_faradaic': 0.9844192795497908},
    "10":{'gamma': 1.585482798581747e-09, 'k_0': 7.468371030073875, 'Cdl': 0.00014730018440412001, 'alpha': 0.649645311459301, 'Ru': 85.67927422566684, 'cpe_alpha_cdl': 0.5242925899552799, 'cpe_alpha_faradaic': 0.010446729049520057}

}
#sim_data=sim_class.test_vals(best, frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx,  label="Data", scatter=1)
if simulation_options["EIS_Cf"]=="C2":
    plot_params=window_c_params
else:
    plot_params=window_params
for window_val in plot_params.keys():
    
    window=int(window_val)

    
    sim_data=laviron.simulate([plot_params[window_val][x] for x in laviron.optim_list], frequencies)

   
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, data_type="phase_mag", label=window)
frequencies=np.flip(data[:,2])*2*np.pi

plt.show()
