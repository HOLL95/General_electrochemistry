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
from EIS_TD import EIS_TD
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)
truncate=10
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])
EIS().bode(np.column_stack((real, imag)), frequencies)
plt.show()
spectra=np.column_stack((real, imag))


DC_val=0
param_list={
       "E_0":0,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':1,  #    (frequency Hz)
        "original_omega":1,
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 1e-6, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "k0_scale":0.1, 
        "k0_shape":100,
        "cpe_alpha_cdl":1,
        "phase":0,
        "E0_mean":DC_val,
        "E0_std":0.02,
        "cap_phase":0,
        "num_peaks":10,
        "Cdl_std":1e-5,
        "Cdl_mean":1e-5
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
 
    "data_representation":"bode",
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'E0_mean':[-0.1, 0.1],
    'E0_std':[1e-3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 300],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-5], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5e-10],
    'k_0': [40, 250], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[0, 2*math.pi],
    "cap_phase":[0, 2*math.pi],
    "k0_shape":[0,200],
    "k0_scale":[0,2],
    
}
import copy

td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
fitting_frequencies=np.multiply(frequencies, 2*math.pi)
td.simulation_options["label"]="cmaes"
td.simulation_options["data_representation"]="bode"
td.def_optim_list(["E0_mean", "E0_std", "k_0","Ru","Cdl","gamma","phase", "cap_phase","alpha"])
data_to_fit=EIS().convert_to_bode(spectra)
cmaes_problem=pints.MultiOutputProblem(td,fitting_frequencies,data_to_fit)
score = pints.GaussianLogLikelihood(cmaes_problem)
lower_bound=np.append(np.zeros(len(td.optim_list)), [0]*td.n_outputs())
upper_bound=np.append(np.ones(len(td.optim_list)), [50]*td.n_outputs())
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
td.other_values["secret_data"]=spectra
for i in range(0, 5):
    x0=list(np.random.rand(len(td.optim_list)))+[5]*td.n_outputs()
    print(len(x0), len(td.optim_list), cmaes_problem.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, td.n_parameters()+td.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
    td.simulation_options["eis_test"]=False
    cmaes_fitting.set_parallel(False)
    found_parameters, found_value=cmaes_fitting.run()   
    real_params=td.change_norm_group(found_parameters[:-td.n_outputs()], "un_norm")

    print(dict(zip(td.optim_list, list(real_params))))
sim_data=td.simulate(found_parameters[:-td.n_outputs()], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx)
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, data_type="phase_mag")
ax.set_title("No Cdl_f fit")
plt.show()
