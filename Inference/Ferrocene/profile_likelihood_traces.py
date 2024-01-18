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
import pandas as pd
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)
truncate=10
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])
EIS().nyquist(np.column_stack((real, imag)),orthonormal=False)
plt.title("Ferrocene Nyquist plot")
plt.show()


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
        "num_peaks":30,
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
    "C_sim":True,
    "optim_list":[],
    "EIS_Cf":"C",
    "EIS_Cdl":"CPE",
    "DC_pot":240e-3,
    "Rct_only":False,
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0.239, 0.5],
    "E0_mean":[0.239, 0.241],
    "E0_std":[1e-5, 0.05],
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
    "k0_scale":[0,200],
    "phase":[-180, 180],
}
import copy

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
#laviron.def_optim_list(["E_0","k0_shape", "k0_scale", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E0_mean", "E0_std", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
laviron.def_optim_list(["E_0","k0_shape", "k0_scale", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl"])
#laviron.def_optim_list(["E0_mean", "E0_std", "k0_shape","k0_scale", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl"])#"E0_mean","E0_std","k0_scale","k0_shape"
spectra=np.column_stack((real, imag))
EIS().bode(spectra, frequencies)
plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)
EIS_params_5={'E_0': 0.19066872485338204, 'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}
EIS_params_5={'E0_mean': 0.19066872485338204, "E0_std":1e-5, 'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}
#sim_data=laviron.simulate([EIS_params_5[x] for x in laviron.optim_list], fitting_frequencies)
#fig, ax=plt.subplots()
#twinx=ax.twinx()
#EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
#EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
#ax.legend()
#ax.set_title("C fit")
#plt.show()




#laviron.def_optim_list(["E_0","k0_shape", "k
laviron.simulation_options["label"]="cmaes"
laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)
cmaes_problem=pints.MultiOutputProblem(laviron,fitting_frequencies,data_to_fit)
score = pints.SumOfSquaresError(cmaes_problem)
lower_bound=np.append(np.zeros(len(laviron.optim_list)), [0]*laviron.n_outputs())
upper_bound=np.append(np.ones(len(laviron.optim_list)), [50]*laviron.n_outputs())
lower_bound=np.zeros(len(laviron.optim_list))
upper_bound=np.ones(len(laviron.optim_list))
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
titles=laviron.optim_list+["score", "best"]
for i in range(0, 10):
    x0=list(np.random.rand(len(laviron.optim_list)))
    cmaes_fitting=pints.CMAES(x0, sigma0=[0.075 for x in range(0, laviron.n_parameters())], boundaries=CMAES_boundaries)
    e = pints.ParallelEvaluator(score)
    threshold=1e-4
    iterations=200
    running=True
    laviron.simulation_options["test"]=False
    current_best=1e7
    unchanged_iterations=0
    file_name="traces/Profile_trace_%d_k0_disp.csv" % (i+1)
    df = pd.DataFrame(columns=titles)
    file_name_best="traces/Profile_trace_%d_k0_disp_best.csv" % (i+1)
    df_best = pd.DataFrame(columns=titles[:-1])
    counter=0
    best_counter=0
    while running==True:
    #for i in range(0,1):   
        xs=cmaes_fitting.ask()
        #f_xs=[score(x) for x in xs]
        f_xs=e.evaluate(xs)
        cmaes_fitting.tell(f_xs)
        #print(xs)
        #print(f_xs)
        best=cmaes_fitting.f_best()
        for i in range(0, len(f_xs)):
            df.loc[counter]=np.append(laviron.change_norm_group(xs[i], "un_norm"), [f_xs[i], best])
            counter+=1
            
        f_best=cmaes_fitting.f_best()
        if (current_best-f_best)>threshold:
            unchanged_iterations=0
        else:
            unchanged_iterations+=1
        print(f_best, current_best, unchanged_iterations)
        if f_best<current_best:
            
            current_best=f_best
            df_best.loc[best_counter]=np.append(laviron.change_norm_group(cmaes_fitting.x_best(), "un_norm"), [best])
            best_counter+=1
        if unchanged_iterations==iterations:
            running=False
    
    df.set_index(titles[0], inplace=True)
    df.to_csv(file_name)
    df_best.set_index(titles[0], inplace=True)
    df_best.to_csv(file_name_best)
    #print(list(laviron.change_norm_group(best, "un_norm")))