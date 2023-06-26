import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser, EIS_genetics
from heuristic_class import Laviron_EIS
from circuit_drawer import circuit_artist
from MCMC_plotting import MCMC_plotting
from pandas import read_csv
import pints
rows=5
cols=4
mplot=MCMC_plotting()






colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
mode="error"
circuit_1={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}
norm_potential=0.001
#RTF=(self.R*self.T)/((self.F**2))
circuit_2={ "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}
circuit_3={ "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
F=96485.3321
R=8.3145
T=298
RT=R*T
FRT=F/(R*T)
k0=10
e0=0.001
Cdl=1e-4
alpha=0.55
gamma=1e-10
area=0.07
dc_pot=0
Ru=100
lav_params={"k_0":k0, "Ru":Ru, "Cdl":Cdl, "gamma":gamma, "E_0":e0, "alpha":alpha, "area":area, "DC_pot":dc_pot}

fitted_params=["Ru", "alpha", "k_0", "Cdl"]
param_list={
       "E_0":e0,
        'E_start':  -5e-3, #(starting dc voltage - V)
        'E_reverse':5e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': area, #(electrode surface area cm^2)
        'Ru': Ru,  #     (uncompensated resistance ohms)
        'Cdl': Cdl, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': gamma,
        "original_gamma":gamma,        # (surface coverage per unit area)
        'k_0': k0, #(reaction rate s-1)
        'alpha': alpha,
        "sampling_freq":1/50,
        "cpe_alpha_faradaic":1
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
    "EIS_Cdl":"C",
    "DC_pot":0,
    "data_representation":"bode",
    "invert_imaginary":False
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
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-9],
    'k_0': [1e-9, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1]
}

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru"])
ru =[0.1, 100, 1000]
fig, ax=plt.subplots()
twinx=ax.twinx()
min_f=-3
max_f=6
points_per_decade=10
frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
freq=[10**x for x in frequency_powers]
"""for r_val in ru:
    param_list["Ru"]=r_val
    vals=[6.283535156769571e-10, 0.01661333703252547, 4.396661844128331e-05, 0.4050375837459084, 100.13774817558738]
    #[param_list[x] for x in laviron.optim_list]
    sim1=laviron.simulate(vals, freqs)
    EIS().bode(sim1, freqs, data_type="phase_mag", ax=ax, twinx=twinx)
plt.show()"""

circs=[mark_circuit]
plot="both"
            
current_file=np.load("BV_param_scans_for_laviron_skipping.npy", allow_pickle=True).item()
keys=list(current_file.keys())
keys=['k_0', 'gamma', 'Cdl', 'Ru', 'alpha']
results_dict={key:{"scale":[], "error":[], "value":[]} for key in keys}

init_scaling_results={'k_0': 0.1595093370010348, 'gamma': 6.198458557728953, 'Cdl': 0.4431452262433833, 'Ru': 1, 'alpha': 1}

#titles=mplot.get_titles(keys, units=False)
#units=mplot.get_units(keys)
for i in range(0, len(keys)):
    copy_params=copy.deepcopy(lav_params)
        #if plot=="both":
        #        twinx[i,z].set_axis_off()
    key=keys[i]
    param_vals=list(current_file[key].keys())
    for j in range(0, len(param_vals)):
            """ax=axis[i,j]
            if "both" in plot:
                twinx=twinxis[i][j]
            else:
                twinx=None"""
            val=param_vals[j]
            fitting_data=current_file[key][val]["data"]
            freq=current_file[key][val]["freq"]*2*np.pi
            phase=fitting_data[:,0]
            mag=fitting_data[:,1]
            copy_params[key]=float(val)
            print(copy_params)
            #print("="*30, circs[z], val)
            #for keyz in fitted_sim_vals:
            #    print(keyz, fitted_sim_vals[keyz])
            laviron.simulation_options["label"]="cmaes"
            
            cmaes_problem=pints.MultiOutputProblem(laviron,freq, np.column_stack((phase, mag)))
            score = pints.GaussianLogLikelihood(cmaes_problem)
            lower_bound=np.append(np.zeros(len(laviron.optim_list)), [0,0,])
            upper_bound=np.append(np.ones(len(laviron.optim_list)), [50, 50])
            CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)

            best_score=-1e6
            print(key, val)
            counter=0
            while best_score<0:
                x0=list(np.random.rand(len(laviron.optim_list)))+[5,5]
                print(len(x0), len(laviron.optim_list), cmaes_problem.n_parameters())
                cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, laviron.n_parameters()+2)], boundaries=CMAES_boundaries, method=pints.CMAES)
                cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
                laviron.simulation_options["test"]=False
                cmaes_fitting.set_parallel(False)
                found_parameters, found_value=cmaes_fitting.run()   
                counter+=1

                if found_value>best_score:
                    
                    dim_params=laviron.change_norm_group(found_parameters[:-2], "un_norm")
                    fitted_vals=dict(zip(laviron.optim_list, dim_params))
                    print(fitted_vals)
                    sim=laviron.simulate(found_parameters[:-2], freq)
                    best_score=found_value
                    fig1, ax1=plt.subplots()
                    twinx1=ax1.twinx()
                    EIS().bode( fitting_data,freq, data_type="phase_mag", data_is_log=True, ax=ax1, twinx=twinx1, scatter=True, compact_labels=True, label="Time domain")
                    EIS().bode( sim,freq, data_type="phase_mag", ax=ax1, twinx=twinx1, scatter=True, compact_labels=True, label="Equivalent circuit")
                    plt.show()
                  
                if counter==10:
                    best_score=1
                    #print(sim)
                    fig1, ax1=plt.subplots()
                    twinx1=ax1.twinx()
                    print(list(found_parameters))
                    print(key, val)
                    print(list(dim_params))
                    print(laviron.optim_list)
                    EIS().bode( fitting_data,freq, data_type="phase_mag", data_is_log=True, ax=ax1, twinx=twinx1)
                    EIS().bode( sim,freq, data_type="phase_mag", ax=ax1, twinx=twinx1)
                    plt.show()
                    
            for z in range(0, len(keys)):
                pred_value=fitted_vals[keys[z]]
               
                error=100*abs(pred_value-copy_params[keys[z]])/copy_params[keys[z]]
                print(pred_value, keys[z], error)
                results_dict[keys[z]]["error"].append(error)
                results_dict[keys[z]]["value"].append(pred_value)
            """if mode=="bode" or plot=="both":
                    if z==0:
                            EIS().bode(fitting_data, freq, data_type="phase_mag", ax=ax, twinx=twinx, compact_labels=True, data_is_log=False, lw=2, alpha=0.65,type=plot)
                    EIS().bode(sim, freq, ax=ax, twinx=twinx, data_type="phase_mag",compact_labels=True, type=plot)
                    ax.set_title("{0}={1} {2}".format(titles[i], val, units[i]))"""
np.save("scaling_errors_expanded.npy",results_dict)  
   