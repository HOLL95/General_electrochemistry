

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pints

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
from harmonics_plotter import harmonics
import numpy as np

data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_PSV_GC-Green_Fc_cv_"
blank_file="Blank_PGE_50_mVs-1_DEC_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
dec_amount=16
volt_data=voltage_data_file[0::dec_amount, 1]

#blank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
#h_class=harmonics(list(range(4, 9)),9.013630669831166, 0.05)
#h_class.plot_harmonics(current_data_file[:,0], farad_time_series=current_data_file[:,1], blank_time_series=blank_data_current[:,1], xaxis=volt_data, plot_func=np.imag)
#plt.show()
#fig,ax =plt.subplots(1,1)
#h_class.plot_ffts(current_data_file[:,0], current_data_file[:,1], ax=ax, harmonics=h_class.harmonics, plot_func=np.real)
#h_class.plot_ffts(current_data_file[:,0], blank_data_current[:,1],ax=ax, harmonics=h_class.harmonics, plot_func=np.real)
#plt.show()
param_list={
    "E_0":0.25,
    'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
    'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
    'omega':9.365311521736066, #8.88480830076,  #    (frequency Hz)
    "original_omega":9.365311521736066,
    'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.05,
    "E0_skew":0.2,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :3*math.pi/2,
    "time_end": -1,
    'num_peaks': 30,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["original_omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[16],
    "GH_quadrature":True,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "top_hat_return":"composite",
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(4,12,1)),
    "experiment_time": current_data_file[0::dec_amount,0],
    "experiment_current": current_data_file[0::dec_amount, 1],
    "experiment_voltage":volt_data,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0.15, 0.32],
    'omega':[0.9*param_list['omega'],1.1*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 500],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[0.15, 0.3],
    "E0_std": [1e-4,  0.08],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
}
import copy
copied_other=copy.deepcopy(other_values)
copied_sim=copy.deepcopy(simulation_options)
copied_params=copy.deepcopy(param_list)
ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)



#copied_other["experiment_current"]=blank_data_current[:,1]
#copied_other["experiment_time"]=blank_data_current[:,0]
#copied_other["experiment_voltage"]=volt_data
#blank=single_electron(None, copied_params, copied_sim, copied_other, param_bounds)
del current_data_file
del voltage_data_file
time_results=ferro.other_values["experiment_time"]
current_results=ferro.other_values["experiment_current"]
voltage_results=ferro.other_values["experiment_voltage"]
h_class=harmonics(other_values["harmonic_range"],1, 0.5)
current=ferro.test_vals([], "timeseries")

ferro.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
test1=ferro.test_vals([param_list[x] for x in ferro.optim_list], "timeseries")
plt.plot(ferro.define_voltages(no_transient=True), test1)
plt.plot(voltage_results, current_results)
plt.show()
plt.plot(time_results, voltage_results)
plt.plot(time_results, ferro.define_voltages(no_transient=True))
plt.show()
fourier_arg=ferro.top_hat_filter(current_results)



h_class.plot_harmonics(time_results, exp_time_series=current_results, xaxis=voltage_results)
plt.show()
#fig, ax=plt.subplots()

if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(ferro, time_results, current_results)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(ferro, dummy_times, fourier_arg)
    #plt.plot(fourier_arg)
    #plt.show()
ferro.simulation_options["label"]="cmaes"
ferro.simulation_options["test"]=False
score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(ferro.optim_list))), list(np.ones(len(ferro.optim_list))))
num_runs=8
for i in range(0, num_runs):
    x0=abs(np.random.rand(ferro.n_parameters()))#
    #x0=ferro.change_norm_group(vals, "norm")
    print(x0)
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0,cmaes_problem.n_parameters())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(True)
    #cmaes_fitting.set_log_to_file(filename=save_file, csv=False)
    found_parameters, found_value=cmaes_fitting.run()
    
    cmaes_results=ferro.change_norm_group(found_parameters[:], "un_norm")
    #f=open(save_file, "a")
    #f.write("["+(",").join([str(x) for x in cmaes_results])+"]")
    #f.close()
    cmaes_time=ferro.test_vals(cmaes_results, likelihood="fourier", test=False)
    print(list(cmaes_results))
