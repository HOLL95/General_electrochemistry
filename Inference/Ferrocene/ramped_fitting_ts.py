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
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from scipy.signal import decimate

data_loc="/home/henry/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"

file_name="2023-10-10_FTV_GC-Green_Fc_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")

#sblank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
h_class=harmonics(list(range(1,11)),9.036368906531866, 0.25)
dec_amount=8
harm_range=list(range(1,9))
volt_data=voltage_data_file[0::dec_amount, 1]
h_class=harmonics(harm_range,9.036368906531866, 0.05)
dec_amounts=[16]

plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}
fig, ax=plt.subplots(h_class.num_harmonics, 1)
for i in range(0, len(dec_amounts)):
    curr_dict=plot_dict
    for key in curr_dict:
        curr_dict[key]=decimate(curr_dict[key], dec_amounts[i])

    h_class.plot_harmonics(curr_dict["time"], exp_time_series=curr_dict["current"], plot_func=abs, hanning=True, xaxis=curr_dict["potential"], DC_component=True, axes_list=ax)
    for axis in ax:
        axis.axvline(0.257, color="black", linestyle="--")
    plt.show()

param_list={
    "E_0":-0.3,
    'E_start':  -225*1e-3, #(starting dc voltage - V)
    'E_reverse':  675*1e-3,
    'omega':9.349514676883269, #8.88480830076,  #    (frequency Hz)
    "v":0.03353,#0.0338951299038171,#0.03348950985573435,
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 7e-11,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000*0, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/400),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 30,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[20],
    "GH_quadrature":False,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "top_hat_return":"abs"
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(4,11,1)),
    "experiment_time": curr_dict["time"],
    "experiment_current": curr_dict["current"],
    "experiment_voltage":curr_dict["potential"],
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.85*param_list['omega'],1.15*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [50, 110],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.3,0.3],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [4.5e-11, 8e-11],
    'k_0': [50, 200], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[0.2115, 0.2585],
    "E0_std": [0.05,  0.075],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0, 2*math.pi],
}
import copy
copied_other=copy.deepcopy(other_values)
copied_sim=copy.deepcopy(simulation_options)
copied_params=copy.deepcopy(param_list)
ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)





time_results=ferro.other_values["experiment_time"]
current_results=ferro.other_values["experiment_current"]
voltage_results=ferro.other_values["experiment_voltage"]
ferro.get_input_freq(ferro.t_nondim(time_results), current_results)
#ferro.get_input_params(ferro.e_nondim(voltage_results), ferro.t_nondim(time_results))
plt.plot(time_results, ferro.e_nondim(voltage_results))
plt.plot(time_results, ferro.e_nondim(ferro.define_voltages(no_transient=True)))
plt.show()



h_class=harmonics(harm_range,8.794196510802587, 0.05)


ferro.def_optim_list(["Cdl","CdlE1", "CdlE2", "CdlE3","omega","cap_phase","phase", ])



    
fourier_arg=ferro.top_hat_filter(current_results)
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(ferro, time_results, current_results)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(ferro, dummy_times, fourier_arg)

ferro.simulation_options["label"]="cmaes"

score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(ferro.optim_list))), list(np.ones(len(ferro.optim_list))))
num_runs=5
ferro.simulation_options["test"]=False

for i in range(0, num_runs):
    x0=abs(np.random.rand(ferro.n_parameters()))#

    #x0=ferro.change_norm_group(init_vals, "norm")
    print(x0)
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.15 for x in range(0,cmaes_problem.n_parameters())], boundaries=CMAES_boundaries, method=pints.CMAES)
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