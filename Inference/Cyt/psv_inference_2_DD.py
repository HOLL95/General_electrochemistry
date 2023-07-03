
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

data_loc="/home/user/Documents/Experimental_data/7_6_23/Text_files/PSV_text"
file_name="PSV_500_100_DD_DEC_cv_"
blank_file="Blank_PGE_50_mVs-1_DEC_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
volt_data=voltage_data_file[:, 1]

h_class=harmonics(list(range(4, 9)),8.8475, 0.1)
h_class.plot_harmonics(current_data_file[:,0], farad_time_series=current_data_file[:,1],  xaxis=volt_data, plot_func=np.imag)
plt.show()
fig,ax =plt.subplots(1,1)
h_class.plot_ffts(current_data_file[:,0], current_data_file[:,1], ax=ax,  plot_func=abs)

plt.show()
param_list={
    "E_0":-0.3,
    'E_start':  min(volt_data[len(volt_data)//4:3*len(volt_data)//4]), #(starting dc voltage - V)
    'E_reverse':max(volt_data[len(volt_data)//4:3*len(volt_data)//4]),
    'omega':9.013630669831166, #8.88480830076,  #    (frequency Hz)
    "original_omega":9.013630669831166,
    'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
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
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),
    "experiment_time": current_data_file[:,0],
    "experiment_current": current_data_file[:, 1],
    "experiment_voltage":volt_data,
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.3,0.3],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],2.5*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
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
cyt=single_electron(None, param_list, simulation_options, other_values, param_bounds)




del current_data_file
del voltage_data_file
time_results=cyt.other_values["experiment_time"]
current_results=cyt.other_values["experiment_current"]
voltage_results=cyt.other_values["experiment_voltage"]
h_class=harmonics(other_values["harmonic_range"],1, 0.05)
current=cyt.test_vals([], "timeseries")
plt.plot(cyt.secret_data_fourier)
plt.show()
cyt.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
cyt.dim_dict["alpha"]=0.5
vals=[-0.3403576203057195, 0.03345299856337525, 26.62893724795758, 203.77762900790842, 9.505015320798037e-06, 0.09802559279315218, 0.09911932121943812, 0.0023921732975998727, 4.999999894560245e-11, 9.015294856429078, 4.390524760328755, 4.3613096488276915, 0.5999993611472638]
vals=[-0.3322103735054666, 0.099999999968551, 106.99327954186367, 71.5634998596017, 1.9999998146645936e-05, -0.09999999947868252, -0.02399493062401674, -0.0008706974291918315, 1.164136551651557e-10, 9.015049902227965, 5.395798821699693, 5.210659866188017, 0.400000002693071]
vals=[-0.3235422429210039, 0.0999999995574518, 96.66237266990319, 49.4141315707467, 9.999700576833107e-05, -0.1498437469686061, -0.007907806379936225, -0.00022932119418286184, 1.1943059048895418e-10, 9.015052044193897, 5.53225110499901, 5.273923188038943, 0.4000000312109022]


test=cyt.test_vals(vals, "timeseries")
fourier_arg=cyt.top_hat_filter(current_results)



h_class.plot_harmonics(time_results, exp_time_series=current_results, blank_time_series=test, xaxis=voltage_results)
plt.show()
fig, ax=plt.subplots()
h_class.plot_ffts(time_results, current_results, ax=ax,  plot_func=abs, log=True)
h_class.plot_ffts(time_results, test,ax=ax, plot_func=abs, log=True)
plt.show()
if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
    #plt.plot(fourier_arg)
    #plt.show()
cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=False
score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
num_runs=20
for i in range(0, num_runs):
    x0=abs(np.random.rand(cyt.n_parameters()))#
    #x0=cyt.change_norm_group(vals, "norm")
    print(x0)
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.05 for x in range(0,cmaes_problem.n_parameters())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
    print(list(cmaes_results))
