
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

data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/PSV_text"
file_name="PGE_50_mVs-1_DEC_cv_"
blank_file="Blank_PGE_50_mVs-1_DEC_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
volt_data=voltage_data_file[:, 1]
blank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
h_class=harmonics(list(range(4, 9)),9.013630669831166, 0.05)
h_class.plot_harmonics(current_data_file[:,0], farad_time_series=current_data_file[:,1], blank_time_series=blank_data_current[:,1], xaxis=volt_data, plot_func=np.imag)
plt.show()
fig,ax =plt.subplots(1,1)
h_class.plot_ffts(current_data_file[:,0], current_data_file[:,1], ax=ax, harmonics=h_class.harmonics, plot_func=np.real)
h_class.plot_ffts(current_data_file[:,0], blank_data_current[:,1],ax=ax, harmonics=h_class.harmonics, plot_func=np.real)
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
    'gamma': 1e-11,
    "original_gamma":1e-11,        # (surface coverage per unit area)
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
    "harmonic_range":list(range(2,9,1)),
    "experiment_time": current_data_file[:,0],
    "experiment_current": current_data_file[:, 1],
    "experiment_voltage":volt_data,
    "bounds_val":2000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,2e-5], #(capacitance parameters)
    'CdlE1': [-0.1,0.1],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.1],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
}
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
vals=[-0.250025315624648, 0.0999984062475107, 1718.8659085418521, 69.35038477736988, 6.298328141389161e-06, 0.09974784721495897, -0.06363368252272678, 0.021113852771861577, 4.888217261657549e-11, 8.940914234543671, 3.771741905559131, 4.646054257112752, 0.4003390264341337]
test=cyt.test_vals(vals, "timeseries")
fourier_arg=cyt.top_hat_filter(current_results)


plt.plot(cyt.top_hat_filter(test))
#plt.plot(fourier_arg)
plt.show()

if simulation_options["likelihood"]=="timeseries":
    cmaes_problem=pints.SingleOutputProblem(cyt, time_results, true_data)
elif simulation_options["likelihood"]=="fourier":
    dummy_times=np.linspace(0, 1, len(fourier_arg))
    cmaes_problem=pints.SingleOutputProblem(cyt, dummy_times, fourier_arg)
    #plt.plot(fourier_arg)
    #plt.show()
cyt.simulation_options["label"]="cmaes"
cyt.simulation_options["test"]=True
score = pints.SumOfSquaresError(cmaes_problem)
CMAES_boundaries=pints.RectangularBoundaries(list(np.zeros(len(cyt.optim_list))), list(np.ones(len(cyt.optim_list))))
num_runs=20
for i in range(0, num_runs):
    x0=abs(np.random.rand(cyt.n_parameters()))#cyt.change_norm_group(gc4_3_low_ru, "norm")
    print(len(x0), cmaes_problem.n_parameters(), CMAES_boundaries.n_parameters(), score.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.15 for x in range(0,cmaes_problem.n_parameters())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
    cmaes_fitting.set_parallel(False)
    found_parameters, found_value=cmaes_fitting.run()
    cmaes_results=cyt.change_norm_group(found_parameters[:], "un_norm")
    cmaes_time=cyt.test_vals(cmaes_results, likelihood="fourier", test=False)
    print(list(cmaes_results))
