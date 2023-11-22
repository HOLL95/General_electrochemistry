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

data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"

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
    'E_start':  -0.2255903087049669, #(starting dc voltage - V)
    'E_reverse': 0.6665871839451643,
    'omega':9.349514676883269, #8.88480830076,  #    (frequency Hz)
    "v":0.0338951299038171,#0.03348950985573435,
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 1.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 7e-11,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "E0_skew":0.2,
    "k0_scale":1,
    "k0_shape":1,
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
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "experiment_time": curr_dict["time"],
    "experiment_current": curr_dict["current"],
    "experiment_voltage":curr_dict["potential"],
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
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
ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)





time_results=ferro.other_values["experiment_time"]
current_results=ferro.other_values["experiment_current"]
voltage_results=ferro.other_values["experiment_voltage"]
ferro.get_input_freq(ferro.t_nondim(time_results), current_results)
#ferro.get_input_params(ferro.e_nondim(voltage_results), ferro.t_nondim(time_results))
#plt.plot(time_results, voltage_results)
#plt.plot(time_results, ferro.define_voltages(no_transient=True))
#plt.show()



h_class=harmonics(harm_range,8.794196510802587, 0.05)


ferro.def_optim_list(["E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
fac=1e-2
#0.2591910307724134
#0.247
#0.235
#0.215
#0.209
time_series_params=[0.235, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, 8.794196510802587, 0, 0, 0.5999998004431891]
#C
EIS_params1={'E_0': 0.23708843969139082, 'k_0': 4.028523388682444, 'gamma': 7.779384163661676e-10, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#CPE
EIS_params2={'E_0': 0.3047451457126534, 'k_0': 39.40663787158313, 'gamma': 1.0829517784499947e-10, 'Cdl': 8.7932554621096e-06, 'alpha': 0.5394294479538084, 'Ru': 80.76397847517714,"omega":8.794196510802587}
#Cfarad
EIS_params3={'E_0': 0.2161051668499098, 'k_0': 106.6602436491309, 'gamma': 2.5360979030661595e-11, 'Cdl': 8.751614540745486e-06, 'alpha': 0.47965820103670564, 'Ru': 80.92159716231082,"omega":8.794196510802587}

EIS_params_4={'E_0': 0.2014214483444881, 'k0_scale': 1.0950956335756536, 'k0_shape': 1.043401547065882, 'gamma': 1.4645920920242938e-09, 'Cdl': 7.945475589121264e-06, 'alpha': 0.359816590101354, 'Ru': 81.69086207153816, 'cpe_alpha_cdl': 0.768033972041215, 'cpe_alpha_faradaic': 0.9119951540999298, 'phase': -0.20113351781378697,"omega":8.794196510802587} 

EIS_composite=[EIS_params_4]
fig, axes_list=plt.subplots(h_class.num_harmonics, 1)
labels=["k0_disp"]
data_time_series=ferro.i_nondim(current_results),
h_class.plot_harmonics(ferro.t_nondim(time_results),data_time_series=ferro.i_nondim(current_results),hanning=True, plot_func=abs, ax=axes_list)


for i in range(0, len(EIS_composite)):
    ferro.def_optim_list(["E_0","k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "omega"])
    vals=[EIS_composite[i][x] for x in ferro.optim_list]

    psv_best_dict=dict(zip(ferro.optim_list, time_series_params))
    sim=ferro.i_nondim(ferro.test_vals(vals, "timeseries"))
    #optim_params=["E0_mean", "E0_std","k_0","Ru","gamma","omega"]
    #ferro.def_optim_list(optim_params)

    #sim2=ferro.i_nondim(ferro.test_vals(time_series_params2, "timeseries"))
    #plt.plot(sim), ,sim2_time_series=sim2
    plot_args=dict(  hanning=True, plot_func=abs, ax=axes_list)#xaxis=voltage_results, DC_component=True
    plot_args[labels[i]+"_time_series"]=sim
    h_class.plot_harmonics(ferro.t_nondim(time_results), **plot_args)
    #h_class.plot_harmonics(ferro.t_nondim(time_results), current_time_series=current_results,simulated_time_series=sim, hanning=True, plot_func=abs)
plt.show()
