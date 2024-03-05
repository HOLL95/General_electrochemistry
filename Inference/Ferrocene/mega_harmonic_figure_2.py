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
from multiplotter import multiplot
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"

file_name="2023-10-10_FTV_GC-Green_Fc_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")

#sblank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
file_name="2023-10-10_FTV_GC-Green_Fc_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
figure=multiplot(3, 1, **{"harmonic_position":0, "num_harmonics":4, "orientation":"portrait", "fourier_position":1, "plot_width":5, "row_spacing":1, "plot_height":1})
#sblank_data_current=np.loadtxt(data_loc+"/"+blank_file+"current")
harm_range=list(range(4,8))
ramped_h_class=harmonics(harm_range,9.036368906531866, 0.05)
dec_amount=8

volt_data=voltage_data_file[0::dec_amount, 1]

plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}





curr_dict=plot_dict
for key in curr_dict:
    curr_dict[key]=decimate(curr_dict[key], dec_amount)
ramped_h_class.get_freq(curr_dict["time"], curr_dict["current"])



file_name="2023-10-10_PSV_GC-Green_Fc_cv_"
blank_file="Blank_PGE_50_mVs-1_DEC_cv_"
psv_current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
psv_voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
dec_amount=16
psv_volt_data=psv_voltage_data_file[0::dec_amount, 1]

psv_param_list={
    "E_0":0.25,
    'E_start':  min(psv_volt_data[len(psv_volt_data)//4:3*len(psv_volt_data)//4]), #(starting dc voltage - V)
    'E_reverse':max(psv_volt_data[len(psv_volt_data)//4:3*len(psv_volt_data)//4]),
    'omega':9.015057685643711, #8.88480830076,  #    (frequency Hz)
    "original_omega":9.015057685643711,
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

solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(psv_param_list["original_omega"])
psv_simulation_options={
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

psv_other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(4,8,1)),
    "experiment_time": psv_current_data_file[0::dec_amount,0],
    "experiment_current": psv_current_data_file[0::dec_amount, 1],
    "experiment_voltage":psv_volt_data,
    "bounds_val":20000,
}

param_bounds={
    'E_0':[-0.1, 0.1],
    'omega':[8,10],#8.88480830076,  #    (frequency Hz)
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
ferro=single_electron(None, psv_param_list, psv_simulation_options, psv_other_values, param_bounds)

time_results=ferro.t_nondim(ferro.other_values["experiment_time"])
current_results=ferro.i_nondim(ferro.other_values["experiment_current"])
voltage_results=ferro.e_nondim(ferro.other_values["experiment_voltage"])
#fig, ax=plt.subplots()
#ax.plot(voltage_results, current_results)
#plt.show()
psv_h_class=harmonics(harm_range, 9.015057685643711, 0.05)
#psv_h_class.get_freq(time_results, current_results)
psv_h_class.plot_harmonics(time_results, Data_time_series=current_results*1e6, plot_func=np.real, axes_list=figure.axes_dict["col1"][:psv_h_class.num_harmonics], xaxis=voltage_results)
ferro.def_optim_list(["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
frequency_params=[0.2591910307724134, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509, -0.0004999993064740369, 2.5653514370132974e-05, 6.037508022415195e-11, 9.015057685643711, 5.58768403688611, 4.964330246307874, 0.5999998004431891]
psv_prediction=ferro.i_nondim(ferro.test_vals(frequency_params, "timeseries"))
psv_h_class.plot_harmonics(time_results, Fitted_time_series=psv_prediction*1e6, plot_func=np.real, axes_list=figure.axes_dict["col1"][:psv_h_class.num_harmonics], xaxis=voltage_results, xlabel="AC potential (V)", ylabel="Current ($\\mu$A)", legend={"bbox_to_anchor":[0.5, 1.2], "ncols":2, "frameon":False, "loc":"center"})









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
    'k_0': 1000, #(reaction rate s-1)
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
    "no_transient":False,
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

import copy

ramped=single_electron(None, param_list, simulation_options, other_values, param_bounds)
ramped.def_optim_list(["E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"])
fac=1e-2
#0.2591910307724134
#0.247
#0.235
#0.215
#0.209
e0_list=[0.247, 0.247, 0.238, 0.237 ]

time_series_params1=[0.2591910307724134, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, ramped_h_class.input_frequency, 0, 0, 0.5999998004431891]

sim=ramped.i_nondim(ramped.test_vals(time_series_params1, "timeseries"))

pot=ramped.e_nondim(ramped.define_voltages(transient=False))
for j in range(1, 3):
    axes_list=figure.axes_dict["col1"][j*ramped_h_class.num_harmonics:(j+1)*ramped_h_class.num_harmonics]
    
    ramped_h_class.plot_harmonics(curr_dict["time"], exp_time_series=curr_dict["current"]*1e6, plot_func=abs, hanning=True,  axes_list=axes_list, legend=None,xaxis=pot, DC_component=True,xlabel="DC potential (V)", ylabel="Current ($\\mu$A)")#
ramped_h_class.get_freq(ramped.t_nondim(ramped.time_vec[ramped.time_idx]), sim)
ramped_h_class.plot_harmonics(curr_dict["time"], Translated_time_series=sim*1e6, plot_func=abs, axes_list=figure.axes_dict["col1"][psv_h_class.num_harmonics:psv_h_class.num_harmonics*2],hanning=True, legend=None,xaxis=pot, DC_component=True,)#xaxis=pot, DC_component=True, 
 


time_series_params=time_series_params1#[0.2515085054963522, 0.0637810584632682, 62.915075289229755, 109.99988420501067, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.334048572909808e-11, ramped_h_class.input_frequency, 0, 0, 0.5999998004431891]






start_harm=4
for i in range(0, len(e0_list)):
    time_series_params[0]=e0_list[i]
    sim=ramped.i_nondim(ramped.test_vals(time_series_params, "timeseries"))
    new_h_class=harmonics(list(range(start_harm+i, start_harm+i+1)), 1, 0.05)
    #print(new_h_class.harmonics, new_h_class.num_harmonics)
    #print([figure.axes_dict["col1"][i+psv_h_class.num_harmonics*2]])
    new_h_class.get_freq(ramped.t_nondim(ramped.time_vec[ramped.time_idx]), sim)
    
    new_h_class.plot_harmonics(ramped.t_nondim(ramped.time_vec[ramped.time_idx]), exp_time_series=sim*1e6, plot_func=abs, hanning=True, legend=None,axes_list=[figure.axes_dict["col1"][i+psv_h_class.num_harmonics*2]],xaxis=pot, DC_component=True, )# 
   

    #h_class.plot_harmonics(ferro.t_nondim(time_results), current_time_series=current_results,simulated_time_series=sim, hanning=True, plot_func=abs)
letters=["(A)", "(B)", "(C)"]
for i in range(0, 3):
    ax=figure.axes_dict["col1"][i*ramped_h_class.num_harmonics]
    
    ax.text(-0.2, 1.2, letters[i], transform=ax.transAxes, fontweight="bold", fontsize=12)


fig=plt.gcf()
plt.subplots_adjust(top=0.97,
bottom=0.075,
left=0.16,
right=0.9,
hspace=0.2,
wspace=0.2)
fig.set_size_inches(4.5, 9)
plt.show()
fig.savefig("monster_harmonics.png", dpi=500)

