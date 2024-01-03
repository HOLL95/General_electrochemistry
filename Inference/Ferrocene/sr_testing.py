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
harm_range=list(range(1,11))
volt_data=voltage_data_file[0::dec_amount, 1]
h_class=harmonics(harm_range,9.036368906531866, 0.05)
dec_amounts=[16]

plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}
fig, ax=plt.subplots(h_class.num_harmonics, 1)
for i in range(0, len(dec_amounts)):
    curr_dict=plot_dict
    for key in curr_dict:
        curr_dict[key]=decimate(curr_dict[key], dec_amounts[i])

    #h_class.plot_harmonics(curr_dict["time"], exp_time_series=curr_dict["current"], plot_func=abs, hanning=True, xaxis=curr_dict["potential"], DC_component=True, axes_list=ax)
    #for axis in ax:
    #    axis.axvline(0.257, color="black", linestyle="--")
    #plt.show()
for sr in [0.032, 0.034, 0.036]:
    param_list={
        "E_0":-0.3,
        'E_start':  -0.2255903087049669, #(starting dc voltage - V)
        'E_reverse': 0.6665871839451643,
        'omega':9.349514676883269, #8.88480830076,  #    (frequency Hz)
        "v":sr,#0.0338951299038171,#0.03348950985573435,
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

    time_series_params=[0.209, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, 8.794196510802587, 0, 0, 0.5999998004431891]
    #time_series_params=[0.2591910307724134, 0.0674086382052161, 74.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*fac, -0.0004999993064740369*fac, 2.5653514370132974e-05*fac, 6.037508022415195e-11, 8.794196510802587, 0, 0, 0.5999998004431891]
    #time_series_params2=[0.25722005928627006, 0.06349029977195912, 40.62521741991397, 6.398882388982527, 0.00033901089088824246, -0.09634378924422059*0, -0.00046147096602321033*0, 2.0914895093075516e-05*0, 6.018367690406668e-11, 8.794196510802587, 0, 0, 0.5999999930196368]
    #time_series_params2=[0.26529520788795874, 0.07380041402223325, 170.55569944276579, 82.90357514867121, 5.5758437147010745e-11, 8.796288136294965]
    #timeseries_params2=[0.25919950037331174, 0.06805648311088913, 198.75368572912876, 88.07303113568388, 0.00034823891990810207, 0.02420675936492858, -0.0004999982190308525, 2.561143319297091e-05, 6.061113052117431e-11, 9.01505671602343, 5.5965024217788315, 4.957512738068864, 0.5999999603113891]


    psv_best_dict=dict(zip(ferro.optim_list, time_series_params))
    sim=ferro.i_nondim(ferro.test_vals(time_series_params, "timeseries"))
    #optim_params=["E0_mean", "E0_std","k_0","Ru","gamma","omega"]
    #ferro.def_optim_list(optim_params)

    #sim2=ferro.i_nondim(ferro.test_vals(time_series_params2, "timeseries"))
    #plt.plot(sim), sim2_time_series=sim2 #data_time_series=ferro.i_nondim(current_results)
    plot_args=dict(sim_time_series=sim  ,hanning=False, plot_func=abs,axes_list=ax, )#

    h_class.plot_harmonics(ferro.t_nondim(time_results), **plot_args)
plt.show()