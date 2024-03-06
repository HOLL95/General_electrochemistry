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
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
from harmonics_plotter import harmonics
from multiplotter import multiplot
import numpy as np
import pints
from scipy.signal import decimate
from pints.plot import trace
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
#data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)
figure=multiplot(1,2, **{"harmonic_position":1, "num_harmonics":4, "orientation":"portrait",  "plot_width":5, "row_spacing":1, "plot_height":1})

truncate=10
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])
file_name="2023-10-10_FTV_GC-Green_Fc_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
dec_amount=8

volt_data=voltage_data_file[0::dec_amount, 1]

plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}





curr_dict=plot_dict
for key in curr_dict:
    curr_dict[key]=decimate(curr_dict[key], dec_amount)





DC_val=0
param_list={
        "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':0,
        "v":100e-3,  #    (frequency Hz)
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
        "num_peaks":10,
        "cap_phase":0
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
    "dispersion_bins":[300, 300],
    "GH_quadrature":False,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "C_sim":True,
    "EIS_Cf":"C",
    "EIS_Cdl":"CPE",
    "DC_pot":240e-3,
    "Rct_only":False,
}
DC_pot=240e-3
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0.15, 0.35],
    "E0_mean":[0.15, 0.35],
    "E0_std":[0.001, 0.1],
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
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
    "cap_phase":[-180,190]
}
import copy
ramped_param_list={
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
    'sampling_freq' : (1.0/200),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 30,
    "k0_shape":0.4,
    "k0_scale":75,  
}

solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

ramped_simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":True,
    "dispersion":False,
    "dispersion_bins":[8,120],
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

ramped_other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(4,11,1)),
    "experiment_time": curr_dict["time"],
    "experiment_current": curr_dict["current"],
    "experiment_voltage":curr_dict["potential"],
    "bounds_val":200000,
}

ramped=single_electron(None, ramped_param_list, ramped_simulation_options, ramped_other_values, param_bounds)

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)

ramped_h_class=harmonics(list(range(4, 8)), 1, 0.075)
ramped_h_class.get_freq(curr_dict["time"], curr_dict["current"])
ramped_h_class.plot_harmonics(curr_dict["time"], Data_time_series=curr_dict["current"]*1e6, hanning=True, plot_func=np.abs, axes_list=figure.axes_dict["col2"], xlabel="Time(s)", ylabel="Current ($\\mu A$)", legend=None)
laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic", "Cfarad"])
#laviron.def_optim_list(["E0_mean", "E0_std", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E_0",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E0_mean","E0_std",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
banned_param={"cpe_alpha_cdl", "cpe_alpha_faradaic", "Cfarad"}
get_set=list(set(laviron.optim_list)-banned_param)
ramped_optim_list=[x for x in laviron.optim_list if x in get_set]#+["phase","cap_phase"]

free_params=dict(zip(["phase","cap_phase"], [0,0]))
ramped.def_optim_list(ramped_optim_list)
#"E0_mean","E0_std","k0_scale","k0_shape"
spectra=np.column_stack((real, imag))
#EIS().bode(spectra, frequencies)
#plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)

"""sim_data=laviron.simulate([cdl_only[x] for x in laviron.optim_list], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
ax.legend()
ax.set_title("C fit")
plt.show()

"""

#laviron.simulation_options["label"]="cmaes"
laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)
#C

EIS_params1={'E_0': 0.23708843969139082-DC_val, 'k_0': 4.028523388682444, 'gamma': 7.779384163661676e-10, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#EIS_params1={'E_0': 0.23708843969139082-DC_val, 'k_0': 75, 'gamma': 2e-11, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#CPE

EIS_params2={'E_0': 0.3047451457126534-DC_val, 'k_0': 39.40663787158313, 'gamma': 1.0829517784499947e-10, 'Cdl': 8.7932554621096e-06, 'alpha': 0.5394294479538084, 'Ru': 80.76397847517714,"omega":8.794196510802587}
#CPEonlycdl
EIS_params2={'E_0': 0.24906140557789685, 'k_0': 33.09311167404058, 'gamma': 2.0954984064091433e-10, 'Cdl': 1.4793284257699079e-05, 'alpha': 0.5034734319903937, 'Ru': 75.59649715071465, 'cpe_alpha_cdl': 0.6905831560232432, 'aoo': -1.4755430597507484, 'aor': 0.32944153083222005, 'arr': 0.5728228035542173}
{'E_0': 0.226498182124326, 'k_0': 26.230905642697792, 'gamma': 9.102442673108454e-10, 'Cdl': 1.4793284086919418e-05, 'alpha': 0.5719530288549428, 'Ru': 75.59649738287867, 'cpe_alpha_cdl': 0.6905831576740543, 'aoo': 2.4029057299240435, 'aor': 1.5414169013423518, 'arr': 9.400849465234167}
{'E_0': 0.20229962649812183, 'k_0': 56.783972682665286, 'gamma': 8.283148465118944e-10, 'Cdl': 1.4793284181675208e-05, 'alpha': 0.5626488136556527, 'Ru': 75.59649730224247, 'cpe_alpha_cdl': 0.6905831564551901, 'aoo': -3.0347580138571697, 'aor': 0.7905590705850951, 'arr': 2.2969081606852466}







#CPE but no extra phase
EIS_params_5={'E_0': 0.23901744045809714, 'k0_scale': 1.7516889068612655, 'k0_shape': 1.0429458816816037, 'gamma': 8.725153422918051e-10, 'Cdl': 7.947339387048066e-06, 'alpha': 0.49219004982384884, 'Ru': 81.68485910243592, 'cpe_alpha_cdl': 0.7680042642561522, 'aoo': 7.696052279975657, 'aor': -2.52513986417158, 'arr': -0.3395674462371634}
{'E_0': 0.24801273214229078, 'k0_scale': 1.6534654850918096, 'k0_shape': 1.0429458715415447, 'gamma': 8.935980590604058e-10, 'Cdl': 7.94733943217117e-06, 'alpha': 0.6476417912004915, 'Ru': 81.68485915461436, 'cpe_alpha_cdl': 0.7680042637955333, 'aoo': -9.996571859130027, 'aor': -5.813845370236466, 'arr': 4.0643511569087725}


ax=figure.axes_dict["col1"][0]
twinx=ax.twinx()
#######################HERE######################
param_dict=EIS_params2
#################################################
circ_params=[param_dict[x] for x in laviron.optim_list]
print(circ_params)
sim_vals=laviron.simulate(circ_params, fitting_frequencies)
param_dict["Cdl"]/=param_list["area"]
if "E0_mean" in ramped.optim_list:
    param_dict["E0_mean"]=0.247
elif "E_0" in ramped.optim_list:
    param_dict["E_0"]=0.247
param_dict["omega"]=ramped_param_list["omega"]
sim_params=[param_dict[x] for x in ramped.optim_list]
#fig,axis=plt.subplots()

print(sim_params)
print(ramped.optim_list)
sim=ramped.i_nondim(ramped.test_vals(sim_params, "timeseries"))
#axis.plot(sim)
#plt.show()
times=ramped.t_nondim(ramped.time_vec)
ramped_h_class.get_freq(times, sim)
ramped_h_class.plot_harmonics(times, Fitted_time_series=sim*1e6, hanning=True, plot_func=abs, axes_list=figure.axes_dict["col2"], legend=None)
#param_dict["E0_mean"]-=240e-3
#td_vals=td.simulate(td_param_list, frequencies)

EIS().bode(np.column_stack((real, imag)),frequencies,ax=ax, twinx=twinx, label="Data", compact_labels=True)
EIS().bode(sim_vals,frequencies,ax=ax, twinx=twinx, data_type="phase_mag", label="Fitted", compact_labels=True)
#EIS().bode(td_vals,frequencies,ax=ax, twinx=twinx, data_type="phase_mag", line=False, scatter=1, label="TD")
ax.legend()
plt.subplots_adjust(top=0.897,
bottom=0.13,
left=0.04,
right=0.982,
hspace=0.1,
wspace=0.5)
fig=plt.gcf()
fig.set_size_inches(9, 4.5)
fig.savefig("Laviron_CPE_Cdl.png", dpi=500)
plt.show()
