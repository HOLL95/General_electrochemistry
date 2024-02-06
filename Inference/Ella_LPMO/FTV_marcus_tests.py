import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
import numpy as np
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from multiplotter import multiplot
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
param_list={
        "E_0":- 0.1,
        'E_start':  -0.4, #(starting dc voltage - V)
        'E_reverse':0.3,
        'omega':0,
        "v":10e-3,  #    (frequency Hz)
        'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.5,
        "sampling_freq":1/(200),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "Cfarad":0,
        "E0_mean":0,
        "E0_std": 0.025,
        "k0_shape":0.4,
        "k0_scale":75,
        "num_peaks":10,
        "cap_phase":0,
        "Upper_lambda":1
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
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],

}

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
    "k0_shape":[0,10],
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
    "cap_phase":[-180,190],
    "Upper_lambda":[0, 10]
}

ramped=single_electron("",param_list, simulation_options, other_values, param_bounds)
ramped.def_optim_list(["k_0"])
k_vals=[0.1, 0.5, 1, 10]
#plt.plot(ramped.define_voltages())
#plt.show()
harmonics_list=list(range(1, 6))
h_class=harmonics(harmonics_list, param_list["omega"], 0.25)
figure=multiplot(2, 2, **{"harmonic_position":[0,1], "num_harmonics":h_class.num_harmonics, "orientation":"landscape", "plot_width":5, "row_spacing":2, "plot_height":1})
harmax=figure.axes_dict
fig, axes =plt.subplots(2, 2)

for i in range(0, len(k_vals)):
    row=i//2
    col=i%2
    ax=axes[row, col]
    
    k0=k_vals[i]
    BVcurrent=ramped.i_nondim(ramped.test_vals([k0], "timeseries"))
    BV_time=ramped.t_nondim(ramped.time_vec)
    ax.plot(BV_time, BVcurrent, label="BV")
    
    harmax["row%d"%(row+1)][col*h_class.num_harmonics].set_title("k_0=%.1f"%k0)
    ramped.simulation_options["Marcus_kinetics"]=True
    Mcurrent=ramped.i_nondim(ramped.test_vals([k0], "timeseries"))
    ax.plot(ramped.t_nondim(ramped.time_vec),Mcurrent , label="Marcus")
    plot_dict=dict(Marcus_time_series=Mcurrent*1e6, BV_time_series=BVcurrent*1e6, hanning=False, axes_list=harmax["row%d"%(row+1)][col*h_class.num_harmonics:(col+1)*h_class.num_harmonics], plot_func=np.abs, xlabel="Time(s)", ylabel="Current($\\mu$A)")
    h_class.plot_harmonics(BV_time, **plot_dict)
    ramped.simulation_options["Marcus_kinetics"]=False
    ax.legend()
    ax.set_title("k_0=%.1f"%k0)
plt.show()