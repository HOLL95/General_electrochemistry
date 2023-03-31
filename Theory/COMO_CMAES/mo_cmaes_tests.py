import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-2] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(source_loc)
import math
import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
import cma, comocma
import time
harm_range=list(range(1, 9))
param_list={
        "E_0":-0.2,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':100e-3,
        'omega':10,  #    (frequency Hz)
        "original_omega":10,
        'd_E': 300e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 250,  #     (uncompensated resistance ohms)
        'Cdl': 5e-5, #(capacitance parameters)
        'CdlE1': 0.000653657774506,
        'CdlE2': 0.000245772700637,
        "CdlE3":-1e-6,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 10, #(reaction rate s-1)
        'alpha': 0.5,
        "E0_mean":0.2,
        "E0_std": 0.09,
        "cap_phase":3*math.pi/2,
        "alpha_mean":0.5,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/400),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 30,
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":time_start,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "cmaes",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
    'k_0': [0.1, 1e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[param_list['E_start'],param_list['E_reverse']],
    "E0_std": [1e-5,  0.1],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [0, 2*math.pi],
    "all_freqs":[1e-3, 100],
    "all_amps":[1e-5, 0.5],
    "all_phases":[0, 2*math.pi],
}
psv=single_electron("", param_list, simulation_options, other_values, param_bounds)
class_dict={"psv":psv}
alteration_list={"param":{"object":param_list, "dcv":{"original_omega":"", "omega":0,  "v":100e-3}},
                "sim":{"object":simulation_options, "dcv":{"method":"dcv", "test":True}},
                "other":{"object":other_values, "dcv":{"time_start":False}}, 
                "bounds":{"object":param_bounds, "dcv":{}}}
extra_methods=["dcv"]
update_class={key:"" for key in alteration_list.keys()}
for i in range(0, len(extra_methods)):
    for key in alteration_list.keys():
        update_object=copy.deepcopy(alteration_list[key]["object"])
        for key_2 in alteration_list[key][extra_methods[i]].keys():
            value=alteration_list[key][extra_methods[i]][key_2]
            if value=="":
                if key_2 in update_object:
                    del update_object[key_2]
            else:
                update_object[key_2]=value
        update_class[key]=update_object
    print(update_class["sim"])
    class_dict[extra_methods[i]]=single_electron("", update_class["param"], update_class["sim"], update_class["other"], update_class["bounds"])
p=class_dict["psv"]
start=time.time()
long_current=p.test_vals([], "timeseries")
print(time.time()-start)
param_list["num_peaks"]=2+p.nd_param.nd_param_dict["sampling_freq"]
psv=single_electron("", param_list, simulation_options, other_values, param_bounds)
start=time.time()
short_current=psv.test_vals([], "timeseries")
short_current_extended=np.append(short_current, [short_current for x in range(0, 28)])
print(time.time()-start)
print(len(short_current_extended), len(long_current), len(p.time_vec[p.time_idx]))
plt.scatter(p.time_vec[p.time_idx], long_current)
plt.scatter(p.time_vec[p.time_idx],short_current_extended[:-1])
h_class=harmonics(range(1, 8), 1, 0.05)
h_class.plot_harmonics(p.time_vec[p.time_idx], normal_time_series=long_current, replicated_time_series=short_current_extended[:-1], xaxis=p.define_voltages(transient=True))
#plt.plot(short_current_extended)
#plt.plot(long_current)
print(p.RMSE(long_current, short_current_extended[:-1]))
#plt.axhline(short_current[0])
#plt.axhline(short_current[1])
#plt.axhline(short_current[2])
plt.show()

fig, ax=plt.subplots(1, 2)
class_keys=list(class_dict.keys())
optim_list=["E_0", "k_0", "gamma", "Cdl", "Ru", "CdlE1", "CdlE2", "CdlE3", "alpha"]
vals=[param_list[x] for x in optim_list]
objectives={}
print(class_dict["dcv"].simulation_options["test"])
for i in range(0, len(class_keys)):
    current_class=class_dict[class_keys[i]]
    current_class.def_optim_list(optim_list)
    
    current=current_class.test_vals(vals, "timeseries", test=False)
    potential=current_class.define_voltages(transient=True)
    
    objectives[class_keys[i]]=current_class.add_noise(current, 0.01*max(current))
    current_class.secret_data_time_series=objectives[class_keys[i]]
    ax[i].plot(potential, objectives[class_keys[i]])
plt.show() 
class_dict["dcv"].simulation_options["test"]=False
dimension=3
num_kernels=5
sigma=0.1
"""def sphere(x):
    return np.sum(np.asarray(x)**2+np.asarray(x))
list_of_solvers = comocma.get_cmas(num_kernels * [dimension * [0]], 0.2) # produce `num_kernels cma instances`
moes = comocma.Sofomore(list_of_solvers, [11,11]) # create a como-cma-es instance


fitness = comocma.FitFun(sphere, lambda x: sphere(x-1)) # a callable bi-objective function
moes.optimize(fitness, iterations=10)
print(moes.archive)

print(moes.pareto_front_cut)
print(vars(moes.logger))
moes.logger.plot_front()
plt.show()
"""

RMSE=class_dict["psv"].RMSE
fit_funs=[lambda x:RMSE(class_dict["psv"].simulate(np.abs(x), []), objectives["psv"]),
                         lambda x:RMSE(class_dict["dcv"].simulate(np.abs(x), []), objectives["dcv"])]
fitness = comocma.FitFun(*fit_funs) 
random_vals=class_dict["psv"].change_norm_group(np.random.rand(len(optim_list)), "un_norm")
random_vals=class_dict["psv"].change_norm_group(vals, "norm")
random_vals=np.random.rand(len(optim_list))
print(list(random_vals))
list_of_solvers = comocma.get_cmas(num_kernels * [random_vals], 0.1)
rand_init=[x(random_vals) for x in fit_funs]

print(rand_init, "+"*20)
moes = comocma.Sofomore(list_of_solvers, reference_point=rand_init)
moes.optimize(fitness, iterations=1000)
moes.logger.plot_front()
print(moes.archive)
plt.show()