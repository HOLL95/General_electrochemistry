import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from pints import plot
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
from input_design import Input_optimiser
import numpy as np
import matplotlib.pyplot as plt
import sys
harm_range=list(range(4, 6))
from scipy import interpolate
from SALib.sample import saltelli
from SALib.analyze import sobol
param_list={
    "E_0":-0.2,
    'E_start':  -600e-3, #(starting dc voltage - V)
    'E_reverse':-100e-3,
    'omega':8.88480830076,  #    (frequency Hz)
    "v":200e-3,
    'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0.000653657774506,
    'CdlE2': 0.000245772700637,
    "CdlE3":-1e-6,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.09,
    "cap_phase":3*math.pi/2,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/200),
    'phase' :3*math.pi/2,
    "time_end": None,
    'num_peaks': 5,
}
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
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
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
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
    "all_freqs":[1e-5, 2000],
    "all_amps":[1e-5, 0.5],
    "all_phases":[0, 2*math.pi],
}

sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
rpotential=sim.e_nondim(sim.define_voltages())
rcurrent=sim.test_vals([], "timeseries")
#plt.plot(rpotential)
#plt.show()
sim.simulation_options["method"]="sum_of_sinusoids"
#E_0 needs to be defined relative to the input parameters!
simulation_params=["E_0", "k_0", "Ru", "gamma", "alpha"]

num_freqs=3
freq_params=[]
labels=["freq", "amp", "phase"]

for i in range(0, num_freqs):          
    freq_params+=[x+"_{0}".format(i+1) for x  in labels]
freq_values=np.zeros(len(freq_params))
rands=np.random.rand(len(freq_values))
for i in range(0, len(freq_params)):
    appropriate_key="all_{0}s".format(freq_params[i][:freq_params[i].index("_")])
    freq_values[i]=sim.un_normalise(rands[i], [param_bounds[appropriate_key][0], param_bounds[appropriate_key][1]])
sim.def_optim_list(freq_params)
sim.test_vals(freq_values, "params")
pot=sim.e_nondim(sim.define_voltages())
param_bounds["E_0"]=[min(pot), max(pot)]
problem =   {
            "num_vars":len(simulation_params),
            "names":simulation_params, 
            "bounds":[[param_bounds[x][0], param_bounds[x][1]] for x in simulation_params]
}
all_params=simulation_params+freq_params
print(all_params)
dim=1024
sample_values=saltelli.sample(problem, dim)
len_sample_values=len(sample_values)
sim.def_optim_list(all_params)
from SALib.test_functions import Ishigami
"""problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]]
}
X = saltelli.sample(problem, 512)
Y = Ishigami.evaluate(X)
plt.plot(Y)
plt.show()
Si = sobol.analyze(problem, Y, print_to_console=False)
print(Si["S1"], sum(Si["S1"]))
print(Si["ST"], sum(Si["ST"]))"""
time_series_vector=np.zeros(len_sample_values)
time=sim.t_nondim(sim.time_vec)

highest_freq=sim.max_freq
print(highest_freq)
print(sim.nd_param.nd_param_dict["freq_array"])
print(sim.dim_dict["freq_array"])

simulation_options["sobol_params"]=simulation_params
simulation_options["num_sinusoids"]=3
simulation_options["sobol_dim"]=dim
plt.plot(pot)
print(list(freq_values))
plt.show()
des=Input_optimiser(param_list, simulation_options, other_values, param_bounds)
print(des.sobol_simulate(freq_values))


for i in range(0, len_sample_values):
    #print(i)
    electro_params=sample_values[i, :]
    
    all_params=np.append(electro_params, freq_values)
    #print(list(all_params))
    current=sim.test_vals(all_params, "timeseries")
    if i==0:
        ts_len=len(current)-1
        time_series_matrix=np.zeros((len_sample_values,ts_len))
    #time_series_vector[i*ts_len:(i+1)*ts_len]=current
    #time_series_vector[i]=current[-1]
    #
    #plt.plot(time, current)
    #pot=sim.e_nondim(sim.define_voltages())
    
    #print(list(freq_values))
    #plt.plot(time, pot)
    #plt.show()
    time_series_matrix[i, :]=current[1:] #row is parameter variation, column is timepoints
#plt.show()
sobol_1=np.zeros(( len(simulation_params), ts_len))
variance=np.zeros(ts_len)
for i in range(0, ts_len):
    
    Si=sobol.analyze(problem, time_series_matrix[:,i])#Calculating sobol indices for every parameter (row), over iterative timepoints (col) 
    
    sobol_1[:,i]=Si["S1"]
    negs=np.where(Si["S1"]<0)

    if len(negs[0])>0:
        for j in range(0, len(negs[0])):
            if (Si["S1"][negs[0][j]]+Si["S1_conf"][negs[0][j]])>0:
                sobol_1[negs[0][j], i]=1e-20
                
            else:
                sobol_1[:,i]=1e-20
                break
   
sobol_mean=np.mean(sobol_1, axis=1)
total_entropy=1/(np.sum(np.multiply(sobol_mean, np.log(sobol_mean))))
osc_points=200
num_intervals=ts_len//osc_points
if num_intervals%osc_points!=0:
    extra_tw=True
    time_window_sobol=np.zeros((len(simulation_params), num_intervals+1))
else:
    extra_tw=False
    time_window_sobol=np.zeros((len(simulation_params), num_intervals))

for i in range(0, num_intervals):
    for j in range(len(simulation_params)):
        sobol_sum=np.sum(sobol_1[j, i*osc_points:(i+1)*osc_points])#row is parameter, column is num_intervals
        time_window_sobol[j,i]=sobol_sum*np.log(sobol_sum)
if extra_tw==True:
    for j in range(len(simulation_params)):
        sobol_sum=np.sum(sobol_1[j, num_intervals*osc_points:])
        time_window_sobol[j,-1]=sobol_sum*np.log(sobol_sum)
time_window_sobol=np.sum(np.sum(time_window_sobol, axis=0))
total_var=1/(np.sum(np.sqrt(np.std(time_series_matrix, axis=1)))) #variance over the columns
print(total_entropy,time_window_sobol, total_var)

"""
print(sobol_1)
sobol_sum_i=np.sum(sobol_1, axis=1) #sum over every timepoint for each parameter (summing over col)
print(sobol_sum_i)
total_horizon_entropy=-1/(np.sum(np.multiply(sobol_sum_i, np.log(sobol_sum_i))))
total_var=1/(np.sum(np.sqrt(np.std(time_series_matrix, axis=1)))) #variance over the columns
osc_points=200
num_intervals=ts_len//osc_points
if num_intervals%osc_points!=0:
    extra_tw=True
    time_window_sobol=np.zeros((len(simulation_params), num_intervals+1))
else:
    extra_tw=False
    time_window_sobol=np.zeros((len(simulation_params), num_intervals))

for i in range(0, num_intervals):
    for j in range(len(simulation_params)):
        sobol_sum=np.sum(sobol_1[j, i*osc_points:(i+1)*osc_points])
        time_window_sobol[j,i]=sobol_sum*np.log(sobol_sum)
if extra_tw==True:
    for j in range(len(simulation_params)):
        sobol_sum=np.sum(sobol_1[j, num_intervals*osc_points:])
        time_window_sobol[j,-1]=sobol_sum*np.log(sobol_sum)
time_window_sobol=-np.sum(np.sum(time_window_sobol, axis=1))
print(total_horizon_entropy, total_var, time_window_sobol, 1/time_window_sobol)
"""



