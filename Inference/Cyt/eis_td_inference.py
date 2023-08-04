
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
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
import numpy as np
import pints
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=0
frequencies=np.flip(data[:,2])
param_list={
       "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "k0_shape":0.4,
        "k0_scale":10,
        "phase":0,
        "E0_mean":DC_val,
        "E0_std":0.02,
        "cap_phase":0,
        "num_peaks":20
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
 
    "data_representation":"bode",
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'E0_mean':[-0.05, 0.05],
    'E0_std':[1e-3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[0, 2*math.pi],
    "k0_shape":[0,2],
    "k0_scale":[0,1e4],
    "cap_phase":[0, 2*math.pi],
}

td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
td.def_optim_list(["E0_mean", "E0_std","gamma","k_0",  "Cdl", "alpha", "Ru", "phase", "cap_phase"])
#sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'Q2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}

#sim_data=sim_class.test_vals(best, frequencies)

#vals={'E_0': -0.05658612595335928, 'gamma': 2.1135822672930202e-10, 'k_0': 732.6257332769766, 'Cdl': 0.009528224698906936, 'alpha': 0.6276141120760577, 'Ru': 630.2057050888834, 'phase': 4.874263497804321, 'cap_phase': 1.7393978664951508}
#vals={'E_0': -0.002527085503396448, 'gamma': 9.999999991636322e-10, 'k_0': 7.430155005355703, 'Cdl': 0.0014769232751485476, 'alpha': 0.35001320906342553, 'Ru': 179.00612624018376, 'phase': 1.5595575772292523, 'cap_phase': 1.629669571069265}
vals={'E_0': 0.0004991258872459603, 'gamma': 9.999999872234705e-10, 'k_0': 1.0722444537125324, 'Cdl': 0.0025924641719568863, 'alpha': 0.3502678097795446, 'Ru': 224.09400307362233, 'phase': 4.428570087924558, 'cap_phase': 4.282159295254902}
#vals={'E0_mean': -0.10000001371553463, 'E0_std': 0.08284543062023773, 'gamma': 9.999999793318866e-10, 'k_0': 1.3868578637114732, 'Cdl': 0.007177203744744742, 'alpha': 0.5869065385285961, 'Ru': 252.92273045081183, 'phase': 1.3881992332762259, 'cap_phase': 1.5764576275344302}
vals={'E_0': 0.06402002478772581, 'gamma': 9.610657703479629e-09, 'k_0': 2.900794007956364, 'Cdl': 0.00013739079410114144, 'alpha': 0.6471164621307125, 'Ru': 117.84351447031456, 'phase': 2.201833784425848e-05, 'cap_phase': 6.28316810297355}
fig,ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx)
for k0 in [7.350150390584221]:
    vals={"E0_mean":DC_val, 'E0_std': 0.07302817320569106, 'gamma': 3.1143856228599335e-09, 'k_0': k0, 'Cdl': 6.816905229705668e-05, 'alpha': 0.3500000044804436, 'Ru': 108.85079519283647, 'phase': 5.749299141832945, 'cap_phase': 5.77388626067403}
    sim_vals=[vals[x] for x in td.optim_list]
    
    k0_vars=td.simulate(sim_vals,  frequencies)
    EIS().bode(k0_vars, frequencies, ax=ax, twinx=twinx, data_type="phase_mag", label=k0)
plt.show()
#sim_vals=[1e-3,1e-10, 10, 1e-5, 0.55, 100, 0,0]
practice=td.simulate(sim_vals,  frequencies)
print(practice)
laviron_circuit={"z1":"R0", "z2":{"p_1":["R1", ("Q2", "alpha2")], "p_2":"C1"}}
circuit_sim=EIS(circuit=laviron_circuit)
equiv_fit=circuit_sim.test_vals({'R0': 106.15130291843991, 'R1': 229.75209631047858, 'C1': 3.834939035094845e-06, 'Q2': 0.00024417333059895694, 'alpha2': 0.7912685393555589}, frequencies=frequencies*2*np.pi)
fig, ax=plt.subplots(1,2)
twinx=ax[0].twinx()
EIS().bode(practice, frequencies, ax=ax[0], twinx=twinx, data_type="phase_mag")
EIS().bode(fitting_data, frequencies, ax=ax[0], twinx=twinx)
EIS().bode(equiv_fit, frequencies, ax=ax[0], twinx=twinx)
EIS().nyquist(practice, ax=ax[1], orthonormal=False)
EIS().nyquist(fitting_data, ax=ax[1], orthonormal=False)
EIS().nyquist(equiv_fit, ax=ax[1], orthonormal=False)
plt.show()
td.def_optim_list(["E0_mean", "E0_std","gamma", "Cdl", "alpha", "Ru", "phase", "cap_phase"])

data_to_fit=EIS().convert_to_bode(fitting_data)
td.simulation_options["label"]="cmaes"
all_data=np.column_stack((fitting_data, data_to_fit))
td.other_values["secret_data"]=data_to_fit
cmaes_problem=pints.MultiOutputProblem(td,frequencies,data_to_fit)


score = pints.GaussianLogLikelihood(cmaes_problem)
lower_bound=np.append(np.zeros(len(td.optim_list)), [0]*td.n_outputs())
upper_bound=np.append(np.ones(len(td.optim_list)), [50]*td.n_outputs())
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
for i in range(0, 10):
    print(td.simulation_options["dispersion_parameters"]*20)
    x0=list(np.random.rand(len(td.optim_list)))+[5]*td.n_outputs()
    #x0=td.change_norm_group(sim_vals, "norm")+[5]*td.n_outputs()
    print(len(x0), len(td.optim_list), cmaes_problem.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, td.n_parameters()+td.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
    td.simulation_options["eis_test"]=False
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()   
    real_params=td.change_norm_group(found_parameters[:-td.n_outputs()], "un_norm")
    #real_params=sim_class.change_norm_group(dict(zip(names, found_parameters[:-2])), "un_norm", return_type="dict" )
    print(dict(zip(td.optim_list, list(real_params))))
    
sim_data=td.simulate(found_parameters[:-td.n_outputs()], frequencies)


fig, ax=plt.subplots(1,2)
twinx=ax[0].twinx()
EIS().bode(fitting_data, frequencies, ax=ax[0], twinx=twinx, compact_labels=True)
EIS().bode(sim_data, frequencies,ax=ax[0], twinx=twinx, compact_labels=True)
EIS().nyquist(fitting_data, ax=ax[1], orthonormal=False)
EIS().nyquist(sim_data, ax=ax[1], orthonormal=False)


plt.show()

