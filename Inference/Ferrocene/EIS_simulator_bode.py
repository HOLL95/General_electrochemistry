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
from heuristic_class import Laviron_EIS
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from pints.plot import trace
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-1/Fc/Exported"
file_name="2023-09-13_EIS_GC-1_Fc_Nyquist_1"
data=np.loadtxt(data_loc+"/"+file_name)
truncate=10
truncate_2=10
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])



param_list={
        "E_0":0.29,
        'E_start':  0.29-10e-3, #(starting dc voltage - V)
        'E_reverse':0.29+10e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 7e-11,
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
    "dispersion_bins":[1000],
    "GH_quadrature":False,
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "EIS_Cf":"C",
    "EIS_Cdl":"CPE",
    "DC_pot":240e-3,
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
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "k0_shape":[0,10],
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
}
import copy

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["E_0","k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
cdl_only={'E_0': 0.24664324205647736, 'k_0': 1.994718355230308, 'gamma': 3.759801157409028e-09, 'Cdl': 3.4463434909037714e-06, 'alpha': 0.3710871009927439, 'Ru': 150.6110608145238, 'cpe_alpha_cdl': 0.8182725135590561, 'cpe_alpha_faradaic': 0.7115624686878474}

cpe_both={'E_0': 0.24973418891564086, 'k_0': 40.0836335638347, 'gamma': 1.2641498960640705e-10, 'Cdl': 4.6263514007055425e-05, 'alpha': 0.4888807987098013, 'Ru': 116.83438715058305, 'cpe_alpha_cdl': 0.6038787155261925, 'cpe_alpha_faradaic': 0.8281817909445726}

cpe_cf={'E_0': 0.348450277027355, 'k_0': 300.30538922957714, 'gamma': 1.106779971557708e-10, 'Cdl': 1.5374783002917477e-05, 'alpha': 0.6087149844538163, 'Ru': 142.5202086735933, 'cpe_alpha_cdl': 0.4487677974230545, 'cpe_alpha_faradaic': 0.6430147019643657}
cpe_k0_disp={'E_0': 0.2399999998696411, 'k0_shape': 2.636206495399338, 'k0_scale': 0.00532832159171903, 'gamma': 1e-07, 'Cdl': 1.0331317546571832e-05, 'alpha': 0.5091625977733464, 'Ru': 139.62081860568944, 'cpe_alpha_cdl': 0.5775985809929987, 'cpe_alpha_faradaic': 0.026967042291477804}
#cdl_only={'E_0': 0.3489562004369635, 'k0_shape': 0.5297304984569937, 'k0_scale': 1989.7781970025053, 'gamma': 9.076215409899688e-11, 'Cdl': 0.0002403502574521676, 'alpha': 0.6054526137094676, 'Ru': 99.24426363766956, 'cpe_alpha_cdl': 0.37914121456801614, 'cpe_alpha_faradaic': 0.5559041613014697}
#cdl_only={'E_0': 0.3005874856739448, 'k0_shape': 1.6308336689233327, 'k0_scale': 624.3054966879358, 'gamma': 1.1576082556467667e-10, 'Cdl':0.00042492199009813855, 'alpha': 0.35501606255646395, 'Ru': 70.49815289790472, 'cpe_alpha_cdl': 0.20404228545932523, 'cpe_alpha_faradaic': 0.03032224348525079}
#cdl_only={'E_0': 0.1791277628361468, 'k0_shape': 1.3305224411425132, 'k0_scale': 693.1909448507198, 'gamma': 7.544399030717744e-11, 'Cdl': 0.00036399016401596717, 'alpha': 0.4068417670156117, 'Ru': 73.4402049151401, 'cpe_alpha_cdl': 0.25130622019468585, 'cpe_alpha_faradaic': 0.7701653061721752, 'phase': -1.9512078470287406}

frequencies=np.flip(data[truncate:-truncate_2,2])
spectra=np.column_stack((real, imag))
#EIS().bode(spectra, frequencies)
#plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)

sim_data=laviron.simulate([cdl_only[x] for x in laviron.optim_list], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
ax.legend()
ax.set_title("C fit")
plt.show()
laviron.simulation_options["label"]="cmaes"
laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)
cmaes_problem=pints.MultiOutputProblem(laviron,fitting_frequencies,data_to_fit)
score = pints.GaussianLogLikelihood(cmaes_problem)
lower_bound=np.append(np.zeros(len(laviron.optim_list)), [0]*laviron.n_outputs())
upper_bound=np.append(np.ones(len(laviron.optim_list)), [50]*laviron.n_outputs())
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
for i in range(0, 1):
    x0=list(np.random.rand(len(laviron.optim_list)))+[5]*laviron.n_outputs()
    print(len(x0), len(laviron.optim_list), cmaes_problem.n_parameters())
    cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, laviron.n_parameters()+laviron.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
    cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
    laviron.simulation_options["test"]=False
    cmaes_fitting.set_parallel(True)
    found_parameters, found_value=cmaes_fitting.run()   
    real_params=laviron.change_norm_group(found_parameters[:-laviron.n_outputs()], "un_norm")

    print(dict(zip(laviron.optim_list, list(real_params))))
sim_data=laviron.simulate(found_parameters[:-laviron.n_outputs()], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx)
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, data_type="phase_mag")
ax.set_title("CPE fit")
plt.show()
laviron.simulation_options["label"]="MCMC"
MCMC_problem=pints.MultiOutputProblem(laviron,frequencies,data_to_fit)
updated_lb=[param_bounds[x][0] for x in laviron.optim_list]+([0]*laviron.n_outputs())
updated_ub=[param_bounds[x][1] for x in laviron.optim_list]+([100]*laviron.n_outputs())

updated_b=[updated_lb, updated_ub]
updated_b=np.sort(updated_b, axis=0)

log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
#log_prior=pints.MultivariateGaussianLogPrior(mean, np.multiply(std_vals, np.identity(len(std_vals))))
print(log_liklihood.n_parameters(), log_prior.n_parameters())
log_posterior=pints.LogPosterior(log_liklihood, log_prior)
real_param_dict=dict(zip(laviron.optim_list, real_params))

mcmc_parameters=np.append([real_param_dict[x] for x in laviron.optim_list], [found_parameters[-laviron.n_outputs():]])#[laviron.dim_dict[x] for x in laviron.optim_list]+[error]
print(mcmc_parameters)
#mcmc_parameters=np.append(mcmc_parameters,error)
xs=[mcmc_parameters,
    mcmc_parameters,
    mcmc_parameters
    ]


mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)
laviron.simulation_options["test"]=False
mcmc.set_parallel(True)
mcmc.set_max_iterations(20000)


chains=mcmc.run()
trace(chains)
plt.show()