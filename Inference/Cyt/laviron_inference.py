
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
from heuristic_class import Laviron_EIS
import numpy as np
import pints
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=-0.2850
frequencies=np.flip(data[:,2])*2*np.pi
param_list={
       "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
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
    "method": "dcv",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "EIS_Cf":"C",
    "EIS_Cdl":"CPE",
    "DC_pot":DC_val,
    "data_representation":"bode",
    "invert_imaginary":False
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[param_list['E_start'],param_list['E_reverse']],
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
    "phase":[-180, 180],
}

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'Q2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}
modified_best={'gamma': 2.04527100556774e-09, 'k_0': 6.445392840569325, 'Cdl': 0.00010679950323186673, 'alpha': 0.45012511579076264, 'Ru': 92.34006406385043, 'cpe_alpha_cdl': 0.5744311931903479, 'cpe_alpha_faradaic': 0.2254001895528145}

#sim_data=sim_class.test_vals(best, frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx)
for k0_vals in [6.445392840569325, 10, 15]:
    modified_best["k_0"]=k0_vals
    sim_data=laviron.simulate([modified_best[x] for x in laviron.optim_list], frequencies)

   
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, data_type="phase_mag")
    ax.set_title("me")
plt.show()
names=laviron.optim_list
print(names)

data_to_fit=EIS().convert_to_bode(fitting_data)
laviron.simulation_options["label"]="cmaes"
laviron.simulation_options["bode_split"]=None
if laviron.simulation_options["bode_split"]=="magnitude":
    cmaes_problem=pints.SingleOutputProblem(laviron,frequencies,data_to_fit[:,1])
elif laviron.simulation_options["bode_split"]=="phase":
    cmaes_problem=pints.SingleOutputProblem(laviron,frequencies,data_to_fit[:,0])
else:
    cmaes_problem=pints.MultiOutputProblem(laviron,frequencies,data_to_fit)

score = pints.GaussianLogLikelihood(cmaes_problem)
lower_bound=np.append(np.zeros(len(laviron.optim_list)), [0]*laviron.n_outputs())
upper_bound=np.append(np.ones(len(laviron.optim_list)), [50]*laviron.n_outputs())
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
x0=list(np.random.rand(len(laviron.optim_list)))+[5]*laviron.n_outputs()
print(len(x0), len(laviron.optim_list), cmaes_problem.n_parameters())
cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, laviron.n_parameters()+laviron.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
laviron.simulation_options["test"]=False
cmaes_fitting.set_parallel(True)
found_parameters, found_value=cmaes_fitting.run()   
real_params=laviron.change_norm_group(found_parameters[:-laviron.n_outputs()], "un_norm")
#real_params=sim_class.change_norm_group(dict(zip(names, found_parameters[:-2])), "un_norm", return_type="dict" )
print(dict(zip(names, list(real_params))))
dd_both_best={'gamma': 8.008619084925477e-11, 'k_0': 209.39397173851847, 'Cdl': 5.573903494143972e-06, 'alpha': 0.648949680392418, 'Ru': 170.95621012741807, 'cpe_alpha_faradaic': 0.8234625988158906, 'cpe_alpha_cdl': 0.45511111861319575}

sim_data=laviron.simulate(found_parameters[:-laviron.n_outputs()], frequencies)
if laviron.n_outputs()==2:
    laviron.simulation_options["data_representation"]="nyquist"
    sim_data=laviron.simulate(found_parameters[:-laviron.n_outputs()], frequencies, print_circuit_params=True)
    fig, ax=plt.subplots(1,2)
    twinx=ax[0].twinx()
    EIS().bode(fitting_data, frequencies, ax=ax[0], twinx=twinx, compact_labels=True)
    EIS().bode(sim_data, frequencies,ax=ax[0], twinx=twinx, compact_labels=True)
    EIS().nyquist(fitting_data, ax=ax[1], orthonormal=False)
    EIS().nyquist(sim_data, ax=ax[1], orthonormal=False)
    laviron.simulation_options["data_representation"]="bode"
else:
    plot_freq=np.log10(frequencies)/2*np.pi
    if laviron.simulation_options["bode_split"]=="magnitude":
        plt.plot(plot_freq,data_to_fit[:,1])
    elif laviron.simulation_options["bode_split"]=="phase":
         plt.plot(plot_freq,data_to_fit[:,0])
   
    plt.plot(plot_freq, sim_data)
    plt.xlabel("Frequency")
    plt.ylabel(laviron.simulation_options["bode_split"])


plt.show()

laviron.simulation_options["label"]="MCMC"
if laviron.simulation_options["bode_split"]=="magnitude":
    MCMC_problem=pints.SingleOutputProblem(laviron,frequencies,data_to_fit[:,1])
elif laviron.simulation_options["bode_split"]=="phase":
    MCMC_problem=pints.SingleOutputProblem(laviron,frequencies,data_to_fit[:,0])
else:
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
mcmc.set_parallel(False)
mcmc.set_max_iterations(20000)
save_file="EIS_modified_{0}_C".format(laviron.simulation_options["bode_split"])
chains=mcmc.run()
f=open(save_file, "wb")
np.save(f, chains)

trace(chains)
plt.show()