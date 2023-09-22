
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
window=0
DC_val=0

if window!=0:

    fitting_data=np.column_stack((np.flip(data[window:-window,0]), np.flip(data[window:-window,1])))
    frequencies=np.flip(data[window:-window,2])*2*np.pi
elif window==0:
    fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
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
        "Cfarad":0,
        "E0_mean":0,
        "E0_std": 0.025,
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
    "invert_imaginary":False,
    "Rct_only":False,
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-50e-3, 50e-3],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-6],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "E0_mean":[-100e-3, 100e-3],
    "E0_std": [0,  0.15],
    "phase":[-180, 180],
}

laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
#laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'Q2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}
test_vals={'E0_mean': 0.03505918675088768, 'E0_std': 0.09751190163557545, 'gamma': 1.460035802088556e-07, 'k_0': 0.4027826228475147, 'Cdl': 2.3961871766797967e-05, 'alpha': 0.6499999999999999, 'Ru': 97.25874003110316, 'cpe_alpha_cdl': 0.7426416354902906, 'cpe_alpha_faradaic': 0.889216201979133, 'phase': -0.7281905012158063, "Cfarad":1e-5}

laviron.def_optim_list(["E0_mean", "E0_std","gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic", "phase"])
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

fig, ax=plt.subplots()
twinx=ax.twinx()
laviron.simulation_options["label"]="MCMC"   
EIS().bode(data_to_fit, frequencies, ax=ax, twinx=twinx, lw=2, alpha=0.75, label="Data", data_type="phase_mag", scatter=1)

orig_gamma=test_vals["gamma"]
for gamma_vals in [orig_gamma]:
    test_vals["gamma"]=gamma_vals
    sim_data=laviron.simulate([test_vals[x] for x in laviron.optim_list], frequencies)
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, data_type="phase_mag", label="E0 disp fit", lw=3)

simulation_options["EIS_Cf"]="CPE"
simulation_options["EIS_Cdl"]="CPE"
laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
best_both={'gamma': 3.51921068905404e-10, 'k_0': 52.275222539383904, 'Cdl': 5.643182816623441e-05, 'alpha': 0.5873572734614309, 'Ru': 95.33508461892154, 'cpe_alpha_cdl': 0.6514911462242302, 'cpe_alpha_faradaic': 0.8744399496768777}
sim_data=laviron.simulate([best_both[x] for x in laviron.optim_list], frequencies)
EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, data_type="phase_mag", label="CPE fit", lw=3)
ax.legend()
plt.show()
laviron.simulation_options["label"]="cmaes"  
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
    laviron.simulation_options["data_representation"]="bode"
    sim_data=laviron.simulate(found_parameters[:-laviron.n_outputs()], frequencies, print_circuit_params=True)
    fig, ax=plt.subplots()
    twinx=ax.twinx()
    EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx, compact_labels=True, label="Data")
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, compact_labels=True, label="Simulation", data_type="phase_mag")
    #EIS().nyquist(fitting_data, ax=ax[1], orthonormal=False)
    #EIS().nyquist(sim_data, ax=ax[1], orthonormal=False)
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
save_file="EIS_modified_{0}_C_dispersed".format(laviron.simulation_options["bode_split"])
chains=mcmc.run()
f=open(save_file, "wb")
np.save(f, chains)

trace(chains)
plt.show()