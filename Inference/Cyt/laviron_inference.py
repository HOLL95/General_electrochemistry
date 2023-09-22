
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
DC_val=-0.2850

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
    "EIS_Cf":"CPE",
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
    'E_0':[param_list['E_start'],param_list['E_reverse']],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[-180, 180],
}
fig, ax=plt.subplots(1,3)
EIS().nyquist(fitting_data, ax=ax[0], scatter=1)
EIS().nyquist(fitting_data, ax=ax[1], scatter=1, orthonormal=False)
EIS().bode(fitting_data, frequencies, ax=ax[2], twinx=ax[2].twinx(), compact_labels=True)
plt.show()
laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)
laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'Q2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}
modified_best={'gamma': 2.04527100556774e-09, 'k_0': 6.445392840569325, 'Cdl': 0.00010679950323186673, 'alpha': 0.45012511579076264, 'Ru': 92.34006406385043, 'cpe_alpha_cdl': 0.5744311931903479, 'cpe_alpha_faradaic': 0.2254001895528145, "Cfarad":1e-5}
modified_best_cfcpe={'gamma': 9.862877626235459e-11, 'k_0': 335.5819847756206, 'Cdl': 3.834939023436789e-06, 'alpha': 0.6462278438069322, 'Ru': 106.15130276664075, 'cpe_alpha_cdl': 0.547032600942984, 'cpe_alpha_faradaic': 0.7912685404285047}
cf_ec={'R0': 106.15130276664075, 'R1': 229.7520961513602, 'C1': 3.834939023436789e-06, 'Q2': 0.0002441733303642724, 'alpha2': 0.7912685404285047}
modified_best_cdlcpe={'gamma': 2.0452710068414738e-09, 'k_0': 6.445392860387815, 'Cdl': 0.00010679950285178704, 'alpha': 0.6455286284473415, 'Ru': 92.34006414011617, 'cpe_alpha_cdl': 0.5744311935370501, 'cpe_alpha_faradaic': 0.901233996346046}
cdl_ec={'R0': 92.34006414011617, 'R1': 576.8481573858269, 'Q1': 0.00010679950285178704, 'alpha1': 0.5744311935370501, 'C2': 0.00013448043367606354}
modified_best_both={'gamma': 3.51921068905404e-10, 'k_0': 52.275222539383904, 'Cdl': 5.643182816623441e-05, 'alpha': 0.5873572734614309, 'Ru': 95.33508461892154, 'cpe_alpha_cdl': 0.6514911462242302, 'cpe_alpha_faradaic': 0.8744399496768777}
both_ec={'R0': 95.33508461892154, 'R1': 413.35253647557875, 'Q1': 5.643182816623441e-05, 'alpha1': 0.6514911462242302, 'Q2': 0.00018832625102362517, 'alpha2': 0.8744399496768777}
modified_best_neither={'gamma': 3.069616091890602e-09, 'k_0': 7.747287787538096, 'Cdl': 5.005944779664158e-06, 'alpha': 0.6095578924032948, 'Ru': 109.74154304677644, 'cpe_alpha_cdl': 0.8384601599285455, 'cpe_alpha_faradaic': 0.141027078168957}
neither_ec={'R0': 109.74154304677644, 'R1': 319.7628554608886, 'C1': 5.005944779664158e-06, 'C2': 0.0002018330587367815}
window_params={
    "1":modified_best_both,
    "5":{'gamma': 1.7525053267940106e-10, 'k_0': 109.27532081864334, 'Cdl': 5.6500820890938345e-05, 'alpha': 0.6086254434046491, 'Ru': 94.81552013641031, 'cpe_alpha_cdl': 0.6488646445451568, 'cpe_alpha_faradaic': 0.8344498347862688},
    "10":{'gamma': 9.329420627295972e-11, 'k_0': 222.2825839979554, 'Cdl': 4.805677069750373e-05, 'alpha': 0.5664056975600678, 'Ru': 96.08847172364777, 'cpe_alpha_cdl': 0.6690873188838039, 'cpe_alpha_faradaic': 0.7992882047764807},
    "15":{'gamma': 2.2493609698466272e-11, 'k_0': 1066.0277550983167, 'Cdl': 3.4399830866899994e-05, 'alpha': 0.5635505378541217, 'Ru': 98.3046029222622, 'cpe_alpha_cdl': 0.7111794124518027, 'cpe_alpha_faradaic': 0.7278632452536045}




}

#sim_data=sim_class.test_vals(best, frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()

for window_val in window_params.keys():
    window=int(window_val)
    
    DC_val=-0.2850
    
    sim_data=laviron.simulate([window_params[window_val][x] for x in laviron.optim_list], frequencies)

   
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, data_type="phase_mag", scatter=1, label=window)
EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx, lw=5, alpha=0.75, label="Data")
plt.show()
laviron.def_optim_list(["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
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
    fig, ax=plt.subplots()
    twinx=ax.twinx()
    EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx, compact_labels=True, label="Data")
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx, compact_labels=True, label="Simulation")
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
save_file="EIS_modified_{0}_C".format(laviron.simulation_options["bode_split"])
chains=mcmc.run()
f=open(save_file, "wb")
np.save(f, chains)

trace(chains)
plt.show()