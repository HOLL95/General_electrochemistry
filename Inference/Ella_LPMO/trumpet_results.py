import matplotlib.pyplot as plt
import math
import os
import sys
import re
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from heuristic_class import DCVTrumpet, DCV_peak_area
import numpy as np
import pints
from pints.plot import trace
files=["Cj", "Cf"]
param_vals=[[0.03077898, 0.41083013, 0.58115891, 0.0500362 ]  ,
[0.03465741, 0.23443141, 0.57578573, 0.02453627]]
param_vals={"Cj":[0.03077898, 0.41083013, 0.58115891, 0.0500362 ],
            "Cf":[0.03465741, 0.23443141, 0.57578573, 0.02453627]}
Ru_vals=[50, 100, 150, 200]

fig, ax=plt.subplots(1,2)
for i in range(0, len(files)):
    for j in range(0, len(Ru_vals)):
        param_list={
            "E_0":0,
            'E_start': -0.2, #(starting dc voltage - V)
            'E_reverse':0.25,
            'omega':0, #8.88480830076,  #    (frequency Hz)
            'd_E': 10*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
            'area': 0.07, #(electrode surface area cm^2)
            "v":22.5e-3,
            'Ru': Ru_vals[j],  #     (uncompensated resistance ohms)
            'Cdl': 1e-5, #(capacitance parameters)
            'CdlE1': 0,#0.000653657774506,
            'CdlE2': 0,#0.000245772700637,
            "CdlE3":0,
            'gamma': 1e-10,
            "original_gamma":1e-10,        # (surface coverage per unit area)
            'k_0': 75, #(reaction rate s-1)
            'alpha': 0.5,
            "E0_mean":0,
            "E0_std": 0.025,
            "E0_skew":0.2,
            "cap_phase":0,
            "alpha_mean":0.5,
            "alpha_std":1e-3,
            'sampling_freq' : (1.0/50),
            'phase' :0,
            "time_end": -1,
            'num_peaks': 10,
            "k0_shape":0.4,
            "k0_scale":75,
            "dcv_sep":0,
            
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
            "dispersion_bins":[32],
            "GH_quadrature":True,
            "test": False,
            "method": "dcv",
            "phase_only":False,
            "likelihood":likelihood_options[0],
            "numerical_method": solver_list[1],
            "label": "MCMC",
            "invert_imaginary":False,
            "Marcus_kinetics":False,
            "optim_list":[],
            
        }

        other_values={
            "filter_val": 0.5,
            "harmonic_range":list(range(3,9,1)),

            "bounds_val":2000,
        }
        param_bounds={
            'E_0':[-0.1, 0.2],
            'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
            'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
            'Cdl': [0,5e-4], #(capacitance parameters)
            'CdlE1': [-0.2,0.2],#0.000653657774506,
            'CdlE2': [-0.1,0.1],#0.000245772700637,
            'CdlE3': [-0.05,0.05],#1.10053945995e-06,
            'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
            'k_0': [0.01, 10], #(reaction rate s-1)
            'alpha': [0.4, 0.8],
            "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
            "E0_mean":[-0.35, -0.25],
            "E0_std": [1e-4,  0.15],
            'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
            "dcv_sep":[0, 0.2]

        }
        trumpets=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
        trumpet_file=np.loadtxt("LPMO_%s_peak_positions.txt"%files[i], delimiter=",")
        scan_rates=trumpet_file[:,0]
        trumpet_positions=np.column_stack((trumpet_file[:,1], trumpet_file[:,2]))
                
            
        in_volts=scan_rates*1e-3
        trumpet_positions=trumpet_positions/trumpets.nd_param.c_E0
        trumpets.secret_data_trumpet=trumpet_positions
        trumpets.def_optim_list(["E_0", "k_0", "alpha", "dcv_sep"])
        #init_vals=[0.014074675370828849, 2.8544294764333173, 0.7416888979810574, 0.04808605753340519]
        #init_vals1=[0.028806806661952566, 1.8910227708278882, 0.7983732003861854, 0.046146272162374935]

        sim=trumpets.simulate(param_vals[i], in_volts, optimise_flag=True)


        ax[i].set_title(files[i])
        if i==1 and j==0:
            
            trumpets.trumpet_plot(in_volts,trumpets.e_nondim(trumpet_positions),  ax=ax[i], label="Data", colour_counter=0)
            trumpets.trumpet_plot( in_volts,trumpets.e_nondim(sim), ax=ax[i], label="Sim",colour_counter=1)
            plt.legend()
        else:
            trumpets.trumpet_plot(in_volts,trumpets.e_nondim(trumpet_positions),  ax=ax[i],  colour_counter=0)
            trumpets.trumpet_plot( in_volts,trumpets.e_nondim(sim), ax=ax[i],colour_counter=1)
        

        
        """#
        #trumpets.simulation_options["label"]="MCMC"
        #trumpets.trumpet_plot(in_volts,trumpets.e_nondim(trumpet_positions),  ax=ax, label="Data")
        #trumpets.trumpet_plot( in_volts,trumpets.e_nondim(sim), ax=ax, label="Sim")
        #plt.legend()
        #plt.show()
        MCMC_problem=pints.MultiOutputProblem(trumpets,in_volts,trumpet_positions)
        updated_lb=[param_bounds[x][0] for x in trumpets.optim_list]+([0]*trumpets.n_outputs())
        updated_ub=[param_bounds[x][1] for x in trumpets.optim_list]+([100]*trumpets.n_outputs())
        updated_b=[updated_lb, updated_ub]
        updated_b=np.sort(updated_b, axis=0)

        log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
        log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
        #log_prior=pints.MultivariateGaussianLogPrior(mean, np.multiply(std_vals, np.identity(len(std_vals))))
        print(log_liklihood.n_parameters(), log_prior.n_parameters())
        log_posterior=pints.LogPosterior(log_liklihood, log_prior)
        real_param_dict=dict(zip(trumpets.optim_list, real_params))

        mcmc_parameters=np.append([real_param_dict[x] for x in trumpets.optim_list], [found_parameters[-trumpets.n_outputs():]])#[trumpets.dim_dict[x] for x in trumpets.optim_list]+[error]
        print(mcmc_parameters)
        #mcmc_parameters=np.append(mcmc_parameters,error)
        xs=[mcmc_parameters,
            mcmc_parameters,
            mcmc_parameters
            ]


        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)
        trumpets.simulation_options["test"]=False
        mcmc.set_parallel(True)
        mcmc.set_max_iterations(10000)
        save_file="MCMC/Ella_set2_MCMC_%sOhms_%s"%(param_list["Ru"], files[i])
        chains=mcmc.run()
        f=open(save_file, "wb")
        np.save(f, chains)
        #trace(chains)
        #plt.show()"""
plt.show()