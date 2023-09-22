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

file_loc="/home/henney/Documents/Oxford/Experimental_data/Ella/MWCNT_scan_sig2"

param_list={
    "E_0":0,
    'E_start': -0.5, #(starting dc voltage - V)
    'E_reverse':0.5,
    'omega':0, #8.88480830076,  #    (frequency Hz)
    'd_E': 10*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    "v":22.5e-3,
    'Ru': 100.0,  #     (uncompensated resistance ohms)
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
    'E_0':[-0.5, 0.5],
    'omega':[0.98*param_list['omega'],1.02*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.2,0.2],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [1, 200], #(reaction rate s-1)
    'alpha': [0.4, 0.8],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[-0.35, -0.25],
    "E0_std": [1e-4,  0.15],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "dcv_sep":[0, 0.2]

}
trumpets=DCVTrumpet(param_list, simulation_options, other_values, param_bounds)
files=os.listdir(file_loc)
file_dict={}
for file in files:
    get_number=re.match("[0-9]+(?=\smV)", file).group(0)
    key=int(get_number)
    data=np.loadtxt(file_loc+"/"+file, skiprows=1)
    file_dict[key]=data
    #time=data[:,0]
    #potential=data[:,1]
    #current=data[:,2]
    #plt.plot(potential, current)
    #plt.show()

key_list=sorted(list(file_dict.keys()))

alpha=True
    for mode in ["lower", "higher"]:
        if mode=="higher":
            first_regime_end=2400
            second_regime_end=5000
        elif mode=="lower":
            first_regime_end=1000
            second_regime_end=5000
            third_regime_end=8000
        idx=key_list.index(second_regime_end)
        trumpet_positions=np.zeros((len(key_list),2))
        for i in range(0, len(key_list[:-1])):
            number=key_list[i]
            data=file_dict[number]
            time=data[:,0]
            potential=data[:,1]
            current=data[:,2]
            if mode=="higher":
                if number<first_regime_end:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="3")
                    subtract=bg.background_subtract([0.05,0.3, 0.01, 0.3,0, 0.35, 0, 0.35])
                elif number<second_regime_end:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="3")
                    subtract=bg.background_subtract([0.15,0.38, 0.05, 0.3,0.05, 0.4, 0, 0.35])
                else:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="4")
                    subtract=bg.background_subtract([0.2,0.45,  -0.05, 0.15,0.1, 0.48, -0.1, 0.25])
            elif mode=="lower":
                if number<first_regime_end:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="3")
                    subtract=bg.background_subtract([-0.25,0, -0.3, -0.05,-0.40, 0.35, -0.45, 0])  
                elif number<second_regime_end:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="3")
                    subtract=bg.background_subtract([-0.25,0.1, -0.4, -0.05,-0.40, 0.35, -0.45, 0])
                elif number<third_regime_end:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="3")
                    subtract=bg.background_subtract([-0.25,0.1, -0.4, -0.05,-0.32, 0.42, -0.48, 0])
                else:
                    bg=DCV_peak_area(time, potential, current, 0.07, func_order="1")
                    subtract=bg.background_subtract([-0.1,0.21, -0.45, -0.05,-0.32, 0.42, -0.52, 0])
            #trumpets.simulation_options["find_in_range"]=[0.05, 0.3]
            
            #
            
        
            full_subtract_potential=np.append(subtract["subtract_0"][0],subtract["subtract_1"][0])
            full_subtract_current=np.append(subtract["subtract_0"][1],subtract["subtract_1"][1])
            trumpet_pos=trumpets.trumpet_positions(full_subtract_current, full_subtract_potential, dim_flag=False)
            #plt.title(number)
            

            #fig, ax=plt.subplots(1,2)
            #ax[0].set_title(number)
            #ax[0].plot(potential, current)
            #ax[0].axvline(trumpet_pos[0])
            #ax[0].axvline(trumpet_pos[1])
            #for i in range(0, 2):
            #    ax[1].plot(subtract["subtract_{0}".format(i)][0],subtract["subtract_{0}".format(i)][1])
                #ax[0].plot(subtract["poly_{0}".format(i)][0],subtract["poly_{0}".format(i)][1])
                #ax[0].plot(subtract["bg_{0}".format(i)][0],subtract["bg_{0}".format(i)][1])
            #    ax[0].plot(subtract["subtract_{0}".format(i)][0],subtract["subtract_{0}".format(i)][1])
            #ax[1].axvline(trumpet_pos[0])
            #ax[1].axvline(trumpet_pos[1])
            #plt.show()

            trumpet_positions[i,:]=[trumpet_pos[0][0],trumpet_pos[1][0] ]
    
        in_volts=np.array(key_list)*1e-3
        trumpet_positions=trumpet_positions/trumpets.nd_param.c_E0
        trumpets.secret_data_trumpet=trumpet_positions
        trumpets.def_optim_list(["E_0", "k_0", "alpha", "dcv_sep"])
        higher_vals=[0.16776916561763955, 45.663389703591946, 0.40029420971731716, 0.02626456929885537]
        lower_vals=[-0.14253585137746128, 15.297594142649025, 0.022634489687786294]
        higher_vals=[0.1665178448507496, 45.9046417558063, 0.026121669037984387]

        if alpha==True:
            trumpets.def_optim_list(["E_0", "k_0","dcv_sep"])
            trumpets.dim_dict["alpha"]=0.5
        #trumpets.def_optim_list(["E_0", "k_0","dcv_sep"])
        sim=trumpets.simulate(lower_vals, in_volts, optimise_flag=True)
        #trumpets.dim_dict["alpha"]=0.5
        fig, ax=plt.subplots()
        #trumpets.trumpet_plot(in_volts,sim,  ax=ax)
        #trumpets.trumpet_plot( in_volts,trumpet_positions, ax=ax)
        #plt.show()
        cmaes_problem=pints.MultiOutputProblem(trumpets,in_volts,trumpet_positions)
        score = pints.GaussianLogLikelihood(cmaes_problem)

        trumpets.simulation_options["label"]="cmaes"
        lower_bound=np.append(np.zeros(len(trumpets.optim_list)), [0]*trumpets.n_outputs())

        upper_bound=np.append(np.ones(len(trumpets.optim_list)), [50]*trumpets.n_outputs())
        CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
        x0=list(np.random.rand(len(trumpets.optim_list)))+[5]*trumpets.n_outputs()
        cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, trumpets.n_parameters()+trumpets.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
        cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-2)
        trumpets.simulation_options["trumpet_test"]=False
        cmaes_fitting.set_parallel(True)
        found_parameters, found_value=cmaes_fitting.run()   
        real_params=trumpets.change_norm_group(found_parameters[:-trumpets.n_outputs()], "un_norm")
        print(list(real_params))
        sim=trumpets.simulate(found_parameters[:-trumpets.n_outputs()], in_volts, optimise_flag=True)
        trumpets.simulation_options["label"]="MCMC"
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
        mcmc.set_max_iterations(20000)
        save_file="sig_2_MCMC_{0}_alpha_{1}".format(mode, alpha)
        chains=mcmc.run()
        f=open(save_file, "wb")
        np.save(f, chains)
