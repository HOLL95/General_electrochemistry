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
print(sys.path)
data_loc="/home/henryll/Documents/Experimental_data/Ella/LPMOComplete/"
files=["PGE+CfAA10_SWV_0.3to-0.3V.txt","PGE+CfAA10_SWV_-0.3to0.3V.txt"]
file="CjAA10_red.txt"
directions=[-1,1, -1, 1]
labels=["Red", "Ox", "Blank1", "Blank2"]
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
starts=[0.3, -0.3]
ends=[-0.3, 0.3]
skip=4
from pints.plot import trace
scan_names=["backwards", "forwards"]

for i in range(0, len(files)):
    fig,ax=plt.subplots()
    file=files[i]
    if "Cf" in file:
        
        data=np.loadtxt(data_loc+file)

    import matplotlib.pyplot as plt
    import math
    import time
    from single_e_class_unified import single_electron


    import pints
    from pints import plot
    F=96485.3329
    R=8.31446261815324

    n=1
    T=298
  
    sampling_factor=200
    scan_direction=directions[i]
    estep=2e-3
    start=starts[i]+(directions[i]*skip*estep)
    param_list={
    "E_0":0.0,
    'E_start': start, #(starting dc voltage - V)
    'scan_increment': estep,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'gamma': 1e-11,
    "omega":2,
    "Ru":0,
    "original_gamma":1e-11,
    "T":273+25,
    "n":n,
    'k_0': 75, #(reaction rate s-1)
    'alpha': 0.5,
    "sampling_factor":sampling_factor,
    "SW_amplitude":2e-3,
    "deltaE":abs(start-ends[i])+estep*2,
    "v":scan_direction,
    "SWV_constant":0,
    "SWV_linear":0,
    "SWV_squared":0,
    "SWV_cubed":0,
    }
    K=param_list["k_0"]/param_list["omega"]
    print(param_list["deltaE"])
    simulation_options={
    "method":"square_wave",
    "experimental_fitting":False,
    "likelihood":"timeseries",
    "square_wave_return":"net",
    "optim_list":[],
    "no_transient":False
    }
    other_values={
        "filter_val": 0.5,
        "harmonic_range":range(0, 1),
        "experiment_time": None,
        "experiment_current": None,
        "experiment_voltage":None,
        "bounds_val":200,
    }
    param_bounds={
    "E_0":[0, 0.15],
    "k_0":[0.1, 1.5],
    "alpha":[0.3, 0.7], 
    "gamma":[1e-12, 5e-10],
    "SWV_constant":[-10, 10],
    "SWV_linear":[-10, 10],
    "SWV_squared":[-10, 10],
    "SWV_cubed":[-10, 10],
    }
    omega_list=[5, 10]
    noise_val=[0.005]#, 0.01, 0.02, 0.03, 0.04, 0.05]
    omega_counter=0
    SWV_chain_dict={}
    SW=single_electron(None, param_list, simulation_options, other_values, param_bounds)
    core_list=["E_0", "k_0", "alpha","gamma", "SWV_constant"]
    extra_terms=["SWV_linear", "SWV_squared", "SWV_cubed"]
    volts=SW.e_nondim(SW.define_voltages())
    f, b, subtract, E_p=SW.SW_peak_extractor(volts)
    #plt.scatter(range(0, len(data[skip:,0])),data[skip:, 0])
    #plt.scatter(range(0, len(E_p)), E_p, s=15)
    #plt.show()

    experimental_current=data[skip:,1]/SW.nd_param.sw_class.c_I0
    for j in range(0, len(extra_terms)):
        core_list+=[extra_terms[j]]
        
        SW.def_optim_list(core_list)
    
        
        #sim_current=SW.simulate([0.0, 2, 0.5, 1e-11, 0.4, 0.1, 0.5, 0], [])
        
        #plt.plot(E_p,experimental_current, label=labels[i])
        #plt.plot(E_p, sim_current)
        #plt.show()

        print(SW.n_outputs())
        cmaes_problem=pints.SingleOutputProblem(SW,np.linspace(0,1, len(experimental_current)),experimental_current)
        score = pints.GaussianLogLikelihood(cmaes_problem)

        SW.simulation_options["label"]="cmaes"
        lower_bound=np.append(np.zeros(len(SW.optim_list)), [0]*SW.n_outputs())

        upper_bound=np.append(np.ones(len(SW.optim_list)), [50]*SW.n_outputs())
        CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
        x0=list(np.random.rand(len(SW.optim_list)))+[5]*SW.n_outputs()
        cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, SW.n_parameters()+SW.n_outputs())], boundaries=CMAES_boundaries, method=pints.CMAES)
        cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-6)

        cmaes_fitting.set_parallel(True)
        found_parameters, found_value=cmaes_fitting.run()   
        real_params=SW.change_norm_group(found_parameters[:-SW.n_outputs()], "un_norm")
        print(list(real_params))
        sim_current=SW.simulate(found_parameters[:-SW.n_outputs()], [])
        #plt.plot(E_p, experimental_current, label=labels[i])
        #plt.plot(E_p, sim_current)
        #plt.show()
        SW.simulation_options["label"]="MCMC"
        MCMC_problem=pints.SingleOutputProblem(SW,np.linspace(0,1, len(experimental_current)),experimental_current)
        updated_lb=[param_bounds[x][0] for x in SW.optim_list]+([0]*SW.n_outputs())
        updated_ub=[param_bounds[x][1] for x in SW.optim_list]+([found_parameters[-1]*10])

        updated_b=[updated_lb, updated_ub]
        updated_b=np.sort(updated_b, axis=0)

        log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
        log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
        log_posterior=pints.LogPosterior(log_liklihood, log_prior)
        mcmc_parameters=np.append(real_params, found_parameters[-1])
        xs=[mcmc_parameters,
        mcmc_parameters,
        mcmc_parameters
        ]


        mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)
        mcmc.set_log_to_screen(False)
        mcmc.set_parallel(True)
        mcmc.set_max_iterations(10000)
        save_file="MCMC/%s_%s_MCMC_result"%(extra_terms[j], scan_names[i])
        chains=mcmc.run()
        f=open(save_file, "wb")
        np.save(f, chains)

        
