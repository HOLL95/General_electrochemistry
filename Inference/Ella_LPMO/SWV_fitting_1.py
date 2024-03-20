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
data_loc="/home/henryll/Documents/Experimental_data/Ella/SWV/SWV_exp/"
files=["CjAA10_red.txt","CjAA10_ox.txt"]
file="CjAA10_red.txt"
directions=[-1,1, -1, 1]
labels=["Red", "Ox", "Blank1", "Blank2"]
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
starts=[0.25, -0.2, 0.25, -0.2]

for i in range(0, len(files)):
    fig,ax=plt.subplots()
    file=files[i]
    if "Cj" in file:
        
        data=np.loadtxt(data_loc+file)
    else:
        bloc="/home/henryll/Documents/Experimental_data/Ella/SWV/Blank/set1/"
        data=np.loadtxt(bloc+file)
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
    param_list={
    "E_0":0.0,
    'E_start': starts[i], #(starting dc voltage - V)
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
    "deltaE":0.45,
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
    "k_0":[0.1, 1,5],
    "alpha":[0.4, 0.6], 
    "gamma":[1e-12, 5e-11],
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
    SW.def_optim_list(["E_0", "k_0", "alpha","gamma", "SWV_constant", "SWV_linear", "SWV_squared"])
  
    volts=SW.e_nondim(SW.define_voltages())
    f, b, subtract, E_p=SW.SW_peak_extractor(volts)
    
    experimental_current=data[:-1,1]/SW.nd_param.sw_class.c_I0
    sim_current=SW.simulate([0.0, 2, 0.5, 1e-11, 0.4, 0.1, 0.5], [])
    
    ax.plot(E_p, experimental_current, label=labels[i])
    ax.plot(E_p, sim_current)
    plt.show()
    #plt.plot(E_p)
    #plt.plot(data[:-1,0], linestyle="--")
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
    plt.plot(E_p, experimental_current, label=labels[i])
    plt.plot(E_p, sim_current)
    plt.show()
    