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
files=["CjAA10_red.txt","CjAA10_ox.txt","blank_0.25to-0.2.txt","blank_-0.2to0.25.txt"]
file="CjAA10_red.txt"
directions=[-1,1, -1, 1]
labels=["Red", "Ox", "Blank1", "Blank2"]
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
starts=[0.25, -0.2, 0.25, -0.2]
fig,ax=plt.subplots(1,3)
for i in range(0, len(files)):
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
    }
    K=param_list["k_0"]/param_list["omega"]
    print(param_list["deltaE"])
    simulation_options={
    "method":"square_wave",
    "experimental_fitting":False,
    "likelihood":"timeseries",
    "square_wave_return":"backwards",
    "optim_list":["E_0", "k_0", "alpha","gamma"],
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
    "E_0":[param_list["E_start"]-param_list["deltaE"], param_list["E_start"]],
    "k_0":[0.1, 1e3],
    "alpha":[0.4, 0.6]
    }
    omega_list=[5, 10]
    noise_val=[0.005]#, 0.01, 0.02, 0.03, 0.04, 0.05]
    omega_counter=0
    SWV_chain_dict={}

    SW=single_electron(None, param_list, simulation_options, other_values, param_bounds)

  
    volts=SW.e_nondim(SW.define_voltages())
    f, b, subtract, E_p=SW.SW_peak_extractor(volts)

    experimental_current=data[:-1,1]/SW.nd_param.sw_class.c_I0
    sim_current=SW.simulate([0.0, 2, 0.5, 1e-11], [])
    
    ax[0].plot(E_p, experimental_current, label=labels[i])
    if "Cj" in file:
        ax[1].plot(E_p, sim_current, label=labels[i])
        
        ax[2].plot(E_p, experimental_current, color=colours[i], label="Exp %s" %labels[i])
        ax[2].plot(E_p, sim_current, color=colours[i], linestyle="--", label="Sim %s" %labels[i])
for axes in ax:
    axes.legend()
    axes.set_xlabel("Potential (V)")
    axes.set_ylabel("Dimensionless current")
plt.show()