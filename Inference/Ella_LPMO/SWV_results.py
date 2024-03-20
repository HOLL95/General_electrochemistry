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
param_dict={"forwards":{"linear":[0.07740790481948955, 0.19599344448843442, 0.6999999999999991, 3.041175684290795e-10, 0.2320093864674817, 0.19210063249038178],
                        "squared":[0.07749365275363175, 0.1669928872667265, 0.6999999999999948, 3.738295390192775e-10, 0.22507297421146255, 0.1861361260313643, 0.13797893860035515],
                        "cubed":[0.061691930601106594, 0.39634748188535107, 0.6603488461739903, 1.308461374520607e-10, 0.24625889639791154, 0.4038259533524897, -0.14825983150715594, -3.441954195458189]},
            "backwards":{"linear":[0.03589930232887069, 0.1921841369287885, 0.4593923029180595, 2.751332072342772e-10, -0.2285004471266312, -0.018874764595000926],
                        "squared":[0.04153382483479503, 0.24670600284162866, 0.5217423500583651, 1.9782041599733974e-10, -0.23737654809077924, -0.021078996599616318, 0.17425745981856977],
                        "cubed":[0.013830762050013984, 0.38330529447647255, 0.300000000000001, 1.1892213679139487e-10, -0.24735781361837716, -0.12535946702507772, 0.32493316073620804, 1.6064600267913463]
                        }
                        }

fig, ax=plt.subplots(1,2)
for i in range(0, len(files)):
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
    experimental_current=data[skip:,1]*1e9#/SW.nd_param.sw_class.c_I0
    ax[i].plot(E_p, experimental_current, label="Data")
    ax[i].set_xlabel("Current (nA)")
    ax[i].set_ylabel("Potential (V)")
    for j in range(0, len(extra_terms)):
        core_list+=[extra_terms[j]]
        
        SW.def_optim_list(core_list)
        first_key=scan_names[i]
        second_key=extra_terms[j][extra_terms[j].index("_")+1:]
        sim_params=param_dict[first_key][second_key]

        sim_current=SW.simulate(sim_params, [])*SW.nd_param.sw_class.c_I0*1e9
        ax[i].plot(E_p, sim_current, label=second_key, lw=2)
        #plt.plot(E_p,experimental_current, label=labels[i])
        #plt.plot(E_p, sim_current)
        #plt.show()
ax[0].legend(loc="lower left")
plt.show()

      