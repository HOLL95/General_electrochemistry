import matplotlib.pyplot as plt
import math
import os
import sys
from pandas import DataFrame
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
import numpy as np
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
data_loc="/home/henryll/Documents/Experimental_data/Ella/LPMOComplete/"
data_loc="/home/henryll/Documents/Experimental_data/Ella/LPMO_8_5/"
files=["CjAA10_SWV_0.3to-0.3V_2mVamp_2Hz.txt","CjAA10_SWV_-0.3to0.3V_2mVamp_2Hz.txt"]
file="CjAA10_red.txt"
directions=[-1,1, -1, 1]
labels=["Red", "Ox", "Blank1", "Blank2"]
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
starts=[0.3, -0.3]
ends=[-0.3, 0.3]
skip=4
from pints.plot import trace
scan_names=["backwards", "forwards"]
param_dict={"forwards":{"linear":[0.037349696003567884, 0.4955722838974685, 0.566549769250069, 1.6586600634127136e-11, 0.1771725959995578, 0.06885308879259],
                        "squared":[0.046043012957300816, 0.10641815739978748, 0.6802850395460577, 9.760029846916142e-11, 0.17060193414848257, 0.06546023683227631, 0.12450439893971676 ],
                        "cubed":[0.05455146127793195, 0.09483553660267302, 0.6999999999998359, 1.1800456274176283e-10, 0.16943355306484342, 0.04911874483640233, 0.13944592221733743, 0.2588304799319463]},
            "backwards":{"linear":[0.005979888177296702, 2.792012856619911, 0.5560189612510722, 2.0880372389247936e-12, -0.17280969112456646, 0.022236504323313255],
                        "squared":[-0.0107329055214571, 0.10127238866305002, 0.30000000000000154, 7.31126734566571e-11, -0.16080077155481476, 0.021314859481760706, -0.25718684650254175],
                        "cubed":[0.023030910952811595, 0.06167724275752923, 0.32282598809943897, 1.6125294795691284e-10, -0.15655126080289428, 0.06645781551964802, -0.3240832252460244, -0.7137871976372256]
                        }
                        }

#filenames=["reverse","forwards"]
fig, ax=plt.subplots(1,2)
for i in range(0, len(files)):
    data_dict={}
    file=files[i]
    if "Cj" in file:
        
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
    data_dict["potential"]=E_p
    experimental_current=data[skip:,1]*1e9#/SW.nd_param.sw_class.c_I0
    data_dict["Experimental current (nA)"]=experimental_current
    ax[i].plot(E_p, experimental_current, label="Data")
    ax[i].set_ylabel("Current (nA)")
    ax[i].set_xlabel("Potential (V)")
    for j in range(0, len(extra_terms)):
        core_list+=[extra_terms[j]]
        
        SW.def_optim_list(core_list)
        first_key=scan_names[i]
        second_key=extra_terms[j][extra_terms[j].index("_")+1:]
        sim_params=param_dict[first_key][second_key]

        sim_current=SW.simulate(sim_params, [])*SW.nd_param.sw_class.c_I0*1e9
        data_dict["%s capacitance simulation (nA)"%second_key]=sim_current
        ax[i].plot(E_p, sim_current, label=second_key, lw=2)
        #plt.plot(E_p,experimental_current, label=labels[i])
        #plt.plot(E_p, sim_current)
        #plt.show()
    DataFrame(data=data_dict).to_csv("Cj_SWV_%s_scan.csv"%scan_names[i])
ax[0].legend(loc="lower left")
plt.show()

      