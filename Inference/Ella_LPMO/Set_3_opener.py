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
data_loc="/home/henryll/Documents/Experimental_data/Ella/LPMOph5/"
data_loc="/home/userfs/h/hll537/Documents/Experimental_data/Ella/ph5/"
files=["CfAA10_SWV_0.3to-0.3.txt","CfAA10_SWV_-0.3to0.3.txt", "Cj_SWV_0.3to-0.3V.txt", "Cj_SWV_-0.3to0.3V.txt"]
labels=["Cf reverse", "Cf forwards", "Cj forwards", "Cj backwards"]
from MCMC_plotting import MCMC_plotting
import copy
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
starts=[0.3, -0.3, 0.3, -0.3]
ends=[-0.3, 0.3, -0.3, 0.3]
skip=4
from pints.plot import trace
scan_names=["CFbackwards", "CFforwards","Cjbackwards", "Cjforwards"]
directions=[-1,1, -1, 1]
mplot=MCMC_plotting()
fig, ax=plt.subplots(2,2)
print(ax)
with open("LPMO_ph5_results.txt", "r") as filehandle:
    with open("LPMO_ph5_results_2.txt", "w") as filehandle_2:
        
        lines = filehandle.read().splitlines()
        counter=0
        index_1=list(range(2, len(lines),2))[:-1]
        index_2=[x//3 for x in range(0, len(index_1))]
    
        param_lines=[[float(y) for y in lines[x].split(" ")[:-1]] for x in np.add(index_1, index_2)]
        for i in range(0, len(files)):
            filehandle_2.write(scan_names[i]+"\n")
            file=files[i]
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
            "E0_mean":0.0,
            "E0_std":0.01,
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
            #print(param_list["deltaE"])
            simulation_options={
            "method":"square_wave",
            "experimental_fitting":False,
            "likelihood":"timeseries",
            "square_wave_return":"net",
            "dispersion_bins":[16],
            "GH_quadrature":True,
            "optim_list":[],
            "no_transient":False,
            "label":"MCMC"
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
            "E_0":[-0.1, 0.15],
            "E0_mean":[-0.1, 0.15],
            "E0_std":[0.0001, 0.1],
            "k_0":[0.01, 5],
            "alpha":[0.4, 0.6], 
            "gamma":[1e-12, 1e-8],
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
            core_list=["E0_mean", "E0_std", "k_0", "alpha","gamma", "SWV_constant"]
            extra_terms=["SWV_linear", "SWV_squared", "SWV_cubed"]
            volts=SW.e_nondim(SW.define_voltages())
            f, b, subtract, E_p=SW.SW_peak_extractor(volts)
            #plt.scatter(range(0, len(data[skip:,0])),data[skip:, 0])
            #plt.scatter(range(0, len(E_p)), E_p, s=15)
            experimental_current=data[skip:,1]#/SW.nd_param.sw_class.c_I0
            ax[i//2][i%2].plot(E_p, experimental_current, label="Data", lw=2, alpha=0.7)
            ax[i//2][i%2].set_xlabel("Potential (V)")
            ax[i//2][i%2].set_ylabel("Current (A)")
            ax[i//2][i%2].set_title(scan_names[i])
            #plt.show()
            results_dict={}
            num_terms=len(extra_terms)
            for j in range(0, len(extra_terms)):
                core_list+=[extra_terms[j]]
                
                SW.def_optim_list(core_list)
                normed_params=param_lines[(i*num_terms)+j]
                un_normed_params=SW.change_norm_group(normed_params[:-1], "un_norm")
                print(un_normed_params+[normed_params[-1]])
                sim_current=SW.simulate(un_normed_params, [])*SW.nd_param.sw_class.c_I0
                ax[i//2][i%2].plot(E_p, sim_current, label=extra_terms[j])
                param_line=[str(x) for x in list(un_normed_params)]+[str(un_normed_params[-1])]
                
                param_names=[x+" "+mplot.unit_dict[x]+" " for x in copy.deepcopy(SW.optim_list)+["error"]]
                param_names=[param_names[x]+(" "*(len(param_line[x])-len(param_names[x]))) for x in range(0, len(param_names))]
                filehandle_2.write((" ").join(param_names+["\n"]))
                filehandle_2.write((" ").join(param_line+["\n"]))
ax[0,0].legend()
plt.show()

