import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser, EIS_genetics
from circuit_drawer import circuit_artist
from pandas import read_csv
rows=6
cols=4
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
mode="error"
circuit_1={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}

circuit_2={ "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}
circuit_3={ "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
circs=[circuit_1, circuit_2, circuit_3,mark_circuit]
for plot in ["both"]:
    fig, axis=plt.subplots(rows, cols, constrained_layout=True)
    for z in range(0, len(circs)):
            
            current_file=np.load("BV_param_scans_best_fits_circuit_{0}_a.npy".format(z), allow_pickle=True).item()
            freq=current_file["data"]["freq"]
            keys=list(current_file.keys())
            
            sub_keys=list(current_file["results"].keys())
            for i in range(0, len(sub_keys)):
                    if i==0:
                        circuit_artist(circs[z], ax=axis[i,z], colour=colours[z])
                        axis[i,z].set_axis_off()
                        #if plot=="both":
                        #        twinx[i,z].set_axis_off()
                    key=sub_keys[i]
                    param_vals=list(current_file["data"][key].keys())
                    for j in range(0, len(param_vals)):
                            ax=axis[i+1,j]

                            val=param_vals[j]
                            fitting_data=current_file["data"][key][val]
                            phase=fitting_data[:,0]
                            mag=np.log10(fitting_data[:,1])
                            fitted_sim_vals=current_file["results"][key][val+"_results"]
                            simulator=EIS(circuit=circs[z])
                            sim=simulator.test_vals(fitted_sim_vals, freq)

                            EIS().nyquist(sim, ax=ax, orthonormal=False)
plt.show()
                        