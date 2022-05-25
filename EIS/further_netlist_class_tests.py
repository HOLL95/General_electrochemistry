import os
import sys
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
from time_domain_simulator_class import time_domain
frequency_powers=np.arange(1, 6, 0.1)
frequencies=[10**x for x in frequency_powers]

for i in range(2, 6):
    file_name="Best_candidates/round_3/{1}/Best_candidates_dict_{0}_12_gen.npy".format(i, "None")
    results_dict=np.load(file_name, allow_pickle=True).item()
    #print(results_dict[0]["scores"])
    fig=plt.figure()
    ax=[[0 for x in range(0, 6)] for y in range(0, 3)]



    gs = fig.add_gridspec(3,6)
    for m in range(0, 2):
        for j in range(0, 6):
            ax[m][j]=fig.add_subplot(gs[m, j])
    td_ax=fig.add_subplot(gs[2, :])
    #plt.show()
    for j in range(0, len(results_dict["scores"])):
            translator=EIS()
            simulator=EIS_genetics()


            translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)
            #translated_circuit={'z1': {'p1': [['C1', ("Q1", "alpha1")], ['R1', "C6"]],
            #                            'p2': {'p1': [{'p1': "C7", 'p2': 'R2'}, [{'p1': 'C8', 'p2': 'R3'}, {'p1': 'C2', 'p2': ("Q2", "alpha2")}]], 'p2': ["C5", 'C3']}},
            #                            'z2': 'C4', 'z0': 'R0'}
            print(translated_circuit)
            #print(translated_circuit)
            #print(len(results_dict["parameters"][j]))
            #print(len(params))
            circuit_artist(translated_circuit, ax[0][j])
            ax[0][j].set_axis_off()


            sim_data=simulator.tree_simulation(results_dict["models"][j], frequencies, results_dict["data"], results_dict["parameters"][j])

            simulator.plot_data(results_dict["data"], ax[1][j], label="Target")
            simulator.plot_data(sim_data, ax[1][j], label="Sim")
            ax[1][j].set_xlabel("$Z_r$")
            ax[1][j].set_ylabel("-$Z_i$")
            sim_dict=dict(zip(params, results_dict["parameters"][j]))
            #for i in range(5, 9):
            #    sim_dict["C"+str(i)]=1
            td=time_domain(translated_circuit, params=sim_dict)
            c, t, p=td.simulate()
            plt.plot(p, c)
            plt.show()
            ax[0][j].set_title(round(results_dict["scores"][j], 2))
    ax[1][2].legend()

    plt.show()
