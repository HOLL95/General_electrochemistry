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
frequency_powers=np.arange(1, 6, 0.1)
frequencies=[10**x for x in frequency_powers]
file_name="AIC_best_scores.npy"
scores=np.load(file_name, allow_pickle=True).item()
for i in range(0, len(scores["scores"])):
    #print(scores["scores"][i])
    plt.plot(scores["scores"][i])
    plt.axhline(scores["true_score"][i])
    plt.show()
    translator=EIS()
    print(scores["circuits"][i])
    circuit_artist(translator.translate_tree(scores["circuits"][i][0]))
    plt.show()
for i in range(1, 6):
    file_name="Best_candidates/round_2/AIC/Best_candidates_dict_{0}_12_gen.npy".format(i)
    results_dict=np.load(file_name, allow_pickle=True).item()
    #print(results_dict[0]["scores"])
    fig, ax=plt.subplots(2, 6)
    for j in range(0, len(results_dict["scores"])):
        translator=EIS()
        simulator=EIS_genetics()


        translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)
        print(len(results_dict["parameters"][j]))
        print(len(params))
        circuit_artist(translated_circuit, ax[0,j])
        try:

            sim_data=simulator.tree_simulation(results_dict["models"][j], frequencies, results_dict["data"], results_dict["parameters"][j])

            simulator.plot_data(results_dict["data"], ax[1,j])
            simulator.plot_data(sim_data, ax[1,j])
        except:
            print(file_name)
        ax[0,j].set_title(results_dict["scores"][j])
    plt.show()
