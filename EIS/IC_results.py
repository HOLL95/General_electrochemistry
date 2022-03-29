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
for method in ["AIC", "BIC"]:
    file_name=method+"_best_scores.npy"
    scores=np.load(file_name, allow_pickle=True).item()
    fig, ax=plt.subplots(1, 5)
    for i in range(0, len(scores["scores"])):
        #print(scores["scores"][i])

        ax[i].plot(range(1, 6),scores["scores"][i])
        ax[i].axhline(scores["true_score"][i], color="black", linestyle="--", label="Fitted score")
        ax[i].set_xlabel("Generation")
        ax[i].set_ylabel("Score")
        ax[i].set_title("Attempt {0}".format(i+1))
    plt.subplots_adjust(top=0.936,
                        bottom=0.099,
                        left=0.052,
                        right=0.989,
                        hspace=0.2,
                        wspace=0.411)

    ax[0].legend()
    fig.set_size_inches(12, 4.5)
    fig.savefig("fit_results/{0}_best_fit.png".format(method), dpi=500)
        #translator=EIS()
        #print(scores["circuits"][i])
        #circuit_artist(translator.translate_tree(scores["circuits"][i][0]))
        #plt.show()
    for i in range(1, 6):
        file_name="Best_candidates/round_2/{1}/Best_candidates_dict_{0}_12_gen.npy".format(i, method)
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
            ax[0,j].set_axis_off()
            try:

                sim_data=simulator.tree_simulation(results_dict["models"][j], frequencies, results_dict["data"], results_dict["parameters"][j])

                simulator.plot_data(results_dict["data"], ax[1,j], label="Target")
                simulator.plot_data(sim_data, ax[1,j], label="Sim")
                ax[1, j].set_xlabel("$Z_r$")
                ax[1, j].set_ylabel("-$Z_i$")
            except:
                print(file_name)
            ax[0,j].set_title(round(results_dict["scores"][j], 2))
        ax[1,2].legend()

        fig.set_size_inches(14,7)
        plt.subplots_adjust(top=0.947,
                            bottom=0.087,
                            left=0.04,
                            right=0.989,
                            hspace=0.051,
                            wspace=0.32)
        fig.savefig("fit_results/{1}_Attempt_{0}.png".format(i, method), dpi=500)
