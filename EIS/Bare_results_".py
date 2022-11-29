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
for method in ["AIC"]:
    file_name="Bare_"+method+"_best_scores.npy"
    scores=np.load(file_name, allow_pickle=True).item()
    fig, ax=plt.subplots(1, 5)

        #translator=EIS()
        #print(scores["circuits"][i])
        #circuit_artist(translator.translate_tree(scores["circuits"][i][0]))
        #plt.show()
    for i in range(1, 12):
        file_name="Best_candidates/Bare/{1}/Best_candidates_dict_{0}_12_gen.npy".format(i, method)
        results_dict=np.load(file_name, allow_pickle=True).item()
        #print(results_dict[0]["scores"])
        for j in range(0, len(results_dict["scores"])):
            translator=EIS()
            simulator=EIS_genetics()


            translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)
            print(len(results_dict["parameters"][j]))
            print(len(params))
            
