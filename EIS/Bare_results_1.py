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
frequencies=[7943.28, 6309.57, 5011.87, 3981.07, 3162.28, 2511.89, 1995.26, 1584.89, 1258.93, 1000.0, 794.328, 630.957, 501.18699999999995, 398.10699999999997, 316.228, 251.18900000000002, 199.52599999999998, 158.489, 125.89299999999999, 100.0, 79.433, 63.096000000000004, 50.119, 39.811, 31.623, 25.119, 19.953, 15.849, 12.589, 10.0, 7.943, 6.31, 5.012, 3.9810000000000003, 3.162, 2.512, 1.995, 1.585, 1.2590000000000001, 1.0, 0.794, 0.631, 0.501, 0.39799999999999996, 0.316, 0.251, 0.2, 0.158, 0.126]


for method in ["AIC"]:
    file_name="Bare_"+method+"_best_scores.npy"
    scores=np.load(file_name, allow_pickle=True).item()
    for i in range(1, 2):
        file_name="Best_candidates/Bare/{1}/Best_candidates_dict_{0}_12_gen.npy".format(i, method)
        results_dict=np.load(file_name, allow_pickle=True).item()
        #print(results_dict[0]["scores"])
        for j in range(0, 1):
            translator=EIS()
            simulator=EIS_genetics()


            translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)
            print(len(results_dict["parameters"][j]))
            print(params, "HEY")
            print(len(params))
            fig, ax=plt.subplots()


            sim_data=simulator.tree_simulation(results_dict["models"][j], frequencies, results_dict["data"], results_dict["parameters"][j])

            simulator.plot_data(results_dict["data"], ax, label="Target", scatter=True)
            simulator.plot_data(sim_data, ax, label="Sim", scatter=True)
            plt.show()
