import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser, EIS_genetics
from circuit_drawer import circuit_artist
data_loc="Experimental_data/5_7_22/"
files=[
"DCV_WT_pre_EIS.csv",
"DCV_blank_post_EIS.csv",
"EIS_blank_0.005V.csv",
"DCV_blank_pre_EIS.csv",
"EIS_WT_0.005V.csv",
"EIS_WT_0.2V.csv",
"EIS_blank_0.005V_wide_window.csv",
"EIS_WT_-0.3V.csv",
]
file_nums=[6, 4, 5, 7]
desired_files=[files[x] for x in file_nums]
file=files[4]
data=read_csv(data_loc+file, sep=",", encoding="unicode_escape", engine="python", skiprows=2, skipfooter=1)
numpy_data=data.to_numpy(copy=True, dtype='float')
truncate=-1
real=np.flip(numpy_data[:truncate, 6])
imag=-np.flip(numpy_data[:truncate,7])
plot_freq=np.flip(np.log10(numpy_data[:,0]))
freq=np.flip(numpy_data[:truncate,0])
phase=np.flip(numpy_data[:,2])

sim=np.column_stack((real,imag))
for method in ["AIC"]:
    for i in range(1, 6):
        file_name="Best_candidates/Cjx183/{1}/5mV/Best_candidates_dict_{0}_12_gen_truncated_1_scaled_1.npy".format(i, method)
        #file_name="Best_candidates/Cjx183/AIC/Blank/Best_candidates_dict_1_12_gen_truncated_1)_multiplicative.npy"
        results_dict=np.load(file_name, allow_pickle=True).item()
        #data=read_csv(current_dict["file_loc"]+current_dict["files"][j], sep=",", encoding=current_dict["encoding"], engine="python", skiprows=current_dict["skiprows"], skipfooter=1)
        #print(results_dict[0]["scores"])
        fig, ax=plt.subplots(2, 6)
        for j in range(0, len(results_dict["scores"])):
            translator=EIS()
            simulator=EIS_genetics()


            translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)
            print(len(results_dict["parameters"][j]))
            print(params, "HEY")
            print(len(params))
            circuit_artist(translated_circuit, ax=ax[0,j])
            ax[0,j].set_axis_off()
            try:

                sim_data=simulator.tree_simulation(results_dict["models"][j], freq, results_dict["data"], results_dict["parameters"][j])
                translator.nyquist(sim_data, ax=ax[1,j], label="Sim", colour="orange", orthonormal=False)
                translator.nyquist(sim, ax=ax[1,j], label="Target", scatter=1, orthonormal=False)

                ax[1, j].set_xlabel("$Z_r$")
                ax[1, j].set_ylabel("-$Z_i$")
            except:
                print(file_name)
            ax[0,j].set_title(round(results_dict["scores"][j], 4))
        ax[1,2].legend()

        fig.set_size_inches(14,7)
        plt.subplots_adjust(top=0.947,
                            bottom=0.087,
                            left=0.04,
                            right=0.989,
                            hspace=0.051,
                            wspace=0.32)
        plt.show()
        #fig.savefig("fit_results/DBCO_{1}_Attempt_{0}.png".format(i, method), dpi=500)
