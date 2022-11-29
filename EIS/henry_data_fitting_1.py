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
from pandas import read_csv
data_loc="Experimental_data/4_7_22/"
files=os.listdir(data_loc)
exp_type="EIS"
plot_1="blank"
plot_2="t_eq"
plot_3="0.05"
get_color=plt.rcParams['axes.prop_cycle'].by_key()['color']
labels=["WT", "blank"]
get_color[2]="red"
c_idx=0
fig, ax=plt.subplots(1,4)
twinx=ax[2].twinx()
file_names=["EIS_CJX183_@0.05V.csv",
"EIS_CJX183_@0.005V_more_gain_less_t_eq.csv",
"EIS_blank_@0.0V.csv"
]
label_dict=dict(zip(file_names, ["0.05V", "0.005V", "blank"]))
print(label_dict.keys())
xaxis=["Potential(V)", "$Z_{re}$", "$log_{10}$(Freq)", "Potential(V)"]
yaxis=["Current($\\mu A$)", "$Z_{im}$", "$Z_{mag}$", "Current($\\mu A$)"]
for file in files:
    if exp_type in file:
        if plot_1 in file:#plot_1 in file or plot_2 in file or plot_3 in file:
            print(file)
            data=read_csv(data_loc+file, sep=",", encoding="unicode_escape", engine="python", skiprows=3, skipfooter=1)
            numpy_data=data.to_numpy(copy=True, dtype='float')
            real=np.flip(numpy_data[:, 6])
            imag=-np.flip(numpy_data[:,7])
            plot_freq=np.flip(np.log10(numpy_data[:,0]))
            freq=np.flip(numpy_data[:,0])
            phase=np.flip(numpy_data[:,2])
            mag=np.divide(np.flip(numpy_data[:,5]), 1000)


            bounds={
            "R0":[0, 10000],
            "Q1":[0, 10],
            "alpha1":[0.1, 0.9],
            "R1":[0, 10000],
            "C1":[0,10]
            }
            blank_circuit={"z1":"R0", "z2":{"p_1":"R1", "p_2":("Q1", "alpha1")}}
            blank_circuit={"z1":"R0", "z2":("Q1","alpha1")}
            translator=EIS(circuit=blank_circuit)
            param_names=["R0", "Q1", "alpha1", "R1"]
            #optim=EIS_optimiser(circuit=blank_circuit, parameter_bounds=bounds, frequency_range=freq, param_names=param_names, test=False, generation_test=True)
            noisy_data=np.column_stack((real, imag))
            gene_test=EIS_genetics(generation_size=12,  selection="AIC", initial_tree_size=1, best_record=True, num_top_circuits=6, generation_test=True, param_normalise=True)
            value,params, data=gene_test.assess_score(blank_circuit, param_names, freq, noisy_data, score_func="AIC", data_representation="nyquist", normalise=True)
            #print(translator.change_norm_group(params, "un_norm", return_type="list"))
            fig, ax=plt.subplots(1,1)
            translator.nyquist(noisy_data, ax=ax)
            translator.nyquist(data, ax=ax, label="sim")
            ax.legend()
            plt.show()
