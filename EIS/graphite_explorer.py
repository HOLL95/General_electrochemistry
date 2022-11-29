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
from Fit_explorer import explore_fit
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
truncate=-35
real=np.flip(numpy_data[:truncate, 6])
imag=-np.flip(numpy_data[:truncate,7])
plot_freq=np.flip(np.log10(numpy_data[:,0]))
freq=np.flip(numpy_data[:truncate,0])
phase=np.flip(numpy_data[:,2])
data_info={
        "file_loc":"Experimental_data/5_7_22/",
        "files":["EIS_WT_0.005V.csv",
                "EIS_WT_0.2V.csv",
                "EIS_WT_-0.3V.csv",
                "EIS_blank_0.005V_wide_window.csv",],
        "file_label":["5mV", "200mV", "minus_300mV", "Blank"]}
sim=np.column_stack((real,imag))
file_names=dict(zip(data_info["file_label"], data_info["files"]))
voltages=["Blank"]
spectrums=dict(zip(voltages, [{}]*2))
for voltage in voltages:
    truncate_amount=35
    method="AIC"
    read_data=read_csv(data_info["file_loc"]+file_names[voltage], sep=",", encoding="unicode_escape", engine="python", skiprows=5, skipfooter=1)
    numpy_data=read_data.to_numpy(copy=True, dtype='float')
    real=(numpy_data[:-truncate_amount, 6])
    imag=-(numpy_data[:-truncate_amount,7])#
    #freq=np.log10(numpy_data[:,0])
    #phase=numpy_data[:,2]
    #mag=numpy_data[:,5]
    spectrums[voltage]=np.column_stack((real, imag))
    #EIS().nyquist(spectra)
    #plt.show()
interesting_blank_ones=[[1 ,1, 4, 5],[0, 5, 0,4]]
for i in range(0, len(interesting_blank_ones[0])):
    file_name="Best_candidates/Cjx183/{1}/{2}/Best_candidates_dict_{0}_12_gen_truncated_{3}.npy".format(interesting_blank_ones[0][i], method, "Blank", truncate_amount)
    results_dict=np.load(file_name, allow_pickle=True).item()

    #print(results_dict[0]["scores"])
    j=interesting_blank_ones[1][i]
    translator=EIS()
    simulator=EIS_genetics()
    translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)

    explore_fit(translated_circuit, dict(zip(params, results_dict["parameters"][j])), frequencies=freq, data=spectrums)
