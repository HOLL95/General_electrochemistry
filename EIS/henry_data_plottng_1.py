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
        if plot_1 in file or plot_2 in file or plot_3 in file:
            print(file)
            data=read_csv(data_loc+file, sep=",", encoding="unicode_escape", engine="python", skiprows=3, skipfooter=1)
            numpy_data=data.to_numpy(copy=True, dtype='float')
            real=(numpy_data[:, 6])
            imag=(numpy_data[:,7])
            freq=np.log10(numpy_data[:,0])
            phase=numpy_data[:,2]
            mag=numpy_data[:,5]
            ax[1].scatter(real, imag, color=get_color[c_idx])
            ax[1].plot(real,imag, label=label_dict[file], color=get_color[c_idx])
            ax[2].plot(freq,mag, color=get_color[c_idx], label=label_dict[file],)
            ax[2].scatter(freq,mag, color=get_color[c_idx])
            twinx.plot(freq, phase, color=get_color[c_idx])
            twinx.set_ylabel("Phase")
            twinx.scatter(freq, phase, color=get_color[c_idx])
            c_idx+=1
    elif "DCV" in file:
        data=read_csv(data_loc+file, sep=",", encoding="unicode_escape", engine="python", skiprows=3, skipfooter=1)
        numpy_data=data.to_numpy(copy=True, dtype='float')

        if "blank" in file:
            potential=numpy_data[:,2]
            current=numpy_data[:,3]
            ax[0].plot(potential, current, color="red", label="blank")
        elif "pre" in file:
            potential=numpy_data[:,0]
            current=numpy_data[:,1]
            ax[0].plot(potential, current, label="WT")
        elif "post" in file:
            potential=numpy_data[:,0]
            current=numpy_data[:,1]
            ax[3].plot(potential, current, label="WT")
for i in range(0, 4):
        ax[i].legend()
        ax[i].set_xlabel(xaxis[i])
        ax[i].set_ylabel(yaxis[i])
plt.legend()
plt.show()
