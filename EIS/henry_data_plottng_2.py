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
data_loc="Experimental_data/5_7_22/"
files=os.listdir(data_loc)
exp_type="EIS"
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
get_color=plt.rcParams['axes.prop_cycle'].by_key()['color']
labels=["WT", "blank"]
get_color[2]="red"
c_idx=0
#twinx=ax[2].twinx()
plotter=EIS()


xaxis=["Potential(V)", "$Z_{re}$", "$log_{10}$(Freq)", "Potential(V)"]
yaxis=["Current($\\mu A$)", "$Z_{im}$", "$Z_{mag}$", "Current($\\mu A$)"]
fig, ax=plt.subplots()
for name in desired_files:
        data=read_csv(data_loc+name, sep=",", encoding="unicode_escape", engine="python", skiprows=3, skipfooter=1)
        numpy_data=data.to_numpy(copy=True, dtype='float')
        real=(numpy_data[:, 6])
        imag=-(numpy_data[:,7])
        freq=np.log10(numpy_data[:,0])
        phase=numpy_data[:,2]
        mag=numpy_data[:,5]
        spectra=np.column_stack((real, imag))
        plotter.nyquist(spectra, label=name, ax=ax, scatter=1)
        """ax[1].scatter(real, imag, color=get_color[c_idx])
        ax[1].plot(real,imag, label=label_dict[file], color=get_color[c_idx])
        ax[2].plot(freq,mag, color=get_color[c_idx], label=label_dict[file],)
        ax[2].scatter(freq,mag, color=get_color[c_idx])
        twinx.plot(freq, phase, color=get_color[c_idx])
        twinx.set_ylabel("Phase")
        twinx.scatter(freq, phase, color=get_color[c_idx])
        c_idx+=1"""
plt.legend()
plt.show()
