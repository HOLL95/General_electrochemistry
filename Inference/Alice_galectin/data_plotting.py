import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_class import EIS
from EIS_optimiser import EIS_genetics
import numpy as np
import pints
from pandas import read_csv
data_loc="/home/henryll/Documents/Experimental_data/Alice/Galectin_31_7/"
files=os.listdir(data_loc)
header=6
footer=2
fig, ax=plt.subplots(1,1)
for name in files:
    if "14_SPE-P-DS_1" in name:
        pd_data=read_csv(data_loc+name, sep=",", encoding="utf-16", engine="python", skiprows=header, skipfooter=footer)
        data=pd_data.to_numpy(copy=True, dtype='float')
        fitting_data=np.column_stack((np.flip(data[:,3]), np.flip(data[:,4])))

        frequencies=np.flip(data[:,0])*2*np.pi
        EIS().nyquist(fitting_data, ax=ax, label=name, orthonormal=False)
ax.legend()
plt.show()
phase=np.flip(data[:,1])
mag=np.flip(data[:,2])