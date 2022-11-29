import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import numpy as np
import matplotlib.pyplot as plt
from harmonics_plotter import harmonics
data_loc="Experimental_data/15_11_22"
print(os.listdir(data_loc))
EIS_file="EIS_ferricynaide_2_C01.txt"
data=np.loadtxt(data_loc+"/"+EIS_file, skiprows=1)
plt.plot(data[:,1], data[:,2])
plt.scatter(data[:,1], data[:,2])
plt.show()