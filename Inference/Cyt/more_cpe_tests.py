
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
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))

frequencies=np.flip(data[:,2])*2*np.pi
circuit1={"z1":"R0", "z2":("Q1", "alpha1")}
circuit2={"z1":"R0", "z2":"C1"}
vals={'R0': 100.18458704315661, 'R1': 717261.5848789338, 'Q1': 6.984560646442896e-05,"alpha1":0.8, "C1":6.984560646442896e-05} 
#vals={'R0': 95.59209978737974, 'R1': 411.84647153196534, 'Q1': 5.60970405238147e-05, 'alpha1': 0.6525309089285137, 'Q2': 0.00018957824554479482, 'alpha2': 0.8726813960101358}

boundaries={"R0":[1e-3, 1e3,],
            "R1":[1e-3, 1e6,], 
            "R2":[0, 1e6,], 
            "Q2":[0,1], 
            "alpha2":[0,1],
            "C2":[0,1],
            "Q1":[0,1],
            "alpha1":[0,1]}

boundaries={key:[0.1*vals[key], 10*vals[key]] for key in vals.keys()}
boundaries["Q3"]=[0,1]
boundaries["alpha3"]=[0,1]

fig, ax=plt.subplots(1,1)
twinx=ax.twinx()
for circuit in [circuit1, circuit2]:
    sim_class=EIS(circuit=circuit,)
    #best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'C2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}

    sim_data=sim_class.test_vals(vals, frequencies)

    names=sim_class.param_names
    print(names)



    #EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx)
    EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx)
plt.show()
