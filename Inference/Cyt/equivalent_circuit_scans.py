
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
import copy
from MCMC_plotting import MCMC_plotting
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))

frequencies=np.flip(data[:,2])*2*np.pi
circuit={"z1":"R0", "z2":{"p_1":("Q1", "alpha1"), "p_2":["R1", ("Q2", "alpha2")]} }
vals={'R0': 95.59209978737974, 'R1': 411.84647153196534, 'Q1': 5.60970405238147e-05, 'alpha1': 0.6525309089285137, 'Q2': 0.00018957824554479482, 'alpha2': 0.8726813960101358}

boundaries={"R0":[1e-3, 1e3,],
            "R1":[1e-3, 1e6,], 
            "Q2":[0,1], 
            "alpha2":[0,1],
            "Q1":[0,1],
            "alpha1":[0,1]}


sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'C2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}



names=sim_class.param_names
print(names)

mplot=MCMC_plotting()
alpha_cdl_vals=[ vals["alpha1"], 0.7, 0.9, 1.0]
alpha_cf_vals=[0.5, 0.7, vals["alpha2"], 1.0]
a_vals={"alpha1":alpha_cdl_vals, "alpha2":alpha_cf_vals}
a_labels={"alpha1":"$C_{dl}$ $\\alpha$=","alpha2":"$C_{f}$ $\\alpha$=" }
a_val_keys=list(a_vals.keys())
fig, ax=plt.subplots(1,2)
twinx=[x.twinx() for x in ax]
for i in range(0, len(alpha_cdl_vals)):
    for j in range(0, len(a_val_keys)):
        if i==0:
            EIS().bode(fitting_data, frequencies, ax=ax[j], twinx=twinx[j], lw=4, compact_labels=True)
        copy_vals=copy.deepcopy(vals)
        copy_vals[a_val_keys[j]]=a_vals[a_val_keys[j]][i]
        print(copy_vals)
        sim_data=sim_class.test_vals(copy_vals, frequencies)
        EIS().bode(sim_data, frequencies,ax=ax[j], twinx=twinx[j],label="{0}{1}".format(a_labels[a_val_keys[j]], mplot.format_values(a_vals[a_val_keys[j]][i])), compact_labels=True )

plt.show()
