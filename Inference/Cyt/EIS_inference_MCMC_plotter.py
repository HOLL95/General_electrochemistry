import numpy as np
import matplotlib.pyplot as plt
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from MCMC_plotting import MCMC_plotting
from pints import plot
desired_files=["EIS_modified_None_C", "EIS_modified_None_CPE", "EIS_DD_None_C", "EIS_DD_None_CPE"]
labels=["Single Cdl C",  "Single Cdl CPE", "Double Cdl C", "Double Cdl CPE"]
combinations=[[1, 3], [0,1], [2,3]]
param_names=["gamma","k_0",  "Cdl", "alpha", "Ru", "cpe_alpha_faradaic", "cpe_alpha_cdl"]
mplot=MCMC_plotting(burn=10000)
for i in range(0, len(combinations)):
    for j in range(0, len(combinations[i])):
        idx=combinations[i][j]
        chains=np.load(desired_files[idx])
        if j==0:
            ax=mplot.plot_params(param_names, chains, label=labels[idx], alpha=0.75)
        else:
            mplot.plot_params(param_names, chains, axes=ax, label=labels[idx], alpha=0.75)
    ax[0,0].legend()
    plt.show()
