import numpy as np
import matplotlib.pyplot as plt
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
import numpy as np
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from MCMC_plotting import MCMC_plotting
from pints.plot import trace
mplot=MCMC_plotting(burn=5000)
scan_names=["backwards", "forwards"]
extra_terms=np.flip(["SWV_squared"])
core_list=["E_0", "k_0", "alpha","gamma", "SWV_constant", "SWV_linear", "SWV_squared"]
fig, axes=plt.subplots(len(core_list),len(core_list))
for i in range(0, len(scan_names)):
    
    
    
    save_file="MCMC/%s_%s_MCMC_result"%(extra_terms[0], scan_names[i])
    chains=np.load(save_file, "r")
    #current_axes=axes[:len(core_list), :len(core_list)]
    if i==0:
        ax, twinx=mplot.plot_2d(core_list, chains,  title_debug=False, axes=axes, pooling=True, rotation=35)
    else:
        mplot.plot_2d(core_list, chains,  title_debug=False, axes=axes, pooling=True, rotation=35, twinx=twinx)
    #core_list=core_list[:-1]
"""for m in range(5, 7):
    print("hello")
    ax[m, 3].set_xticks([])
    ax[m, 3].set_xlabel("")
for lcv1 in range(0, 8):
    xlim=[1e6,-1e6]
    ylim=[1e6, -1e6]
    for lcv2 in range(0, 8):
        if lcv2>=lcv1:
            current_xlim=ax[lcv2,lcv1].get_xlim()
            xlim=[min(current_xlim[0], xlim[0]), max(current_xlim[1], xlim[1])]
            
            #print(current_xlim, lcv1, lcv2)
    for lcv2 in range(0, 8):
        if lcv2>=lcv1:
    #        print("Setting!", xlim, i,j)
            ax[lcv2,lcv1].set_xbound(lower=xlim[0], upper=xlim[1])"""
mplot.axes_legend(["Reverse", "Forwards"], ax[0, 5])
plt.show()