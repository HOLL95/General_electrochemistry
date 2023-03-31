import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
print(sys.path)
sys.path.append(source_loc)
from pints import plot
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
from MCMC_plotting import MCMC_plotting
mplot=MCMC_plotting()
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pints.plot
harm_range=list(range(4, 8))
from scipy import interpolate
from scipy.interpolate import CubicSpline

dimensions=5

frequencies=[10]
SRS=[400]
true_sf=400

results_dict=np.load("Param_scans_again.npy", allow_pickle=True).item()
param_scan_bounds=results_dict["params"]
E_start_vals=results_dict["E_vals"]
values=results_dict["values"]

param_scans=list(param_scan_bounds.keys())
units=mplot.get_units(param_scans)
titles=mplot.get_titles(param_scans, units=False)

results_dict={}
results_dict["params"]=param_scan_bounds
results_dict["E_vals"]=E_start_vals
E_mid=(np.add(E_start_vals, 0.3*2))/2
plot_Es=1000*(E_mid-0.3)
value_array=np.zeros((len(param_scans), dimensions, len(E_start_vals), 6))
for i in range(0, 6):
    fig ,ax=plt.subplots(2, 5)
    desired_harm=i
    for z in range(0, len(values)):
        key=param_scans[z]
        for j in range(0, len(values[z])):
            magnitude=values[z, j, :, desired_harm]

            
            if key=="gamma" or "CdlE" in key:
                dp=0
            else:
                dp=2
            ax[z//5, z%5].semilogy(plot_Es, magnitude, label=mplot.format_values(param_scan_bounds[key][j], dp=dp))
            if units[z]!="":
                ax[z//5, z%5].set_title("{0} ({1})".format(titles[z], units[z]))
            else:
                ax[z//5, z%5].set_title(titles[z])

        ax[z//5, z%5].legend(loc="upper center", ncols=2, handlelength=0.5, fontsize="medium", columnspacing=0.5)
        ax[z//5, z%5].set_xlabel("$E_{mid}-E^0 (mV)$")
        ax[z//5, z%5].set_ylabel("Absolute magnitude")
    plt.subplots_adjust(left=0.091,
                        bottom=0.08, 
                        right=0.946, 
                        top=0.973, 
                        wspace=0.425, 
                        hspace=0.2)
    fig.set_size_inches(14, 8)
    #plt.show()
    
    fig.savefig("harmonic_{0}.png".format(2*(i+1)), dpi=500)
    fig.clf()

#home/henney/Documents/Oxford/Voltammetry_results/Harmonic_minimum/    