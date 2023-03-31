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
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import pints.plot
harm_range=list(range(4, 8))
from scipy import interpolate
from scipy.interpolate import CubicSpline
from MCMC_plotting import MCMC_plotting
mplot=MCMC_plotting()
E_start_vals=np.linspace(-50e-3, 50e-3, 100)
std_vals=np.linspace(0.01, 0.05, 15)
results=np.load("Minima_location_std.npy", allow_pickle=True).item()

k_vals=results["vals"]["k_0"]
alpha_vals=results["vals"]["alpha"]
minima=results["minima"]
dimensions=len(k_vals)

for i in range(0, dimensions):
    label="$E^0\\sigma$="+ mplot.format_values(std_vals[i], dp=3)+" V"
    plt.plot(alpha_vals, minima[i,:]*1000, label=label)
    plt.xlabel("$\\alpha$")
    plt.ylabel("min$(E_{mid})$ mV")
plt.legend()
plt.show()
                

 

            