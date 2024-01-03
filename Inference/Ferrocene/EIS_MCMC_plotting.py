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
desired_files=["EIS_k_disp_tests_2"]
mplot=MCMC_plotting(burn=90000)

names=["k0_shape", "k0_scale", "gamma", "Cdl", "Ru", "cpe_alpha_cdl", "sigma_1","sigma_2"]



chains=np.load(desired_files[0])
       
ax=mplot.plot_params(names, chains, alpha=0.75)
ax[0,0].legend()
plt.show()
mplot.plot_2d(names, chains, pooling=False)
import statsmodels.api as sm


plt.show()

