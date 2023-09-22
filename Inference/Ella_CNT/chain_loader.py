import numpy as np
import matplotlib.pyplot as plt
from pints.plot import trace
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from MCMC_plotting import MCMC_plotting
mplot=MCMC_plotting(burn=5000)
files=["sig_1_MCMC_alpha" ,"sig_2_MCMC_lower_alpha_True", "sig_2_MCMC_higher_alpha_True"]
for file in files:
    chains=np.load(file)
    params=["E_0", "k_0", "dcv_sep"]#, "sigma_1", "sigma_2"]
    mplot.plot_2d(params,chains)
    fig=plt.gcf()
    fig.savefig(file, dpi=500)
    #plt.show()