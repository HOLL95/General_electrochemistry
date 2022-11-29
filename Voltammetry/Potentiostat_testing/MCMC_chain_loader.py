import numpy as np
import matplotlib.pyplot as plt
from pints import plot
import os
import sys
import pints
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from MCMC_plotting import MCMC_plotting
b_val=5000
mplot=MCMC_plotting(burn=b_val)
vals=[0.2, 1000, 100.0, 1e-4, 0.000653657774506, 0.000245772700637, -1e-06, 2e-11, 0.5, 4.71238898038469, 4.71238898038469, 1.5]

params=["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha", "phase", "cap_phase", "error"]
fs=np.flip([10, 50, 100, 200])
fs=[200, 100, 50, 10]
stds=np.zeros((len(params), len(fs)))
normal_stds=np.zeros((len(params), len(fs)))
rhats=np.zeros((len(params), len(fs)))
for epilogue in ["_normal_sampling", ""]:
    for i in range(0,len(fs)):
        frequency=fs[i]
        file="MCMC/{0}_Hz_PSV_2_pc_MCMC_no_transform{1}".format(frequency, epilogue)
        chains=np.load(file)
        
        for j in range(0, len(params), 4):
            fig, ax=plt.subplots(4, 1)
            pl_counter=0
            for m in range(j, j+4):
                for i in range(0, 3):
                    ax[pl_counter].plot(chains[i, b_val:, m], alpha=0.6)
                pl_counter+=1
            plt.show()
        #axes=mplot.plot_params(params, chains, true_values=vals, pool=False, Rhat_title=True, alpha=0.5)
        #plt.tight_layout()
        #plt.show()
    #mplot.axes_legend(["{0} Hz".format(x) for x in fs], axes[-1, -1])
    