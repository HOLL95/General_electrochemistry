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
b_val=12000
mplot=MCMC_plotting(burn=b_val)

params=["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha", "phase", "cap_phase"]
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
        converge=pints.rhat_all_params(chains)
        for j in range(0, len(params)):
            
            std_chain=mplot.chain_appender(chains, j)
            if epilogue=="_normal_sampling":
                normal_stds[j, i]=np.std(std_chain)
                rhats[j,i]=converge[j]
            else:
                stds[j, i]=np.std(std_chain)
              
total_stds=np.divide(stds, normal_stds)
fig, ax=plt.subplots(2, 1)
x1=np.arange(0, len(params)//2)
x2=np.arange(x1[-1]+1, len(params))
width=0.15
pad=0.02
width_counter=-0.5
for i in range(0, len(fs)):
    width_counter+=(width+pad)
    ax[0].bar(x1+width_counter, total_stds[:len(params)//2, i], width, label=str(fs[i])+" Hz")
    print()
    ax[1].bar(x2+width_counter, total_stds[len(params)//2:, i], width)
ax[0].set_xticks(x1, mplot.get_titles(params[:len(params)//2], units=False))
ax[1].set_xticks(x2, mplot.get_titles(params[len(params)//2:], units=False))
ax[0].legend()
ax[0].set_ylabel("Standard deviation ratio")
ax[1].set_ylabel("Standard deviation ratio")
plt.show()
        #if i ==0:
        #    axes=mplot.plot_params(params, chains)
        #else:
        #    axes=mplot.plot_params(params, chains, axes=axes)
    #mplot.axes_legend(["{0} Hz".format(x) for x in fs], axes[-1, -1])
#plt.show()