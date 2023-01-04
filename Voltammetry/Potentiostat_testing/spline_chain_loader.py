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
vals=[0.2, 1000, 100.0, 1e-4, 0.000653657774506, 0.000245772700637, -1e-06, 2e-11, 0.5, 4.71238898038469, 4.71238898038469, 1.5]

params=["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha", "phase", "cap_phase", "error"]
noises=[0.5]

for epilogue in ["noisy", "interpolated"]:
    for i in range(0,len(noises)):
        file="MCMC/interpolation_assessment/MCMC_{0}pc_{1}".format(noises[i], epilogue)
        chains=np.load(file)
        
        print(epilogue)
        axes=mplot.plot_params(params, chains,  pool=False, Rhat_title=True, alpha=0.5)
        plt.subplots_adjust( 
                            left=0.074,
                            bottom=0.085,
                            right=0.946,
                            top=0.923,
                            wspace=0.378,
                            hspace=0.522
                               )
        #plt.tight_layout()
        plt.show()
        pints.plot.trace(chains)
        plt.show()
    #mplot.axes_legend(["{0} Hz".format(x) for x in fs], axes[-1, -1])
    