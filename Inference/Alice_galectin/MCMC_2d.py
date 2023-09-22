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
import numpy as np
from MCMC_plotting import MCMC_plotting
mplot=MCMC_plotting(burn=5000)
file_loc="MCMC"
files=os.listdir(file_loc)

name_dict={
    "CPE":['R0', 'R1', 'Q1', 'alpha1', 'W1'],
    "2CPE":['R0', 'R1', 'Q2', 'alpha2', 'Q1', 'alpha1']}

repeats=list(range(1, 4))
models=["2CPE"]#list(name_dict.keys())
concentrations=["0", "0.005", "0.010", "0.015"]
for i in range(0, len(concentrations)):
    
    for j in range(0, len(models)):
        print(concentrations[i], models[j])
        col_len=int(2+len(name_dict[models[j]]))
        param_names=name_dict[models[j]]
        all_names=param_names+["sigma_1", "sigma_2"]
        fig, ax=plt.subplots(col_len, col_len)
        for k in range(0, len(repeats)):
            filename=("_").join([str(x) for x in [repeats[k], concentrations[i], models[j], "bode", "MCMC"]])
            chains=np.load(file_loc+"/"+filename)
            
            
            zscored_chains=mplot.convert_to_zscore(chains)
            mplot.plot_2d(all_names,zscored_chains, label=repeats[k], burn=0, axis=ax,  alpha=0.1, title_debug=True, pooling=True)
        ax[0,0].legend()
        plt.show()
        

                
        
        
        