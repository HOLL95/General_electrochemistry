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
mplot=MCMC_plotting(burn=12500)
file_loc="MCMC"
files=os.listdir(file_loc)
common_params=["R0", "R1", "Q1", "alpha1", "sigma_1", "sigma_2"]
name_dict={
    "CPE":['R0', 'R1', 'Q1', 'alpha1', 'W1',"sigma_1", "sigma_2"],
    "2CPE":['R0', 'R1', 'Q2', 'alpha2', 'Q1', 'alpha1', "sigma_1", "sigma_2"]}
idxs={key:[name_dict[key].index(x) for x in common_params] for key in name_dict.keys()}
print(idxs)

repeats=list(range(1, 4))
models=list(name_dict.keys())
concentrations=["0", "0.005", "0.010", "0.015"]
col_len=len(common_params)

for i in range(0, len(concentrations)):
    fig, ax=plt.subplots(col_len, col_len)
    labels=[]
    for j in range(0, len(models)):
        print(concentrations[i], models[j])
        
        param_names=name_dict[models[j]]
        all_names=param_names+["sigma_1", "sigma_2"]
        
        for k in range(0, len(repeats)):
            filename=("_").join([str(x) for x in [repeats[k], concentrations[i], models[j], "bode", "MCMC"]])
            chains=np.load(file_loc+"/"+filename)
            new_chain=np.zeros((chains.shape[0], chains.shape[1], len(common_params)))
            for p in range(0, len(common_params)):
                new_chain[:,:, p]=chains[:,:,idxs[models[j]][p]]
            labels.append("{0} Model:{1}".format(models[j], repeats[k]))
            zscored_chains=mplot.convert_to_zscore(new_chain)
            new_chain[np.where(new_chain<0)]=None
            plot_chain=np.log10(new_chain)
            if j==0 and k==0:
                ax,twinx=mplot.plot_2d(common_params,plot_chain, label=repeats[k], burn=mplot.options["burn"], axis=ax,  alpha=0.1, title_debug=True, pooling=True)
            else:
                mplot.plot_2d(common_params,plot_chain, label=repeats[k], burn=mplot.options["burn"], axis=ax,  alpha=0.1, title_debug=True, pooling=True, twinx=twinx, rotation=45)
    
    mplot.axes_legend(labels,ax[0,-1])
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(bottom=0.125, left=0.136, right=0.91, top=0.975)
    fig.savefig("2D_MCMC_common_model_parameters_{0}_logscale.png".format(concentrations[i]), dpi=500)
        

                
        
        
        