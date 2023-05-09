import numpy as np
from pints import plot, rhat
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
print(sys.path)
sys.path.append(source_loc)
from MCMC_plotting import MCMC_plotting
import matplotlib.pyplot as plt
#chains=np.load("Combined_initial_scaling_{0}".format(10))
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
file_dict={
            
            
            "Harm_min_Ru_fitting":{"params":["k_0", "alpha"],"burn":5000,"true_vals":True, "colour":colours[2], "label":"Harmonic minimum"}, 
            "Trumpet_no_Ru_fitting":{"params":["E_0", "k_0", "alpha"],"burn":2500,"true_vals":True, "colour":colours[3], "label":"Trumpet plot"},    
            "EIS_functional" :{"params":["k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru"],"burn":5000, "true_vals":True,"colour":colours[1], "label":"EIS"}, 
            "Combined_initial_results":{"params":["E_0", "gamma","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru"], "order":[1, 5, 4, 2, 6, 3, 0], 
                                    "burn":5000,"true_vals":True, "colour":colours[0], "label":"Combined"}, 
}
mplot=MCMC_plotting()
desired_files=list(file_dict.keys())#
desired_file="Combined_initial_results"
other_files=desired_files[:-1]
true_values={"k_0":100, "E_0":-0.2, "Ru":250, "gamma":1e-10, "Cdl":1e-5, "alpha":0.55, "cpe_alpha_faradaic":0.8}
chains=np.load(desired_file)

master_param_list=file_dict[desired_file]["params"]

axis, twinx=mplot.plot_2d(file_dict[desired_file]["params"], chains, burn=file_dict[desired_file]["burn"], order=file_dict[desired_file]["order"], rotation=30,  pooling=True, density=False, log=True)



for z in range(0, len(other_files)):
    key=other_files[z]
    chains=np.load(other_files[z])
    pooled_chains=mplot.concatenate_all_chains(chains)
    for i in range(0, len(file_dict[key]["params"])):
        for j in range(0, len(file_dict[key]["params"])):
            idx_i=master_param_list.index(file_dict[key]["params"][i])
            if i==j:
                
                axes=axis[idx_i, idx_i]
               
                axes=twinx[idx_i]
                axes.hist(pooled_chains[i], density=False, alpha=0.8, color=file_dict[key]["colour"], log=True)
                
            else:
                idx_j=master_param_list.index(file_dict[key]["params"][j])
                print(idx_i, idx_j, )
                if idx_i>idx_j:
                    axis[idx_i, idx_j].scatter(pooled_chains[j], pooled_chains[i], s=0.1, alpha=0.5, color=file_dict[key]["colour"])


for i in range(0, len(master_param_list)):
    for j in range(0, len(master_param_list)):
        if i==j:
            twinx[i].axvline(true_values[master_param_list[i]], color="black", linestyle="--", lw=1)     
        elif i>j:
            axis[i,j].scatter(true_values[master_param_list[j]],true_values[master_param_list[i]], color="black", s=20, marker="x")

for i in range(0, len(desired_files)):
    key=desired_files[i]
    axis[0,-1].scatter(0,0, color=file_dict[key]["colour"], label=file_dict[key]["label"]), 
axis[0,-1].scatter(0, 0, color="white")
axis[0,-1].legend()
fig=plt.gcf()
plt.show()

fig.savefig("all_together"+"_correlations.png", dpi=500)

