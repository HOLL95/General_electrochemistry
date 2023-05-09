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
file_dict={
            "EIS_both":{"params":["E_0", "gamma","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"], "burn":5000, "true_vals":False,"figure":5},
            "EIS_e0"  :{"params":["E_0","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"],"burn":5000,"true_vals":False,"figure":6},   
            "EIS_gamma" :{"params":["gamma","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"],"burn":5000,"true_vals":True,"figure":7},  
            "EIS_functional" :{"params":["k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"],"burn":5000, "true_vals":True,"figure":8},  
            "Harm_min_Ru_fitting":{"params":["k_0", "alpha", "sigma"],"burn":5000,"true_vals":True,"figure":9}, 
            "Harm_min_Ru_fitting_real":{"params":["k_0", "alpha", "Ru","sigma"],"burn":5000,"true_vals":True,"figure":10}, 
            "Trumpet_no_Ru_fitting":{"params":["E_0", "k_0", "alpha","sigma_1", "sigma_2"],"burn":2500,"true_vals":True,"figure":11},     
            "Trumpet_Ru_fitting":{"params":["E_0", "k_0","alpha","Ru", "sigma_1", "sigma_2"],"burn":2500,"true_vals":True,"figure":12},   
            "Combined_initial_results":{"params":["Ru","E_0", "Cdl", "cpe_alpha_faradaic", "k_0", "gamma", "alpha"], 
                                    "burn":5000,"true_vals":True, "figure":13}, 
            }



mplot=MCMC_plotting()
desired_file=list(file_dict.keys())#

true_values={"k_0":100, "E_0":-0.2, "Ru":250, "gamma":1e-10, "Cdl":1e-5, "alpha":0.55, "cpe_alpha_faradaic":0.8}
values=["mean", "std", "error", "rhat"]
symbols=[r"$\mu$", r"$\sigma$", r"$\epsilon$", r"$\hat{R}$"]
num_v=len(values)
functions=[np.mean, np.std, lambda x, y:100*(np.abs(np.mean(x)-y)/abs(y)), rhat]
params=list(true_values.keys())
row_offset=2
col_offset=2

results_array=[["-" for x in range(0, len(params)+col_offset)] for y in range(0, len(values)*len(desired_file)+row_offset)]
results_array[0][0]="Parameter"
results_array[1][0]="True value"
results_array[0][1]=""
results_array[1][1]=""
idx=dict(zip(params, range(0, len(params))))
print(idx)
for i in range(0, len(desired_file)):
    pc=[]
    current_entry=file_dict[desired_file[i]]
    chains=np.load(desired_file[i])
    mplot=MCMC_plotting(burn=current_entry["burn"])
    
    #results_array[i*num_v:(i+1)*num_v, 1]=symbols
    catted_chains=mplot.concatenate_all_chains(chains)
    for j in range(0, len(current_entry["params"])):
        if current_entry["params"][j] not in params:
            continue
        for k in range(0, num_v):
            
            row_idx=(i*num_v)+k+row_offset
            results_array[row_idx][1]=symbols[k]
            
            if "Trumpet" in desired_file[i]:
                end=" (trumpet plot)"
            elif "EIS" in desired_file[i]:
                end=" (EIS)"
            elif "Harm_min" in desired_file[i]:
                end=" (harmonic minimum)"
            results_array[row_idx][0]="Figure " +str(current_entry["figure"])+end
            col_idx=idx[current_entry["params"][j]]+col_offset
            if values[k]=="error":
                args=[catted_chains[j], true_values[current_entry["params"][j]]]
                
                appendage=" %"
                #print(np.mean(catted_chains[j]), true_values[current_entry["params"][j]], current_entry["params"][j])
            elif values[k]=="rhat":
                args=[chains[:, current_entry["burn"]:, j]]
                appendage=""
            else:
                args=[catted_chains[j]]
                appendage=""
            logged_value=functions[k](*args)
            if "%" in appendage:
                pc.append(logged_value)
            results_array[row_idx][col_idx]=str(mplot.format_values(logged_value))+appendage
    print(pc)
    print(np.mean(pc))
titles=mplot.get_titles(params, units=True)
for i in range(0, len(titles)):
    results_array[0][i+col_offset]=titles[i]
    results_array[1][i+col_offset]=str(true_values[params[i]])

for line in results_array:

    print(",".join(line))

    




