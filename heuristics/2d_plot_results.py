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
            "EIS_both":{"params":["E_0", "gamma","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"], "burn":5000, "true_vals":False},
            "EIS_e0"  :{"params":["E_0","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"],"burn":5000,"true_vals":False},   
            "EIS_gamma" :{"params":["gamma","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"],"burn":5000,"true_vals":True},  
            "EIS_functional" :{"params":["k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma_1", "sigma_2"],"burn":5000, "true_vals":True},  
            "Harm_min_Ru_fitting_real":{"params":["k_0", "alpha", "Ru","sigma"],"burn":5000,"true_vals":True}, 
            "Harm_min_Ru_fitting":{"params":["k_0", "alpha", "sigma"],"burn":5000,"true_vals":True}, 
            "Trumpet_no_Ru_fitting":{"params":["E_0", "k_0", "alpha","sigma_1", "sigma_2"],"burn":2500,"true_vals":True},     
            "Trumpet_Ru_fitting":{"params":["E_0", "k_0","alpha","Ru", "sigma_1", "sigma_2"],"burn":2500,"true_vals":True},   
            "Combined_initial_results":{"params":["E_0", "gamma","k_0",  "Cdl", "alpha", "cpe_alpha_faradaic", "Ru", "sigma"], "order":[1, 5, 4, 2, 6, 3, 0, 7], 
                                    "burn":5000,"true_vals":True}, 
}
mplot=MCMC_plotting()
desired_file=list(file_dict.keys())#
desired_file=["Combined_initial_results"]
true_values={"k_0":100, "E_0":-0.2, "Ru":250, "gamma":1e-10, "Cdl":1e-5, "alpha":0.55, "cpe_alpha_faradaic":0.8}
for i in range(0, len(desired_file)):

    chains=np.load(desired_file[i])
    if file_dict[desired_file[i]]["true_vals"] is not False:
        trv_arg=true_values
    else:
        trv_arg=None
        continue
    if "order"  in file_dict[desired_file[i]]:
        mplot.plot_2d(file_dict[desired_file[i]]["params"], chains, burn=file_dict[desired_file[i]]["burn"], order=file_dict[desired_file[i]]["order"], rotation=30, true_values=trv_arg)
    else:
        mplot.plot_2d(file_dict[desired_file[i]]["params"], chains, burn=file_dict[desired_file[i]]["burn"], rotation=30, true_values=trv_arg)
    fig=plt.gcf()
    #fig.set_size_inches(14, 9)
    plt.subplots_adjust(left=0.077,
                        bottom=0.11,
                        right=0.921,
                        top=0.88,
                        wspace=0.46, 
                        hspace=0.358)
    plt.show()
    fig.savefig(desired_file[i]+"_correlations.png", dpi=500)
    fig.clf()


