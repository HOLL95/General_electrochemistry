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

import statsmodels.api as sm







names=["k0_shape", "k0_scale", "gamma", "Cdl", "Ru", "cpe_alpha_cdl", "sigma_1","sigma_2"]



chains=np.load(desired_files[0])
catted_chains=mplot.concatenate_all_chains(chains)
#catted_chains=[chains[2, :, param] for param in range(0, len(names))]
chain_dict=dict(zip(names, catted_chains))
trough_params=["gamma", "k0_scale"]
variable_params=names[:-2]
trough_params_target=[2e-10, 20]
num_vars=100
ax, twinx=mplot.plot_2d(names, chains)
for i in range(0, len(trough_params)):
    key=trough_params[i]
    depend_axis=names.index(key)

    independent_var=chain_dict[key]
    line_vals=[np.mean(independent_var), trough_params_target[i]]
    
    min_val=0.8*min(line_vals)
    max_val=1.2*max(line_vals)
    plot_line=np.logspace(np.log10(min_val), np.log10(max_val), num_vars)
    not_that_variable=[x for x in variable_params if x is not key]
    other_vars=np.column_stack(([chain_dict[x] for x in not_that_variable]))
    X=sm.add_constant(independent_var)
    for j in range(0, len(not_that_variable)):
        
        y = other_vars[:,j]

       
        model = sm.OLS(y, X).fit()


        predictions = model.predict(np.column_stack((np.ones(num_vars), plot_line)))
        independ_axis=names.index(not_that_variable[j])
        multiple=2
        var=np.std(X)
        plot_var=multiple*var
        print(key, min(plot_line), max(plot_line))
        print(not_that_variable[j], min(predictions), max(predictions))
        if depend_axis>independ_axis:
            

            axis=ax[depend_axis, independ_axis]
            axis.plot(predictions, plot_line, linestyle="-")
            axis.plot(predictions*0.5, plot_line, linestyle="--", color="black")
            axis.plot(predictions*1.5, plot_line, linestyle="--", color="black")
        elif independ_axis>depend_axis:
            axis=ax[independ_axis, depend_axis]
            axis.plot(plot_line, predictions, linestyle="-")
            axis.plot(plot_line,predictions*0.5,  linestyle="--", color="black")
            axis.plot( plot_line,predictions*1.5, linestyle="--", color="black")
plt.show()     
    

