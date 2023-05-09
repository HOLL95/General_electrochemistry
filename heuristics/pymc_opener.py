import numpy as np
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import matplotlib.pyplot as plt
import pymc as pm
import pints.plot as pintsplot
import arviz as az
from MCMC_plotting import MCMC_plotting
mplot=MCMC_plotting()
idata = az.from_netcdf("file_0.nc")
pintsplot.trace(mplot.convert_idata_to_pints_array(idata))
plt.show()
#az.plot_trace(idata, var_names=("m", "c"))
#print(vars(idata))
chains=idata.to_dict()
print(chains["posterior"])
params=list(chains["posterior"].keys())
num_params=len(params)
num_chains=len(chains["posterior"][params[0]])
num_samples=len(chains["posterior"][params[0]][0])
empty_pints=np.zeros((num_chains, num_samples, num_params))

for i in range(0, num_params):
    key=params[i]
    for j in range(0, num_chains):
        empty_pints[j, :, i]=chains["posterior"][key][j]
pintsplot.trace(empty_pints)
plt.show()