import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser
randles={"z1":"R1", "z2":{"p_1":("Q1", "alpha1"), "p_2":["R2", "W1"]}}
frequency_powers=np.arange(1, 6, 0.1)
frequencies=[10**x for x in frequency_powers]
bounds={
"R1":[0, 1000],
"Q1":[0, 1e-2],
"alpha1":[0.1, 0.9],
"R2":[0, 1000],
"W1":[1, 200]
}
true_params={"R1":100, "Q1":1e-3, "alpha1":0.5, "R2":10, "W1":40}
param_names=["R1", "Q1", "alpha1", "R2", "W1"]
optim=EIS_optimiser(circuit=randles, parameter_bounds=bounds, frequency_range=frequencies, param_names=param_names, test=False)
sim=optim.test_vals(true_params, frequencies)
#plt.plot(sim[:,0], -sim[:,1])
#plt.plot(noisy_data[:,0], -noisy_data[:,1])
#plt.show()
for i in range(1, 7):
    sigma=i*0.001*np.sum(sim)/(2*len(sim))
    noisy_data=np.column_stack((optim.add_noise(sim[:,0], sigma), optim.add_noise(sim[:,1], sigma)))
    params, value, _,sim_data=optim.optimise(noisy_data)
    print(params)
    plt.subplot(2, 3, i)
    plt.plot(sim_data[:,0], -sim_data[:,1])
    plt.plot(noisy_data[:,0], -noisy_data[:,1])
plt.show()
