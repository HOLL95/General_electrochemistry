
import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_class import EIS
from EIS_optimiser import EIS_genetics
import numpy as np
import pints
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))

frequencies=np.flip(data[:,2])*2*np.pi
circuit={"z1":"R0","z3":("Q2", "alpha2"), "z2":{"p_1":("Q2", "alpha2"), "p_2":["R1", ("Q1", "alpha1")]}}
vals={'R0': 100.18458704315661, 'R1': 717261.5848789338, 'Q1': 6.984560646442896e-05,"alpha1":1, 'Q2': 6.656403132020056e-06, "alpha2":1} 
boundaries={"R0":[1e-3, 1e3,],
            "R1":[1e-3, 1e6,], 
            "Q2":[0,1], 
            "alpha2":[0,1],
            "C2":[0,1],
            "Q1":[0,1],
            "alpha1":[0,1]}


sim_class=EIS(circuit=circuit, fitting=True, parameter_bounds=boundaries, normalise=True)
#best={'R0': 93.8751449937169, 'R1': 426.57522762509535, 'C2': 0.00018098264633571246, 'alpha2': 0.9017743689145461, 'Q1': 5.75131567495785e-05, 'alpha1': 0.6456615312839018}

#sim_data=sim_class.test_vals(best, frequencies)

names=sim_class.param_names
print(names)



fig, ax=plt.subplots(1,1)
twinx=ax.twinx()
EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx)
#EIS().bode(sim_data, frequencies,ax=ax, twinx=twinx)
plt.show()
data_to_fit=sim_class.convert_to_bode(fitting_data)
sim_class.options["data_representation"]="nyquist"
cmaes_problem=pints.MultiOutputProblem(sim_class, frequencies,fitting_data)
score = pints.GaussianLogLikelihood(cmaes_problem)
sigma=0.5#sigma_fac*np.abs(np.sum(data))/2*len(data)
lower_bound=[0 for x in names]+[0.1*sigma]*2
upper_bound=[1 for x in names]+[100*sigma]*2
CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
random_init=list(abs(np.random.rand(sim_class.n_parameters())))+[sigma, sigma]
cmaes_fitting=pints.OptimisationController(score, random_init, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)

cmaes_fitting.set_parallel(True)
found_parameters, found_value=cmaes_fitting.run()   
print(found_parameters)

real_params=sim_class.change_norm_group(dict(zip(names, found_parameters[:-2])), "un_norm", return_type="dict" )
print(real_params)
sim_class.options["data_representation"]="nyquist"
sim_data=sim_class.test_vals(real_params, frequencies)
fig, ax=plt.subplots(1,2)
twinx=ax[0].twinx()
EIS().bode(fitting_data, frequencies, ax=ax[0], twinx=twinx)
EIS().bode(sim_data, frequencies,ax=ax[0], twinx=twinx)
EIS().nyquist(fitting_data, ax=ax[1],orthonormal=False)
EIS().nyquist(sim_data, ax=ax[1],orthonormal=False)
plt.show()