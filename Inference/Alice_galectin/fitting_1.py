import matplotlib.pyplot as plt
import math
import os
import re
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
from pandas import read_csv
data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Galectin_31_7/"
files=os.listdir(data_loc)
header=6
footer=2

model_dict={"CPE":{"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", "W1"]}},
            "2CPE":{"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", ("Q2", "alpha2")]}},
            #"C":{"z1":"R0", "z2":{"p_1":"C2","p_2":["R1", "W1"]}}
            }
names=["R0", "C1", "R1", "Q1", "alpha1"]

boundaries={"R0":[0, 1e4,],
            "R1":[1e-6, 1e6], 
            "C2":[0,1],
            "Q2":[0,2], 
            "alpha2":[0,1],
            "Q1":[0,2], 
            "alpha1":[0,1],
            "W1":[0,1e6]}

get_conc=re.compile(r"0\.[0-9]+(?=\.csv$)")

#
#concentration
#->Repeat
#---->Data
#---->Model
#------->Data type
#---------->Parameter values
#---------->Generated fits
monster_save_dict={}

repeat_dict={"2023-07-14_SPE-P-DS_1":"1", 
            "2023-07-27_SPE-P-DS_1":"2",
            "2023-07-27_SPE-P-DS_5":"3"}
for name in files:
    
    pd_data=read_csv(data_loc+name, sep=",", encoding="utf-16", engine="python", skiprows=header, skipfooter=footer)
    data=pd_data.to_numpy(copy=True, dtype='float')
    fitting_data=np.column_stack((np.flip(data[:,3]), -np.flip(data[:,4])))

    frequencies=np.flip(data[:,0])*2*np.pi
   
    concentration=re.search(r"(0|0\.[0-9]+)(?=\.csv$)", name).group(0)
    save_dict={"Data":fitting_data, "Frequencies":np.flip(data[:,0])}
    for model in model_dict.keys():
        save_dict[model]={x:{"Values":[], "Fit":[]} for x in ["bode", "nyquist"]}
    for key in repeat_dict.keys():
        if key in name:
            repeat=repeat_dict[key]
    if concentration not in monster_save_dict:
        monster_save_dict[concentration]={}
    monster_save_dict[concentration][repeat]=save_dict
    for model in model_dict.keys():
        sim_class=EIS(circuit=model_dict[model], fitting=True, parameter_bounds=boundaries, normalise=True)
        for mode in ["bode", "nyquist"]:

    
            
            names=sim_class.param_names
            #print(names)



        
            data_to_fit=sim_class.convert_to_bode(fitting_data)
            sim_class.options["data_representation"]=mode
            if sim_class.options["data_representation"]=="bode":
                sim_class.secret_data=data_to_fit
            elif sim_class.options["data_representation"]=="nyquist":
                sim_class.secret_data=fitting_data
            cmaes_problem=pints.MultiOutputProblem(sim_class, frequencies,sim_class.secret_data)
            best=1e12
            score = pints.SumOfSquaresError(cmaes_problem)
            sigma=1000#sigma_fac*np.abs(np.sum(data))/2*len(data)
            lower_bound=[0 for x in names]#+[0.1*sigma]*2
            upper_bound=[1 for x in names]#+[100*sigma]*2
            #transformation=pints.ComposedTransformation(*[pints.LogTransformation(1) if "alpha" not in x else pints.IdentityTransformation(1) for x in names +["sigma1", "sigma2"]])
            for j in range(0, 3):
                CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
                random_init=list(abs(np.random.rand(sim_class.n_parameters())))#+[sigma, sigma]
                cmaes_fitting=pints.OptimisationController(score, random_init, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
                cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)

                cmaes_fitting.set_parallel(True)
                sim_class.options["test"]=False

                found_params, found_value=cmaes_fitting.run()   
                if found_value<best:
                    found_parameters=found_params
                    best=found_value
            #print(found_parameters)
            real_params=sim_class.change_norm_group(dict(zip(names, found_parameters)), "un_norm", return_type="dict" )
            
            
            print(real_params)
            sim_class.options["data_representation"]="nyquist"
            sim_data=sim_class.test_vals(real_params, frequencies)
            #print(monster_save_dict[concentration])
            monster_save_dict[concentration][repeat][model][mode]["Values"]=real_params
            monster_save_dict[concentration][repeat][model][mode]["Fit"]=sim_data
            
            #fig, ax=plt.subplots(1,2)
            #twinx=ax[0].twinx()
            #EIS().bode(fitting_data, frequencies, ax=ax[0], twinx=twinx)
            #EIS().bode(sim_data, frequencies,ax=ax[0], twinx=twinx)
            #EIS().nyquist(fitting_data, ax=ax[1],orthonormal=False)
            #EIS().nyquist(sim_data, ax=ax[1],orthonormal=False)
            #plt.show()

np.save("alice_fitting_results_2.npy", monster_save_dict)