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
from pints.plot import trace
import copy
data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Galectin_31_7/"
files=os.listdir(data_loc)
header=6
footer=2

model_dict={"CPE":{"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", "W1"]}},
            "2CPE":{"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", ("Q2", "alpha2")]}},
            #"C":{"z1":"R0", "z2":{"p_1":"C2","p_2":["R1", "W1"]}}
            }
name_dict={"CPE":["R0","R1","Q1", "alpha1","W1"],
            "2CPE":["R0","R1","Q1", "alpha1","Q2","alpha2"],
            #"C":{"z1":"R0", "z2":{"p_1":"C2","p_2":["R1", "W1"]}}
            }

boundaries={"R0":[0, 1e4,],
            "R1":[-10, 1e6], 
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
save_file_loc="MCMC/"
mode_names=["bode"]
#files=["2023-07-14_SPE-P-DS_1_0.015.csv"]
for name in files:
    
    pd_data=read_csv(data_loc+name, sep=",", encoding="utf-16", engine="python", skiprows=header, skipfooter=footer)
    data=pd_data.to_numpy(copy=True, dtype='float')
    fitting_data=np.column_stack((np.flip(data[:,3]), -np.flip(data[:,4])))

    frequencies=np.flip(data[:,0])*2*np.pi
   
    concentration=re.search(r"(0|0\.[0-9]+)(?=\.csv$)", name).group(0)
    save_dict={"Data":fitting_data, "Frequencies":np.flip(data[:,0])}
    for model in model_dict.keys():
        save_dict[model]={x:{"samples":[]} for x in mode_names}
    for key in repeat_dict.keys():
        if key in name:
            repeat=repeat_dict[key]
    if concentration not in monster_save_dict:
        monster_save_dict[concentration]={}
    monster_save_dict[concentration][repeat]=save_dict
    for model in model_dict.keys():
        names=name_dict[model]
        sim_class=EIS(circuit=model_dict[model], fitting=True, parameter_bounds=boundaries, normalise=True, parameter_names=names)
        for mode in mode_names:
            
            
    
            
            
            print(names)



        
            data_to_fit=sim_class.convert_to_bode(fitting_data)
            sim_class.options["data_representation"]=mode
           
            if sim_class.options["data_representation"]=="bode":
                sim_class.secret_data=data_to_fit
               
            elif sim_class.options["data_representation"]=="nyquist":
                sim_class.secret_data=fitting_data
            data_mean=np.mean(sim_class.secret_data, axis=0)
            subtracted=np.square(np.subtract(sim_class.secret_data, data_mean))
            true_sigma=(1/len(frequencies))*np.sum(subtracted, axis=0)
            sigma=np.abs(np.mean(sim_class.secret_data, axis=0))
           
            cmaes_problem=pints.MultiOutputProblem(sim_class, frequencies,sim_class.secret_data)
            best=-1e12
            #score = pints.SumOfSquaresError(cmaes_problem)
            score = pints.GaussianLogLikelihood(cmaes_problem)
            lower_bound=[0 for x in names]+list(sigma*0.001)
            upper_bound=[1 for x in names]+list(sigma*1000)
            print(lower_bound)
            print(upper_bound)
            #transformation=pints.ComposedTransformation(*[pints.LogTransformation(1) if "alpha" not in x else pints.IdentityTransformation(1) for x in names +["sigma1", "sigma2"]])
            for j in range(0, 4):
                CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
                random_init=list(abs(np.random.rand(sim_class.n_parameters())))+list(sigma)
                cmaes_fitting=pints.OptimisationController(score, random_init, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
                cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)

                cmaes_fitting.set_parallel(True)
                sim_class.options["test"]=False

                found_params, found_value=cmaes_fitting.run()   
                if found_value>best:
                    found_parameters=found_params
                    best=found_value
                    print("sigma", found_params[:-2])
                    sim_data=sim_class.simulate(found_params[:-2], frequencies)

                    #fig, ax=plt.subplots(1,2)
                    #twinx=ax[0].twinx()
                    #EIS().bode(data_to_fit, frequencies, ax=ax[0], twinx=twinx, data_type="phase_mag")
                    #EIS().bode(sim_data, frequencies,ax=ax[0], twinx=twinx, data_type="phase_mag")
                    #EIS().nyquist(fitting_data, ax=ax[1],orthonormal=False)
                    #EIS().nyquist(sim_data, ax=ax[1],orthonormal=False)
                    #plt.show()
            #print(found_parameters)
            #sim_class.options["data_representation"]="nyquist"
            
            real_params=sim_class.change_norm_group(dict(zip(names, found_parameters[:-2])), "un_norm", return_type="dict" )
            print(np.mean(sim_class.secret_data, axis=0), found_parameters[-2:], sim_class.RMSE(sim_class.secret_data, sim_data)*((len(frequencies)-1)/len(frequencies)))

            #print(real_params)
            sim_class=EIS(circuit=model_dict[model], fitting=True, parameter_bounds=boundaries, normalise=False, data_representation=mode,parameter_names=names)
            if sim_class.options["data_representation"]=="bode":
                sim_class.secret_data=data_to_fit
            elif sim_class.options["data_representation"]=="nyquist":
                sim_class.secret_data=fitting_data
            MCMC_problem=pints.MultiOutputProblem(sim_class,frequencies,sim_class.secret_data)

            sigma=sim_class.RMSE(sim_class.secret_data, sim_data)
            sigma=found_parameters[-2:]
            updated_lb=[boundaries[x][0]-0.001 for x in names]+list(sigma*0.01)
            updated_ub=[boundaries[x][1]+0.001 for x in names]+list(sigma*100)
            
            updated_b=[updated_lb, updated_ub]
            updated_b=np.sort(updated_b, axis=0)

            log_liklihood=pints.GaussianLogLikelihood(MCMC_problem)
            log_prior=pints.UniformLogPrior(updated_b[0], updated_b[1])
            #log_prior=pints.MultivariateGaussianLogPrior(mean, np.multiply(std_vals, np.identity(len(std_vals))))
            log_posterior=pints.LogPosterior(log_liklihood, log_prior)
            real_param_dict=dict(zip(names, real_params))
            
            mcmc_parameters=np.append([real_params[x] for x in names], sigma)#[sim_class.dim_dict[x] for x in sim_class.optim_list]+[error]
            #mcmc_parameters=np.append(mcmc_parameters,error)
            xs=[mcmc_parameters,
                mcmc_parameters,
                mcmc_parameters
                ]

            mcmc = pints.MCMCController(log_posterior, 3, xs,method=pints.HaarioACMC)#, transformation=MCMC_transform)
            sim_class.options["test"]=False
            mcmc.set_parallel(True)
            mcmc.set_max_iterations(25000)
            try:
                chains=mcmc.run()
                save_file_name=("_").join([repeat, concentration, model, mode, "MCMC"])
                #trace(chains)
                #plt.show()
                f=open(save_file_loc+save_file_name, "wb")
                np.save(f, chains)
                f.close()
            except:
                continue

         
