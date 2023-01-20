import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
import matplotlib.pyplot as plt
import sys
harm_range=list(range(4, 6))
from scipy import interpolate
from SALib.sample import saltelli
from SALib.analyze import sobol
import time
import multiprocessing as mp
import ctypes as c
class Input_optimiser(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "sobol_params" not in self.simulation_options:
            raise ValueError("Need to define the sobol_parameters")
        if "num_sinusoids" not in self.simulation_options:
            raise ValueError("Need to define the number of input sinusoids")
        if "sobol_dim" not in self.simulation_options:
            self.simulation_options["sobol_dim"]=32
        if "save_file" not in self.simulation_options:
            self.simulation_options["save_file"]=False
        elif "save_num" not in self.simulation_options:
            self.simulation_options["save_num"]=10
        if "skip_errors" not in simulation_options:
            self.simulation_options["skip_errors"]=True
        if "format" not in self.simulation_options:
            self.simulation_options["format"]="timeseries"
        labels=["freq", "amp", "phase"]
        sinusoid_params=[]
        for i in range(0, self.simulation_options["num_sinusoids"]):          
            sinusoid_params+=[x+"_{0}".format(i+1) for x  in labels]
        self.sinusoid_params=sinusoid_params
        self.all_params=self.simulation_options["sobol_params"]+sinusoid_params
        if self.simulation_options["save_file"]!=False:
            num_records=self.simulation_options["save_num"]
            self.save_dict={"params":sinusoid_params, "param_values":np.zeros((num_records, len(sinusoid_params))), "scores":np.linspace(1e8, 1e9, num_records)}
    def sobol_simulate(self, parameters):
        self.def_optim_list(self.sinusoid_params)
        if self.simulation_options["label"]=="MCMC":
            normed_params=parameters
        elif self.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(parameters, "un_norm")
        self.test_vals(normed_params, "timeseries")
        pot=self.e_nondim(self.define_voltages())
        self.param_bounds["E_0"]=[min(pot), max(pot)]
        self.def_optim_list(self.all_params)
        self.problem =   {
            "num_vars":len(self.simulation_options["sobol_params"]),
            "names":self.simulation_options["sobol_params"], 
            "bounds":[[self.param_bounds[x][0], self.param_bounds[x][1]] for x in self.simulation_options["sobol_params"]]  
        }       
        sample_values=saltelli.sample(self.problem, self.simulation_options["sobol_dim"],calc_second_order=False)
        len_sample_values=len(sample_values)
        param_mat=np.zeros((len_sample_values, len(self.all_params)))
        param_mat[:, :len(self.simulation_options["sobol_params"])]=sample_values
        param_mat[:, len(self.simulation_options["sobol_params"]):]=normed_params
        ts_len=len(pot)-1
        self.create_global_2d(len(self.simulation_options["sobol_params"]), ts_len, "sobol_arr")
        scanned_ts=self.matrix_simulate(param_mat, self.simulation_options["format"])
        sobol_enumerate_arg=enumerate(np.transpose(scanned_ts))#Now row is timepoint, column is parameter, because enumerate sends each row to pool
            with mp.Pool(processes=mp.cpu_count()) as P:
                P.map(self.sobol_wrapper,sobol_enumerate_arg)
        globals()['neg_counter']=0
        #########################################################################
        #print(time.time()-start, globals()['neg_counter'])
        max_vals=np.max(np.abs(scanned_ts), axis=1)
        if self.i_nondim(np.mean(max_vals))>10:
            return 1e9
        variance=np.zeros(ts_len)    
        osc_points=int(1/self.dim_dict["sampling_freq"])
        num_intervals=ts_len//osc_points
        if self.simulation_options["score"]=="Shannon_entropy":
            if num_intervals%osc_points!=0:
                extra_tw=True
                time_window_sobol=np.zeros((len(self.simulation_options["sobol_params"]), num_intervals+1))
            else:
                extra_tw=False
                time_window_sobol=np.zeros((len(self.simulation_options["sobol_params"]), num_intervals))

            for i in range(0, num_intervals):
                for j in range(len(self.simulation_options["sobol_params"])):
                    sobol_sum=np.sum(globals()["sobol_arr"][j, i*osc_points:(i+1)*osc_points])#row is parameter, column is num_intervals
                    time_window_sobol[j,i]=sobol_sum*np.log(sobol_sum)
            if extra_tw==True:
                for j in range(len(self.simulation_options["sobol_params"])):
                    sobol_sum=np.sum(globals()["sobol_arr"][j, num_intervals*osc_points:])
                    time_window_sobol[j,-1]=sobol_sum*np.log(sobol_sum)
            sobol_mean=np.mean(globals()["sobol_arr"], axis=1)
            total_entropy=1/(np.sum(np.multiply(sobol_mean, np.log(sobol_mean))))
            time_window_entropy=np.sum(np.sum(time_window_sobol, axis=0))
            total_var=1/(np.sum(np.sqrt(np.std(globals()["ts_arr"], axis=1)))) #variance over the columns
            return_val=sum([total_entropy*1000, 0.1*time_window_entropy, total_var])
            
        elif self.simulation_options["score"]=="Sobol-D":
            
            matrix=np.matmul(globals()["sobol_arr"],np.transpose(globals()["sobol_arr"]))
            inv_matrix=np.linalg.inv(matrix)
            return_val=np.linalg.det(inv_matrix)
        if self.simulation_options["save_file"]!=False:
            if return_val<max(self.save_dict["scores"]):
                max_idx=np.where(self.save_dict["scores"]==max(self.save_dict["scores"]))
                self.save_dict["scores"][max_idx]=return_val
                self.save_dict["param_values"][max_idx[0][0], :]=normed_params
                np.save(self.simulation_options["save_file"], self.save_dict)
                #print(self.save_dict)
        #print(return_val)
        return return_val
    def sobol_wrapper(self, timepoints):
        Si=sobol.analyze(self.problem, timepoints[1], calc_second_order=False)#Calculating sobol indices for every timepoint (each row is now one tim)
        negs=np.where(Si["S1"]<0)
        
        if len(negs[0])>0:
            for j in range(0, len(negs[0])):
                if (Si["S1"][negs[0][j]]+Si["S1_conf"][negs[0][j]])>0:
                    Si["S1"][negs[0][j]]=1e-20
                    globals()["neg_counter"]+=1
                else:
                    Si["S1"]=[1e-20]*len(Si["S1"])
                    globals()["neg_counter"]+=5
                    break
        globals()["sobol_arr"][:, timepoints[0]]=Si["S1"]
      
            
        

   