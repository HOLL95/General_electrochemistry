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
        labels=["freq", "amp", "phase"]
        sinusoid_params=[]
        for i in range(0, self.simulation_options["num_sinusoids"]):          
            sinusoid_params+=[x+"_{0}".format(i+1) for x  in labels]
        self.sinusoid_params=sinusoid_params
        self.all_params=self.simulation_options["sobol_params"]+sinusoid_params
    def sobol_simulate(self, parameters):
        self.def_optim_list(self.sinusoid_params)
        if self.simulation_options["label"]=="MCMC":
            normed_params=parameters
        elif elf.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(parameters, "un_norm")
        self.test_vals(normed_params, "timeseries")
        pot=self.e_nondim(self.define_voltages())
        
        
        self.param_bounds["E_0"]=[min(pot), max(pot)]
        self.def_optim_list(self.all_params)
        problem =   {
            "num_vars":len(self.simulation_options["sobol_params"]),
            "names":self.simulation_options["sobol_params"], 
            "bounds":[[self.param_bounds[x][0], self.param_bounds[x][1]] for x in self.simulation_options["sobol_params"]]  
        }       
        sample_values=saltelli.sample(problem, self.simulation_options["sobol_dim"])
        len_sample_values=len(sample_values)
        start=time.time()
        param_mat=np.zeros((len_sample_values, len(self.all_params)))
        param_mat[:, :len(self.simulation_options["sobol_params"])]=sample_values
        param_mat[:, len(self.simulation_options["sobol_params"]):]=normed_params
        #current=self.test_vals(param_mat[0, :], "timeseries")
         
        ts_len=len(pot)-1
        mp_arr = mp.Array(c.c_double, len_sample_values*ts_len)
        arr = np.frombuffer(mp_arr.get_obj())
        globals()['arr']=arr.reshape((len_sample_values, ts_len))
        self.pool=mp.Pool(mp.cpu_count())
        for i,row in enumerate(param_mat):
            self.pool.apply_async(self.ts_wrapper, args=(i, row), callback=self.sim_callback)
        self.pool.close()
        self.pool.join()
        

        sobol_1=np.zeros(( len(self.simulation_options["sobol_params"]), ts_len))
        variance=np.zeros(ts_len)
        start=time.time()
        for i in range(0, ts_len):
            Si=sobol.analyze(problem, time_series_matrix[:,i])#Calculating sobol indices for every parameter (row), over iterative timepoints (col) 
            
            sobol_1[:,i]=Si["S1"]
            negs=np.where(Si["S1"]<0)

            if len(negs[0])>0:
                for j in range(0, len(negs[0])):
                    if (Si["S1"][negs[0][j]]+Si["S1_conf"][negs[0][j]])>0:
                        sobol_1[negs[0][j], i]=1e-20
                        
                    else:
                        sobol_1[:,i]=1e-20
                        break
        print(time.time()-start, "sobol_time")
        osc_points=int(1/self.dim_dict["sampling_freq"])
        num_intervals=ts_len//osc_points
        if num_intervals%osc_points!=0:
            extra_tw=True
            time_window_sobol=np.zeros((len(self.simulation_options["sobol_params"]), num_intervals+1))
        else:
            extra_tw=False
            time_window_sobol=np.zeros((len(self.simulation_options["sobol_params"]), num_intervals))

        for i in range(0, num_intervals):
            for j in range(len(self.simulation_options["sobol_params"])):
                sobol_sum=np.sum(sobol_1[j, i*osc_points:(i+1)*osc_points])#row is parameter, column is num_intervals
                time_window_sobol[j,i]=sobol_sum*np.log(sobol_sum)
        if extra_tw==True:
            for j in range(len(self.simulation_options["sobol_params"])):
                sobol_sum=np.sum(sobol_1[j, num_intervals*osc_points:])
                time_window_sobol[j,-1]=sobol_sum*np.log(sobol_sum)
        sobol_mean=np.mean(sobol_1, axis=1)
        total_entropy=1/(np.sum(np.multiply(sobol_mean, np.log(sobol_mean))))
        time_window_entropy=np.sum(np.sum(time_window_sobol, axis=0))
        total_var=1/(np.sum(np.sqrt(np.std(time_series_matrix, axis=1)))) #variance over the columns
        return [total_entropy, time_window_entropy, total_var]#sum([total_entropy*1000, 0.1*time_window_entropy, total_var])#sum([total_entropy*1000, 0.1*time_window_entropy, total_var])
    def ts_wrapper(self, i, params):
        current=self.test_vals(params, "timeseries")
        return (i, current)

    def sim_callback(self, result):
        global arr
        arr[result[0], :]=result[1][1:]