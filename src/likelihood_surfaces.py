import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
class Kalman(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
    def likelihood_surfaces(self, parameters, data, **kwargs):
        if "pc" not in kwargs:
            kwargs["pc"]=0.1
        if "size" not in kwargs:
            kwargs["size"]=20
        if "scan_parameters" not in kwargs:
            desired_range=range(0, len(parameters))
        else:
            if type(kwargs["scan_parameters"]) is not list:
                raise TypeError("Parameters needs to be list not "+str(type(kwargs["scan_parameters"])))
            else:
                desired_range=[self.optim_list.index(x) for x in kwargs["scan_parameters"]]
        save_dict={}

        for i in desired_range:
            save_dict={}
            print(self.optim_list[i])
            for j in range(0, len(parameters)):
                if i==j:
                    pass
                if i>j:
                    start=time.time()
                    y_param, x_param=self.optim_list[i],self.optim_list[j]
                    y_idx, x_idx=self.optim_list.index(y_param), self.optim_list.index(x_param)
                    y_val, x_val=parameters[i], parameters[j]
                    y_list=np.linspace(y_val*(1-kwargs["pc"]), y_val*(1+kwargs["pc"]), kwargs["size"])
                    x_list=np.linspace(x_val*(1-kwargs["pc"]), x_val*(1+kwargs["pc"]), kwargs["size"])
                    XX,YY=np.meshgrid(x_list, y_list)
                    param_matrix=[[[0 for x in range(0, len(parameters))] for x in range(0, kwargs["size"])] for y in range(0, kwargs["size"])]
                    for q in range(0, kwargs["size"]):
                        for k in range(0, kwargs["size"]):
                            sim_params=copy.deepcopy(parameters)
                            sim_params[x_idx]=x_list[k]
                            sim_params[y_idx]=y_list[q]
                            param_matrix[q][k]=sim_params
                    param_list=list(itertools.chain(*param_matrix))
                    mp_argument=zip(param_list, ["fourier"]*(kwargs["size"]**2))
                    with multiprocessing.Pool(processes=4) as pool:
                        results = pool.starmap(self.test_vals, mp_argument)
                    errors=[self.RMSE(x, data) for x in results]
                    Z=[errors[i:i+kwargs["size"]] for i in range(0, len(errors), kwargs["size"])]
                    save_dict[x_param+"_"+y_param]={"X":XX, "Y":YY, "Z":Z}
                    print(x_param+"_"+y_param)
                    print(XX)
                    print(YY)
                    print(Z)
            np.save("Likelihood_surfaces_"+self.optim_list[i]+".npy", save_dict)
    def likelihood_curves(self, parameters, data, **kwargs):
        if "pc" not in kwargs:
            kwargs["pc"]=0.1
        if "size" not in kwargs:
            kwargs["size"]=20
        if "scan_parameters" not in kwargs:
            desired_range=range(0, len(parameters))
        else:
            if type(kwargs["scan_parameters"]) is not list:
                raise TypeError("Parameters needs to be list not "+str(type(kwargs["scan_parameters"])))
            else:
                desired_range=[self.optim_list.index(x) for x in kwargs["scan_parameters"]]
        save_dict={}
        for i in desired_range:
            x_param=self.optim_list[i]
            x_idx=self.optim_list.index(x_param)
            x_val=parameters[i]
            x_list=np.linspace(x_val*(1-kwargs["pc"]), x_val*(1+kwargs["pc"]), kwargs["size"])
            param_list=[[0 for x in range(0, len(parameters))] for y in range(0, kwargs["size"])]
            for q in range(0, kwargs["size"]):
                sim_params=copy.deepcopy(parameters)
                sim_params[x_idx]=x_list[q]
                param_list[q]=sim_params

            mp_argument=zip(param_list, ["fourier"]*kwargs["size"])
            with multiprocessing.Pool(processes=4) as pool:
                results = pool.starmap(self.test_vals, mp_argument)
            errors=[self.RMSE(x, data) for x in results]
            save_dict[x_param]={"X":x_list, "Y":errors}
        np.save("Likelihood_curves_high_gamma.npy", save_dict)