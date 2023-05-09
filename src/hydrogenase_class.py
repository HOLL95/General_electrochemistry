import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
import warnings
import time
from params_class import params
class multi_electron(single_electron):
     def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):

        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "mechanism" not in self.simulation_options:
            raise ValueError("Need to define a mechanism")
        elif self.simulation_options["mechanism"]=="EECR"
            elements=["AoRo", "AiRo", "ArRo", "AiRo", "AoRr", "ArRr"]
            linked_list=[   
                        [("AoRr","BV") ("ArRo", "Cat")],
                        [("AoRo", "BV"), ("AiRo", "Cat"), ("ArRr", "Cat")],
                        [("AoRr", "Cat"), ("AiRr", "BV")],
                        [("AiRo", "BV"), ("ArRo", "Cat")],
                        [("AoRo", "Cat"), ("AiRr", "Cat"), ("ArRr", "BV")],
                        [("ArRo", "BV"), ("AoRr", "Cat")],
                        ]
            subtracted_elem="ArRr"
            negative_linkage_dict=dict(zip(elements, linked_list))
            

    def BV_k(self, number, E,I, flag):
        str_key="_{0}".format(number)
        k0, e0, alpha=[self.nd_param.nd_param_dict[param+str_key] for param in ["k_0","E_0", "alpha"]]
        
        if flag=="oxidation":
            value=k0*np.exp((1-alpha)*(E-e0-self.nd_pram.nd_param_dict["Ru"]))
        elif flag=="reduction":
            value=k0*np.exp((1-alpha)*(E-e0-self.nd_pram.nd_param_dict["Ru"]))
        return value
    def simulate(self, parameters, times):
        if len(parameters)!= len(self.optim_list):
            print(self.optim_list)
            print(parameters)
            raise ValueError('Wrong number of parameters')
        if self.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(parameters, "un_norm")
        else:
            normed_params=copy.deepcopy(parameters)
        self.nd_param=params(self.dim_dict)
        K_matrix=np.zeros((5, 5))