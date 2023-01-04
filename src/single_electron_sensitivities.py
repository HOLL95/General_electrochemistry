import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
import matplotlib.pyplot as plt
harm_range=list(range(4, 6))
from scipy import interpolate
from scipy.integrate import odeint
import time
import multiprocessing as mp
class Sensitivity(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        self.sens_params=["E_0", "k_0", "Ru", "Cdl", "CdlE1", "CdlE2", "CdlE3", "gamma", "alpha"]
        self.num_sens=len(self.sens_params)
        self.sensitivity_array=np.array([])
        self.t_array=np.array([], dtype="float64")
        self.def_optim_list(self.sens_params)
    def Sensitivity_ODE_system(self, I, theta, E, time):
        
        E_0, k_0, Ru, Cdl, CdlE1, CdlE2, CdlE3, gamma, alpha=[self.nd_param.nd_param_dict[x] for x in self.sens_params]
        
            
        
        exp=np.exp
        sensitivities=np.zeros((len(time), self.num_sens))
        Er=(E - (I*Ru))
        ErE0=(E - E_0 - (I*Ru))
        
        dE=np.array([self.voltage_query(x)[1] for x in self.time_vec])
        dtheta_dt=-k_0*theta*exp(-alpha*ErE0) + k_0*(1 - theta)*exp((1 - alpha)*ErE0)
        Cdlp=Cdl*(CdlE1*Er + CdlE2*Er**2 + CdlE3*Er**3 + 1)
        dI=(dE-(I/Cdlp)+gamma*dtheta_dt*(1/Cdlp))/Ru 
        
        # E_0
        sensitivities[:,0]= gamma*(-alpha*k_0*theta*exp(-alpha*(ErE0)) + k_0*(1 - theta)*(alpha - 1)*exp((1 - alpha)*(ErE0)))
        # k_0
        sensitivities[:,1]= gamma*(-theta*exp(-alpha*(ErE0)) + (1 - theta)*exp((1 - alpha)*(ErE0)))
        # Ru
        sensitivities[:,2]= -Cdl*dI*(CdlE1*(E - I*Ru) + CdlE2*(E - I*Ru)**2 + CdlE3*(E - I*Ru)**3 + 1) + Cdl*(-Ru*dI + dE)*(-CdlE1*I - 2*CdlE2*I*(E - I*Ru) - 3*CdlE3*I*(E - I*Ru)**2) + gamma*(-I*alpha*k_0*theta*exp(-alpha*(ErE0)) - I*k_0*(1 - alpha)*(1 - theta)*exp((1 - alpha)*(ErE0)))
        # Cdl
        sensitivities[:,3]= (-Ru*dI + dE)*(CdlE1*(E - I*Ru) + CdlE2*(E - I*Ru)**2 + CdlE3*(E - I*Ru)**3 + 1)
        # CdlE1
        sensitivities[:,4]= Cdl*(E - I*Ru)*(-Ru*dI + dE)
        # CdlE2
        sensitivities[:,5]= Cdl*(E - I*Ru)**2*(-Ru*dI + dE)
        # CdlE3
        sensitivities[:,6]= Cdl*(E - I*Ru)**3*(-Ru*dI + dE)
        # gamma
        sensitivities[:,7]= -k_0*theta*exp(-alpha*(ErE0)) + k_0*(1 - theta)*exp((1 - alpha)*(ErE0))
        # alpha
        sensitivities[:,8]= gamma*(-k_0*theta*(-E + E_0 + I*Ru)*exp(-alpha*(ErE0)) + k_0*(1 - theta)*(-E + E_0 + I*Ru)*exp((1 - alpha)*(ErE0)))

        
     
        return sensitivities
    
    def simulate_S1(self):


        abserr = 1.0e-6
        relerr = 1.0e-6
        stoptime = self.time_vec[-1]
        numpoints = len(self.time_vec)

        w0 = [0,0, self.voltage_query(0)[0]]
        # Call the ODE solver.

        wsol, info = odeint(self.current_ode_sys, w0, self.time_vec,
                      atol=abserr, rtol=relerr, full_output=True)
        
        
        return wsol[:,0], wsol[:,1], wsol[:,2]
    def get_symbolic_sensitivity(self, **kwargs):
        if "params" in kwargs:
            self.update_params(kwargs["params"])
        current, theta, potential=self.simulate_S1()
        sensitivities=self.Sensitivity_ODE_system(np.array(current), np.array(theta), np.array(potential), self.time_vec)
        for i in range(0, self.num_sens):
            sensitivities[:,i]=sensitivities[:,i]*self.nd_param.nd_param_dict[self.sens_params[i]]
        return sensitivities
        
        
        
        return FIM
    def calc_FIM(self, sensitivities, noise):
        points=sensitivities.shape[0]
        covariance=np.zeros((points,points))
        for i in range(0, len(sensitivities)):
            covariance[i,i]=(1/noise)
        scaled=np.matmul(np.transpose(sensitivities), covariance)
        FIM=np.matmul(scaled, sensitivities)
        return FIM
    def D_optimality(self, **kwargs):
        if "FIM" not in kwargs:
            if "noise" not in kwargs:
                raise ValueError("Need to provide either the FIM or an estimate of the noise")
            else:
                if "type" not in kwargs:
                    kwargs["type"]="symbolic"
                if kwargs["type"]=="symbolic":
                    Sens_func=self.get_symbolic_sensitivity
                    args={}
                elif kwargs["type"]=="numeric":
                    Sens_func=self.get_numeric_sensitivity
                    args={}
                    if "Delta" not in kwargs:
                        args["Delta"]=1e-2
                if "params" in kwargs:
                    args["params"]=kwargs["params"]
                sensitivity=Sens_func(noise=kwargs["noise"], **args)
                FIM=self.calc_FIM(sensitivity, kwargs["noise"])
                return np.linalg.det(FIM)
        else:
            return np.linalg.det(kwargs["FIM"])        
        
    def find_nearest(self, array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx
    def get_numeric_sensitivity(self, **kwargs):
        if "params" not in kwargs:
            params=[self.dim_dict[x] for x in self.sens_params]
        else:
            params=kwargs["params"]
        orig_params=copy.deepcopy(params)
        #reference_current=self.test_vals(orig_params, "timeseries")
        if "Delta" not in kwargs:
            kwargs["Delta"]=1e-3
        if "max_size" not in kwargs:
            kwargs["max_size"]=25000
        
        current=(self.test_vals(orig_params, "timeseries"))
        plt.plot(current)
        plt.show()
        for i in range(0, len(params)):
            currents=[]
            
            for j in [-1, 1]:
                sim_params=copy.deepcopy(orig_params)
                sim_params[i]=sim_params[i]+(sim_params[i]*j*kwargs["Delta"])
                pot=self.e_nondim(self.define_voltages())
                self.update_params(sim_params)
                current=(self.test_vals(sim_params, "timeseries"))
                #current, _,_=self.simulate_S1()
                currents.append(current)
            gradient=(np.divide(np.subtract(currents[1], currents[0]), 2*kwargs["Delta"]))
           
            if i==0:
                
                if len(currents[0])>kwargs["max_size"]:
                    resize=(len(currents[0])//kwargs["max_size"])+1
                    current_len=len(currents[0][0::resize])
                else:
                    resize=False
                    current_len=len(currents[0])
                sensitivity=np.zeros((current_len, len(params)))
            if resize==False:
                sensitivity[:,i]=gradient
            else:
                sensitivity[:,i]=gradient[0::resize]
        return sensitivity

            
        