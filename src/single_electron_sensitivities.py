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
        self.sensitivity_array=[]
        self.t_array=[]
        self.current_t=0
    def Sensitivity_ODE_system(self, state_var, time):
        I, theta, E =state_var
        E_0, k_0, Ru, Cdl, CdlE1, CdlE2, CdlE3, gamma, alpha=[self.nd_param.nd_param_dict[x] for x in self.sens_params]
        
            
        
        
        sensitivities=np.zeros(self.num_sens)
        Er=(E - I*Ru)
        ErE0=(E - E_0 - I*Ru)
        return_vars=self.current_ode_sys(state_var, time)
        dI=return_vars[0]
        dE=return_vars[1]
        #E_0
        sensitivities[0]=gamma*(alpha*k_0*theta*np.exp(alpha*ErE0) + k_0*(1 - theta)*(alpha - 1)*np.exp((1 - alpha)*ErE0))
        #k_0
        sensitivities[1]=gamma*(-theta*np.exp(alpha*ErE0) + (1 - theta)*np.exp((1 - alpha)*ErE0))
        #Ru
        sensitivities[2]=-Cdl*dI*(CdlE1*Er + CdlE2*Er**2 + CdlE3*Er**3 + 1) + Cdl*(-Ru*dI + dE)*(-CdlE1*I - 2*CdlE2*I*Er - 3*CdlE3*I*Er**2) + gamma*(I*alpha*k_0*theta*np.exp(alpha*ErE0) - I*k_0*(1 - alpha)*(1 - theta)*np.exp((1 - alpha)*ErE0))
        #Cdl
        sensitivities[3]=(-Ru*dI + dE)*(CdlE1*Er + CdlE2*Er**2 + CdlE3*Er**3 + 1)
        #CdlE1
        sensitivities[4]=Cdl*Er*(-Ru*dI + dE)
        #CdlE2
        sensitivities[5]=Cdl*Er**2*(-Ru*dI + dE)
        #CdlE3
        sensitivities[6]=Cdl*Er**3*(-Ru*dI + dE)
        #gamma
        sensitivities[7]=-k_0*theta*np.exp(alpha*ErE0) + k_0*(1 - theta)*np.exp((1 - alpha)*ErE0)
        #alpha
        sensitivities[8]=gamma*(-k_0*theta*ErE0*np.exp(alpha*ErE0) + k_0*(1 - theta)*(-E + E_0 + I*Ru)*np.exp((1 - alpha)*ErE0))
        if self.current_t>time:
            self.sensitivity_array[-1]=sensitivities
            self.t_array[-1]=time
        else:
            self.sensitivity_array.append(sensitivities)
            self.t_array.append(time)
        self.current_t=time
        return return_vars
    def simulate_S1(self):


        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = self.time_vec[-1]
        numpoints = len(self.time_vec)

        w0 = [0,0, self.voltage_query(0)[0]]
        # Call the ODE solver.

        wsol = odeint(self.Sensitivity_ODE_system, w0, self.time_vec,
                      atol=abserr, rtol=relerr)
        return wsol[:,0], wsol[:,1], wsol[:,2]
    def get_FIM(self, noise):
        
        covariance=np.linalg.inv(noise*np.identity(len(self.t_array)))
        scaled=np.matmul(np.transpose(self.sensitivity_array), covariance)
        FIM=np.matmul(scaled, self.sensitivity_array)
        return FIM
    def D_optimality(self, **kwargs):
        if "FIM" not in kwargs:
            if "noise" not in kwargs:
                raise ValueError("Need to provide either the FIM or an estimate of the noise")
            else:
                FIM=self.get_FIM(kwargs["noise"])
                return np.linalg.det(FIM)
        else:
            return np.linalg.det(kwargs["FIM"])        
        
