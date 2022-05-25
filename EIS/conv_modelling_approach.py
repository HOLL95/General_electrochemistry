import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from collections import deque
from params_class import params
from single_e_class_unified import single_electron
import mpmath
class conv_model(single_electron):
    def linear_ramp_inv(self, t):
        inv_lap_func=lambda s: 1/(s**2*(s**(-self.nd_param.nd_param_dict["psi"])+self.nd_param.nd_param_dict["tau"]))
        inv_lap_val=mpmath.invertlaplace(inv_lap_func,t,method='talbot')
        if t<self.dim_dict["tr"]:
            return inv_lap_val
        else:
            return -inv_lap_val
    def W(self, t):
        inv_lap_func=lambda s: 1/(s*(1+self.nd_param.nd_param_dict["tau"]*s**self.nd_param.nd_param_dict["psi"]))
        inv_lap_val=mpmath.invertlaplace(inv_lap_func,t,method='talbot')
        return inv_lap_val
    def B(self, t):
        inv_lap_func=lambda s: 1/(s*(self.nd_param.nd_param_dict["tau"]+s**-self.nd_param.nd_param_dict["psi"]))
        inv_lap_val=mpmath.invertlaplace(inv_lap_func,t,method='talbot')
        return inv_lap_val
    def E(self, t):

        if t<self.dim_dict["tr"]:
            E_dc=self.nd_param.nd_param_dict["E_start"]+(t);
        else:
            E_dc=self.nd_param.nd_param_dict["E_reverse"]-((t-self.dim_dict["tr"]))
        return E_dc+(self.nd_param.nd_param_dict["d_E"]*(np.sin((self.nd_param.nd_param_dict["nd_omega"]*t))))
    def theta_calc(self, I_f, t, I_t):
        #could also maybe use finite diffs?
        alpha=self.nd_param.nd_param_dict["alpha"]
        ErE0=self.E(t)-self.nd_param.nd_param_dict["Ru"]*I_t
        numerator=I_f-self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0)
        denom=self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0)+self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0)
        return numerator/denom
    def I_f(self, theta, t, I_t):
        alpha=self.nd_param.nd_param_dict["alpha"]
        ErE0=self.E(t)-self.nd_param.nd_param_dict["Ru"]*I_t
        d_thetadt=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
        return d_thetadt
    def Farad_calc(self,):
        if self.t_counter==0:
            return 0
        summand=np.zeros(self.t_counter)

        for i in range(0, self.t_counter):
            summand[i]=self.W_array[self.t_counter-i]*(self.If_array[i+1]-self.If_array[i])
        return np.sum(summand)

    def V_t_calc(self, t):
        first_summand=(self.dt/2)*(self.B_array[0]*self.cos_array[self.t_counter]+self.B_array[self.t_counter])
        second_summand=np.zeros(self.t_counter)

        for i in range(0, self.t_counter):
            second_summand[i]=self.B_array[self.t_counter-i]*self.cos_array[i]
        total_summand=first_summand+self.dt*np.sum(second_summand)

        return self.nd_param.nd_param_dict["Cdl"]*self.nd_param.nd_param_dict["nd_omega"]*self.nd_param.nd_param_dict["d_E"]*total_summand
    def residual(self, x):
        i_t=x[0]
        current_t=self.time_vec[self.t_counter]
        self.If_array[self.t_counter]=self.I_f(self.theta, current_t, i_t)
        #linear_c=self.linear_ramp_inv(current_t)
        #oscillatory_c=self.V_t_calc(current_t)
        #print(linear_c, oscillatory_c)
        #Farad_componenet=self.Farad_calc()
        #result=linear_c-i_t+oscillatory_c

        result=self.If_array[self.t_counter]-i_t
        #self.theta=self.theta+self.dt*self.If_array[self.t_counter]
        return result
    def simulate_current(self,):
        self.dt=self.nd_param.nd_param_dict["sampling_freq"]
        self.nd_param.nd_param_dict["tau"]=self.nd_param.nd_param_dict["Ru"]*self.nd_param.nd_param_dict["Cdl"]
        self.nd_param.nd_param_dict["vCpe"]=self.nd_param.nd_param_dict["Cdl"]
        #do parameter stuff here
        self.B_array=np.zeros(len(self.time_vec))
        self.W_array=np.zeros(len(self.time_vec))
        for i in range(1, len(self.time_vec)*0):
            print(i, len(self.time_vec))
            self.B_array[i]=self.B(self.time_vec[i])
            self.W_array[i]=self.W(self.time_vec[i])
        self.If_array=np.zeros(len(self.time_vec))
        self.cos_array=np.cos(self.nd_param.nd_param_dict["nd_omega"]*self.time_vec)
        self.theta=1
        self.total_current=np.zeros(len(self.time_vec))
        prev_current=0
        for i in range(1, len(self.time_vec)):
            print(i, len(self.time_vec))
            self.t_counter=i
            self.total_current[i]=fsolve(self.residual, prev_current)
            prev_current=self.total_current[i]
            self.If_array[i]=self.I_f(self.theta, self.time_vec[i], self.total_current[i])
            self.theta=self.theta_calc(self.If_array[i], self.time_vec[i], self.total_current[i])
        return self.total_current
param_list={
    "E_0":0.25,
    'E_start':  0.0, #(starting dc voltage - V)
    'E_reverse':0.5,
    'omega':10, #8.88480830076,  #    (frequency Hz)
    "v":0.25,
    'd_E': 150*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 0.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4*0, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "psi":0.5,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10    , #(reaction rate s-1)
    'alpha': 0.5,
    "cap_phase":0,
    'sampling_freq' : (1.0/800),
    'phase' :0,
}
sim_options={
    "method":"ramped",
    "experimental_fitting":False,
    "likelihood":"timeseries"
}
test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
print(test_class.nd_param.nd_param_dict["v"])
plt.plot(test_class.t_nondim(test_class.time_vec), test_class.test_vals([], "timeseries"))
#plt.plot(test_class.t_nondim(test_class.time_vec), test_class.define_voltages())
current=test_class.simulate_current()
plt.plot(test_class.t_nondim(test_class.time_vec),current)
plt.show()
