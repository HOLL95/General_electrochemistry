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
import isolver_martin_brent
import mpmath
class conv_model(single_electron):
    def linear_ramp_inv(self, t):
        inv_lap_func=lambda s: 1/(s**2*(s**(-self.nd_param.nd_param_dict["psi"])+self.nd_param.nd_param_dict["tau"]))
        inv_lap_val=mpmath.invertlaplace(inv_lap_func,t,method='talbot')
        if t<self.dim_dict["tr"]:
            return self.nd_param.nd_param_dict["Cdl"]*inv_lap_val
        else:
            return -self.nd_param.nd_param_dict["Cdl"]*inv_lap_val
    def linear_cdl_t(self):
        cdl=np.zeros(len(self.time_vec))
        for i in range(0, len(self.time_vec)):
            cdl[i]=self.nd_param.nd_param_dict["Cdl"]*isolver_martin_brent.c_dEdt(self.nd_param.nd_param_dict["tr"] ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],self.time_vec[i])
        return cdl
    def linear_ramp_anal(self, t):
        if t<self.dim_dict["tr"]:
            return self.nd_param.nd_param_dict["Cdl"]*(1-np.exp(-t/self.nd_param.nd_param_dict["tau"]))
        else:
            return -self.nd_param.nd_param_dict["Cdl"]*(1-np.exp(-t/self.nd_param.nd_param_dict["tau"]))
    def V_t_anal(self, t):
        numerator=np.cos(self.nd_param.nd_param_dict["nd_omega"]*t)+self.nd_param.nd_param_dict["nd_omega"]*self.nd_param.nd_param_dict["tau"]*np.cos(self.nd_param.nd_param_dict["nd_omega"]*t)-np.exp(-t/self.nd_param.nd_param_dict["tau"])
        denom=1+(self.nd_param.nd_param_dict["nd_omega"]**2)*(self.nd_param.nd_param_dict["tau"]**2)
        return self.nd_param.nd_param_dict["Cdl"]*self.nd_param.nd_param_dict["nd_omega"]*self.nd_param.nd_param_dict["d_E"]*numerator/denom
    def W(self, t):
        inv_lap_func=lambda s: 1/(s*(1+self.nd_param.nd_param_dict["tau"]*s**self.nd_param.nd_param_dict["psi"]))
        inv_lap_val=mpmath.invertlaplace(inv_lap_func,t,method='talbot')
        return inv_lap_val
    def W_anal(self, t):
        return 1-np.exp(-t/self.nd_param.nd_param_dict["tau"])
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
        if t==0:
            return 0
        Er=self.E(t)-(self.nd_param.nd_param_dict["Ru"]*I_t)
        ErE0=Er-self.nd_param.nd_param_dict["E_0"]
        alpha=self.nd_param.nd_param_dict["alpha"]
        I_f=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
        return I_f
    def Farad_calc(self,):
        if self.t_counter==0:
            return 0
        summand=np.zeros(self.t_counter)

        for i in range(0, self.t_counter):
            summand[i]=self.W_array[self.t_counter-i]*(self.If_array[i+1]-self.If_array[i])
        return np.sum(summand)
    def Farad_calc_anal(self,):
        if self.t_counter==0:
            return 0
        summand=np.zeros(self.t_counter)

        for i in range(0, self.t_counter):
            summand[i]=self.W_array[self.t_counter-i]*(self.If_array[i+1]-self.If_array[i])
        return np.sum(summand)
    def theta_calc(self, t,current):
        Et=self.E(t)
        Er=Et-(self.nd_param.nd_param_dict["Ru"]*current)
        ErE0=Er-self.nd_param.nd_param_dict["E_0"]
        alpha=self.nd_param.nd_param_dict["alpha"]
        theta=(self.theta+self.dt*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))/(1+self.dt*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0)+self.dt*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
        return theta
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
        if self.CPE_sim==True:
            linear_c=self.linear_ramp_inv(current_t)
            #oscillatory_c=self.V_t_calc(current_t)
            Farad_componenet=self.Farad_calc_anal()
        else:
            linear_c=self.linear_ramp_anal(current_t)
            #oscillatory_c=self.V_t_anal(current_t)
            Farad_componenet=self.Farad_calc()
        #print(linear_c, oscillatory_c)

        result=linear_c-i_t+Farad_componenet#oscillatory_c+

        #result=self.If_array[self.t_counter]-i_t
        #self.theta=self.theta+self.dt*self.If_array[self.t_counter]
        return result
    def simulate_current(self,**kwargs):
        if "CPE" not in kwargs:
            self.CPE_sim=False
        else:
            self.CPE_sim=kwargs["CPE"]
        self.dt=self.nd_param.nd_param_dict["sampling_freq"]
        self.nd_param.nd_param_dict["tau"]=self.nd_param.nd_param_dict["Ru"]*self.nd_param.nd_param_dict["Cdl"]
        self.nd_param.nd_param_dict["vCpe"]=self.nd_param.nd_param_dict["Cdl"]
        #do parameter stuff here
        self.B_array=np.zeros(len(self.time_vec))
        self.W_array=np.zeros(len(self.time_vec))
        self.If_array=np.zeros(len(self.time_vec))
        if kwargs["CPE"]==True:
            for i in range(1, len(self.time_vec)):
                print(i, len(self.time_vec))
                self.B_array[i]=self.B(self.time_vec[i])
                self.W_array[i]=self.W(self.time_vec[i])

            self.cos_array=np.cos(self.nd_param.nd_param_dict["nd_omega"]*self.time_vec)
        else:
            for i in range(1, len(self.time_vec)):
                self.W_array[i]=self.W_anal(self.time_vec[i])
        self.theta=0
        self.total_current=np.zeros(len(self.time_vec))
        prev_current=0
        for i in range(0, len(self.time_vec)):
            print(i, len(self.time_vec))
            self.t_counter=i
            if i!=0:
                self.total_current[i]=fsolve(self.residual, self.total_current[i-1])
                self.If_array[i]=self.I_f(self.theta, self.time_vec[i], self.total_current[i])
            self.theta=self.theta_calc(self.time_vec[i],self.total_current[i])
        return self.total_current
param_list={
    "E_0":0.25,
    'E_start':  0.0, #(starting dc voltage - V)
    'E_reverse':0.5,
    'omega':10, #8.88480830076,  #    (frequency Hz)
    "v":0.25,
    'd_E': 150*1e-3*0,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 10.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-6, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-11,
    "psi":0.5,
    "original_gamma":1e-11,        # (surface coverage per unit area)
    'k_0': 10    , #(reaction rate s-1)
    'alpha': 0.5,
    "cap_phase":0,
    'sampling_freq' : (1.0/200),
    'phase' :0,
}
import copy
orig_param_list=copy.deepcopy(param_list)
sim_options={
    "method":"ramped",
    "experimental_fitting":False,
    "likelihood":"timeseries"
}
save_dict={}
params=["psi", "Ru", "Cdl"]
vals=[[0.3, 0.5, 0.7], [10, 100, 1000], [1e-6, 1e-5, 1e-4]]
val_dict=dict((zip(params, vals)))
for i in range(0, len(params)):
    param_list=orig_param_list
    key=params[i]
    for j in range(0, len(val_dict[key])):
        value=val_dict[params[i]][j]
        param_list[key]=value
        test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
        numerical_ts=test_class.test_vals([], "timeseries")
        #plt.plot(test_class.t_nondim(test_class.time_vec), numerical_ts)
        #plt.plot(test_class.t_nondim(test_class.time_vec), )
        ##plt.plot(test_class.t_nondim(test_class.time_vec), test_class.define_voltages())

        current=test_class.simulate_current(CPE=True)
        plt.plot(current)
        plt.show()
        save_key=key+"="+str(value)
        save_dict[save_key]={}
        save_dict[save_key]["current"]=test_class.i_nondim(current)
        save_dict[save_key]["time"]=test_class.t_nondim(test_class.time_vec)
        save_dict[save_key]["potential"]=test_class.e_nondim(test_class.define_voltages())
        save_dict[save_key]["params"]=param_list
        save_dict[save_key]["numerical"]=test_class.i_nondim(numerical_ts)
        #plt.plot(test_class.t_nondim(test_class.time_vec),current, label=psi)
        #voltage=[test_class.E(t) for t in test_class.time_vec]
        #plt.plot(voltage)
np.save("FTACV_CPE_param_scans", save_dict)
plt.legend()
plt.show()
