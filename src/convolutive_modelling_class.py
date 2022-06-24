import mpmath
import numpy as np
from scipy.optimize import fsolve
from single_e_class_unified import single_electron
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
        ErE0=self.potential_vec[self.t_counter]-self.nd_param.nd_param_dict["Ru"]*I_t
        numerator=I_f-self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0)
        denom=self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0)+self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0)
        return numerator/denom
    def I_f(self, theta, t, I_t):
        if t==0:
            return 0
        Er=self.potential_vec[self.t_counter]-(self.nd_param.nd_param_dict["Ru"]*I_t)
        ErE0=Er-self.nd_param.nd_param_dict["E_0"]
        alpha=self.nd_param.nd_param_dict["alpha"]
        I_f=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
        return I_f
    def Farad_calc(self,):
        if self.t_counter<2:
            return 0
        return  np.sum(self.W_array[0:self.t_counter-2:]*(self.If_array[self.t_counter-1:1:-1] - self.If_array[self.t_counter-2:0:-1]))

    def theta_calc(self, t,current):
        Et=self.potential_vec[self.t_counter]
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
        current_if=self.I_f(self.theta, current_t, i_t)
        self.I_conv=self.Farad_calc()
        farad_quantity=(current_if-self.If_array[self.t_counter-1])/self.W_array[1] + self.I_conv
        if  self.CPE_sim==True:
            oscillation_func=self.V_t_calc
            ramp_func=self.linear_ramp_inv
        else:
            oscillation_func=self.V_t_anal
            ramp_func=self.linear_ramp_anal
        if self.simulation_options["method"]=="ramped":
            self.cap_current=ramp_func(current_t)+oscillation_func(current_t)
            result=self.cap_current+farad_quantity-i_t
        elif self.simulation_options["method"]=="dcv":
            self.cap_current=ramp_func(current_t)
            result=self.cap_current+farad_quantity-i_t
        elif self.simulation_options["method"]=="sinusoidal":
            self.cap_current=oscillation_func(current_t)
            result=self.cap_current+farad_quantity-i_t
        return result
    def simulate_current(self,**kwargs):
        if "CPE" not in kwargs:
            self.CPE_sim=True
        else:
            self.CPE_sim=kwargs["CPE"]
        self.dt=self.nd_param.nd_param_dict["sampling_freq"]
        self.nd_param.nd_param_dict["tau"]=self.nd_param.nd_param_dict["Ru"]*self.nd_param.nd_param_dict["Cdl"]
        self.nd_param.nd_param_dict["vCpe"]=self.nd_param.nd_param_dict["Cdl"]
        self.potential_vec=self.define_voltages()
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
                self.If_array[i]=((self.total_current[i]-self.cap_current-self.I_conv)/self.W_array[1])+self.If_array[i-1]
            self.theta=self.theta_calc(self.time_vec[i],self.total_current[i])
        return self.total_current
