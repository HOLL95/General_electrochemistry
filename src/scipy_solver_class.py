try:
    import isolver_martin_brent
    import SWV_surface
except:
    pass
from scipy.stats import norm, lognorm
import math
import numpy as np
import itertools
from params_class import params
from dispersion_class import dispersion
from scipy.integrate import odeint
from decimal import Decimal
import copy
import warnings
import re
import matplotlib.pyplot as plt
class scipy_funcs:
    def __init__(self, current_class):
        self.current_class=current_class
        self.nd_param=current_class.nd_param
    def Armstrong_dcv_current(self, times, dcv_voltages):
        T=(273+25)
        F=96485.3328959
        R=8.314459848
        first_denom=(F**2*self.current_class.dim_dict["area"]*self.current_class.dim_dict["v"]*self.current_class.dim_dict["gamma"])/(R*T)
        current=np.zeros(len(times))
        for i in range(0, len(times)):
            exponent=np.exp(F*(dcv_voltages[i]-self.current_class.dim_dict["E_0"])/(R*T))
            current[i]=(exponent/(1+(exponent**2)))*first_denom
        return current
    def voltage_query(self, time):
        if self.current_class.simulation_options["method"]=="sinusoidal":
                Et=isolver_martin_brent.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
                dEdt=isolver_martin_brent.dEdt(self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
        elif self.current_class.simulation_options["method"]=="ramped":
                Et=-1*isolver_martin_brent.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
                dEdt=-1*isolver_martin_brent.c_dEdt(self.nd_param.nd_param_dict["tr"] ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
        elif self.current_class.simulation_options["method"]=="dcv":
                Et=isolver_martin_brent.dcv_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) , 1,time)
                dEdt=isolver_martin_brent.dcv_dEdt(self.nd_param.nd_param_dict["tr"],1,time)
        elif self.current_class.simulation_options["method"]=="square_wave":
                time_position=int(np.ceil(time))
                Et=SWV_surface.potential(time_position, self.nd_param.nd_param_dict["sampling_factor"], self.nd_param.nd_param_dict["scan_increment"],
                                                        self.nd_param.nd_param_dict["SW_amplitude"], self.nd_param.nd_param_dict["E_start"])/self.nd_param.sw_class.c_E0

                dEdt=None
        elif self.current_class.simulation_options["method"]=="square_wave_fourier":
                Et=SWV_surface.fourier_Et(math.pi, self.nd_param.nd_param_dict["scan_increment"], self.nd_param.nd_param_dict["fourier_order"], self.nd_param.nd_param_dict["SW_amplitude"], self.nd_param.nd_param_dict["E_start"], time)
                dEdt=SWV_surface.fourier_dEdt(math.pi, self.nd_param.nd_param_dict["scan_increment"], self.nd_param.nd_param_dict["fourier_order"], self.nd_param.nd_param_dict["SW_amplitude"], time)
                #print(Et)
                Et=Et/self.nd_param.sw_class.c_E0
                dEdt=dEdt/self.nd_param.sw_class.c_E0
        return Et, dEdt
    def current_ode_sys(self, state_vars, time):
        current, theta, potential=state_vars
        Et, dEdt=self.voltage_query(time)
        alpha=self.nd_param.nd_param_dict["alpha"]
        if "square_wave" not in self.current_class.simulation_options["method"]:
            Er=Et-(self.nd_param.nd_param_dict["Ru"]*current)
            ErE0=Er-self.nd_param.nd_param_dict["E_0"]
            self.Cdlp=self.nd_param.nd_param_dict["Cdl"]*(1+self.nd_param.nd_param_dict["CdlE1"]*Er+self.nd_param.nd_param_dict["CdlE2"]*(Er**2)+self.nd_param.nd_param_dict["CdlE3"]*(Er**3))
            if self.current_class.simulation_options["scipy_type"]=="single_electron":
                d_thetadt=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
                dIdt=(dEdt-(current/self.Cdlp)+self.nd_param.nd_param_dict["gamma"]*d_thetadt*(1/self.Cdlp))/self.nd_param.nd_param_dict["Ru"]

            elif self.current_class.simulation_options["scipy_type"]=="EC":
                d_thetadt=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))-(self.nd_param.nd_param_dict["k_1"]*(theta))+(self.nd_param.nd_param_dict["k_2"]*(1-theta))
                farad_current=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
                dIdt=(dEdt-(current/self.Cdlp)+self.nd_param.nd_param_dict["gamma"]*farad_current*(1/self.Cdlp))/self.nd_param.nd_param_dict["Ru"]
        else:
            if self.current_class.simulation_options["method"]=="square_wave":
                dEdt=(Et-potential)/self.nd_param.nd_param_dict["CmRs"]
                Er=potential-(self.nd_param.nd_param_dict["Ru"]*current)
            elif self.current_class.simulation_options["method"]=="square_wave_fourier":
                Er=Et-(self.nd_param.nd_param_dict["Ru"]*current)
            ErE0=Er-((self.nd_param.nd_param_dict["E_0"])/self.nd_param.sw_class.c_E0)
            self.Cdlp=self.nd_param.nd_param_dict["Cdl"]*(1+self.nd_param.nd_param_dict["CdlE1"]*Er+self.nd_param.nd_param_dict["CdlE2"]*(Er**2)+self.nd_param.nd_param_dict["CdlE3"]*(Er**3))
            d_thetadt=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
            if self.Cdlp!=0:
                dIdt=(dEdt-(current/self.Cdlp)+self.nd_param.nd_param_dict["gamma"]*d_thetadt*(1/self.Cdlp))/self.nd_param.nd_param_dict["Ru"]
            else:
                dIdt=self.nd_param.nd_param_dict["gamma"]*d_thetadt

        f=[dIdt, d_thetadt, dEdt]
        return f
    def simulate_timeseries(self):


        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = self.current_class.time_vec[-1]
        numpoints = len(self.current_class.time_vec)

        w0 = [0,0, self.voltage_query(0)[0]]
        # Call the ODE solver.

        wsol = odeint(self.current_ode_sys, w0, self.current_class.time_vec,
                      atol=abserr, rtol=relerr)
        return wsol[:,0], wsol[:,1], wsol[:,2]
    def simulate_current(self,nd_param_dict=None, *args):
        if nd_param_dict!=None:
            self.nd_param.nd_param_dict=nd_param_dict
        current,theta, potential=self.simulate_timeseries()
        volts=self.current_class.define_voltages()
        self.potential=potential
        return current
    def simulate_sensitivities(self):
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = self.current_class.time_vec[-1]
        numpoints = len(self.current_class.time_vec)

        w0 = [0,0, self.voltage_query(0)[0]]+[0]*5*3
        print(w0)
        # Call the ODE solver.

        wsol = odeint(self.system_sensitivity, w0, self.current_class.time_vec,
                      atol=abserr, rtol=relerr)

        return wsol


    def system_sensitivity(self, state_vars, time):
        I=state_vars[0]
        theta=state_vars[1]
        potential=state_vars[2]
        current_sensitivities=np.array(state_vars[3:])
        Et, dEdt=self.voltage_query(time)


        #theta, Et, I
        #dIdt, Dedt, dthetadt,

        Er=Et-(self.nd_param.nd_param_dict["Ru"]*I)
        ErE0=Er-self.nd_param.nd_param_dict["E_0"]
        E_0=self.nd_param.nd_param_dict["E_0"]
        alpha=self.nd_param.nd_param_dict["alpha"]
        exp11=self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0)
        exp12=self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0)
        Cdl=self.nd_param.nd_param_dict["Cdl"]#*(1+self.nd_param.nd_param_dict["CdlE1"]*Er+self.nd_param.nd_param_dict["CdlE2"]*(Er**2)+self.nd_param.nd_param_dict["CdlE3"]*(Er**3))
        self.Cdlp=self.nd_param.nd_param_dict["Cdl"]#*(1+self.nd_param.nd_param_dict["CdlE1"]*Er+self.nd_param.nd_param_dict["CdlE2"]*(Er**2)+self.nd_param.nd_param_dict["CdlE3"]*(Er**3))
        d_thetadt=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
        dIdt=(dEdt-(I/self.Cdlp)+self.nd_param.nd_param_dict["gamma"]*d_thetadt*(1/self.Cdlp))/self.nd_param.nd_param_dict["Ru"]
        jacobian=np.zeros((3, 3))
        k_0=self.nd_param.nd_param_dict["k_0"]
        Ru=self.nd_param.nd_param_dict["Ru"]
        gamma=self.nd_param.nd_param_dict["gamma"]
        alpha_minus_1=1-alpha
        #dI/dI, dE/dI, dtheta/dI
        jacobian[0,0]=(gamma*(-Ru*alpha*theta*exp12 - Ru*(1 - alpha)*(1 - theta)*exp11)/Cdl - 1/Cdl)/Ru
        jacobian[0,1]=0
        jacobian[0, 2]=-Ru*alpha*theta*exp12 - Ru*(1 - alpha)*(1 - theta)*exp11
        #dI/dE, dE/dE, dtheta/dE
        jacobian[1,0]=gamma*(alpha*theta*exp12 + (1 - alpha)*(1 - theta)*exp11)/(Cdl*Ru)
        jacobian[1,1]=0
        jacobian[1,2]=alpha*theta*exp12 + (1 - alpha)*(1 - theta)*exp11
        #dI/dtheta, dE/dtheta, dtheta/dtheta
        jacobian[2,0]=gamma*(-exp11 - exp12)/(Cdl*Ru)
        jacobian[2,1]=0
        jacobian[2,2]=-exp11 - exp12
        #DX/DE_0
        exp_11_no_k=np.exp((1-alpha)*ErE0)
        exp_12_no_k=np.exp((-alpha)*ErE0)
        sensitivity_vec=np.zeros((3, 5))

        #DX/DE_0
        sensitivity_vec[0,0]=gamma*(-alpha*k_0*theta*exp_12_no_k + k_0*(1 - theta)*(alpha - 1)*exp_11_no_k)/(Cdl*Ru)

        sensitivity_vec[2,0]=-alpha*k_0*theta*exp_12_no_k + k_0*(1 - theta)*(alpha - 1)*exp_11_no_k
        #DX/Dk_0
        sensitivity_vec[0,1]=gamma*(-theta*exp_12_no_k + (1 - theta)*exp_11_no_k)/(Cdl*Ru)

        sensitivity_vec[2,1]=-theta*exp_12_no_k + (1 - theta)*exp_11_no_k
        #DX/Dalpha
        sensitivity_vec[0,2]=gamma*(-k_0*theta*(-Et + E_0 + I*Ru)*exp_12_no_k + k_0*(1 - theta)*(-Et + E_0 + I*Ru)*exp_11_no_k)/(Cdl*Ru)

        sensitivity_vec[2,2]=-k_0*theta*(-Et + E_0 + I*Ru)*exp_12_no_k + k_0*(1 - theta)*(-Et + E_0 + I*Ru)*exp_11_no_k
        #DX/Dru
        sensitivity_vec[0,3]=-(dEdt - I/Cdl + gamma*(-k_0*theta*exp_12_no_k + k_0*(1 - theta)*exp_11_no_k)/Cdl)/Ru**2 + gamma*(-I*alpha*k_0*theta*exp_12_no_k - I*k_0*(1 - alpha)*(1 - theta)*exp_11_no_k)/(Cdl*Ru)

        sensitivity_vec[2,3]=-I*alpha*k_0*theta*exp_12_no_k - I*k_0*(1 - alpha)*(1 - theta)*exp_11_no_k
        #DX/DCdl
        sensitivity_vec[0,4]=(I/Cdl**2 - gamma*(-k_0*theta*exp_12_no_k + k_0*(1 - theta)*exp_11_no_k)/Cdl**2)/Ru
        z=0
        returned_derivatives=np.zeros(len(state_vars))
        for i in range(0, len(current_sensitivities), 3):
            returned_derivatives[i+3:i+6]=((np.matmul(jacobian, current_sensitivities[i:i+3].reshape(3,1))+sensitivity_vec[:,z].reshape(3,1))).reshape(1,3)
            z+=1
        returned_derivatives[0:3]=[dIdt, d_thetadt, dEdt]
        return returned_derivatives
