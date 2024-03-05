import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_TD import EIS_TD
from EIS_class import EIS
Hz=10
num_osc=1
time_end=num_osc/Hz
sf=1/(200000*Hz)
times=np.arange(0, time_end, sf)
phase=3*math.pi/2

maxiter=1000

T=(273+25)
F=96485.3328959
R=8.314459848
FRT=F/(R*T)
potential=0.3*np.sin(2*math.pi*Hz*times+phase)*FRT
lambda_1=0.65
def marcus_kinetics(E, E0, lambda_1, integral_0, flag, k0, inverse_v):
    T=(273+25)
    F=96485.3328959
    R=8.314459848
    FRT=F/(R*T)
    y=FRT*(E-E0)
    
    integral_1=scipy.integrate.romberg(I_theta, a=-50, b=50, args=(lambda_1, y,inverse_v, flag), divmax=20)
    return k0*(integral_1/integral_0)
inverse_v=FRT*lambda_1
def I_theta(x, lambda_1, y, inverse_v, flag):
      
    if flag=="forwards":
        numerator=np.exp(-(inverse_v/4)*(1+((y+x)/inverse_v))**2)  
        denominator=1+np.exp(-x)
    elif flag=="backwards":
        numerator=np.exp(-(inverse_v/4)*(1-((y+x)/inverse_v))**2)  
        denominator=1+np.exp(x)
    return numerator/denominator
integral_0_kf=scipy.integrate.romberg(I_theta, a=-50, b=50, args=(lambda_1, 0, inverse_v, "forwards"), divmax=20)
integral_0_kb=scipy.integrate.romberg(I_theta, a=-50, b=50, args=(lambda_1, 0, inverse_v, "backwards"), divmax=20)
print(integral_0_kb)
E0=0
current=np.zeros(len(potential))

dt=sf
k0=1000
BV_current=np.zeros(len(potential))
def BV_kinetics(E, E0, k0,flag):

    y=(E-E0)
    if flag=="forwards":
        return k0*np.exp(-0.5*y)
    else:
        return k0*np.exp((0.5)*y)
def interaction_kinetics(E, E0, k0,flag):

    y=(E-E0)
    if flag=="forwards":
        return k0*np.exp(-0.5*y)
    else:
        return k0*np.exp((0.5)*y)
gamma=1e-11
a_vals=[1]
aoo=0
arr=0
aor=0

incidental_vals=[ -0.5]
orig_interaction_dict={"arr":0, "aoo":0, "aor":0}
fig, ax=plt.subplots(1,3)
keys=list(orig_interaction_dict.keys())
param_list={
    "E_0":E0,
    'E_start':  -0.15, #(starting dc voltage - V)
    'E_reverse':0.15,
    'omega':9.365311521736066, #8.88480830076,  #    (frequency Hz)
    "original_omega":9.365311521736066,
    'd_E': 299*1e-3,   #(ac voltage amplitude - V) freq_range[j],#
    'area': 0.07, #(electrode surface area cm^2)
    'Ru': 100.0,  #     (uncompensated resistance ohms)
    'Cdl': 1e-4, #(capacitance parameters)
    'CdlE1': 0,#0.000653657774506,
    'CdlE2': 0,#0.000245772700637,
    "CdlE3":0,
    'gamma': 1e-10,
    "original_gamma":1e-10,        # (surface coverage per unit area)
    'k_0': 1000, #(reaction rate s-1)
    'alpha': 0.5,
    "E0_mean":0.2,
    "E0_std": 0.05,
    "E0_skew":0.2,
    "cap_phase":0,
    "alpha_mean":0.5,
    "alpha_std":1e-3,
    'sampling_freq' : (1.0/2**8),
    'phase' :0,
    "time_end": -1,
    'num_peaks': 50,
    "aoo":0,
    "aor":0,
    "arr":0,
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
time_start=2/(param_list["original_omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[32],
    "GH_quadrature":True,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": "scipy",
    "scipy_type":"self_interaction",
    "label": "MCMC",
    "optim_list":[]
}

other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0, 0.4],
    'omega':[0.9*param_list['omega'],1.1*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 3e2],  #     (uncompensated resistance ohms)
    'Cdl': [0,5e-4], #(capacitance parameters)
    'CdlE1': [-0.2,0.2],#0.000653657774506,
    'CdlE2': [-0.1,0.1],#0.000245772700637,
    'CdlE3': [-0.05,0.05],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [10, 7e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "E0_mean":[0, 0.4],
    "E0_std": [1e-4,  0.15],
    "E0_skew": [-10, 10],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    'phase' : [0.8*3*math.pi/2, 1.2*3*math.pi/2],
    "aoo":[-10,10],
    "aor":[-10,10],
    "arr":[-10,10],

}
import copy

ferro=single_electron(None, param_list, simulation_options, other_values, param_bounds)
ferro.def_optim_list(["aoo", "aor", "arr"])
current=ferro.test_vals([0,1,0], "timeseries")
ferro.simulation_options["scipy_type"]="single_electron"
current2=ferro.test_vals([0,0,0], "timeseries")
voltage=ferro.define_voltages()
plt.plot(voltage, current, label="Interacting")
plt.plot(voltage, current2, label="Normal")
plt.legend()
plt.show()
td_params=copy.deepcopy(param_list)
td_params["E_start"]=-10e-3
td_params["d_E"]=10e-3
td=EIS_TD(td_params, copy.deepcopy(simulation_options), copy.deepcopy(other_values), copy.deepcopy(param_bounds))
frequencies=td.define_frequencies(0, 6)
td.def_optim_list(["aoo", "aor", "arr", "Cdl"])
td_vals=td.simulate([0,1,0, td_params["Cdl"]*td_params["area"]], frequencies)
#EIS().nyquist(td_vals)
#plt.show()  

"""for z in range(0, len(keys)):
    parameter=keys[z]
    
    for j in range(0, len(incidental_vals)):
        scan_dict=copy.deepcopy(orig_interaction_dict)
        scan_dict[parameter]=incidental_vals[j]
        aoo=scan_dict["aoo"]
        aor=scan_dict["aor"]
        arr=scan_dict["arr"]
        S=arr-aoo
        G=aoo+arr-(2*aor)
        gamma_max=1
        gamma_tot=1
        theta_e=gamma_tot/gamma_max
        E0_ap=E0+(R*T/F)*theta_e*S

        inter_k0=k0*np.exp(-2*theta_e*aoo)
        #interaction_k0=np.exp(-2*gamma*a_vals[j])
        theta=0
        theta_1=0
        for i in range(1, len(potential)):
            #z=np.linspace(-100, 100, int(1e3))
            #plots=[I_theta(x,lambda_1, potential[i],inverse_v, "backwards") for x in z]
            #plt.plot(z, plots)
            bv_kf=BV_kinetics(potential[i], E0, k0, "forwards")
            bv_kb=BV_kinetics(potential[i], E0, k0, "backwards")
            
        
            current_red=1-theta
            gamma_r=current_red*gamma_tot
            fo=(gamma_tot*theta)/gamma_tot
            fr=gamma_r/gamma_tot
            coeff_1=np.exp(-theta_e*S*fr)
            coeff_2=(fr*np.exp(-0.5*(potential[i]-E0_ap))*np.exp(theta_e*G*(1-fr)))-((1-fr)*np.exp(theta_e*G*fr))
            #dthetadt=interaction_kb*(1-theta)-interaction_kf*theta
            #current[i]=dthetadt

            #dthetadt=BV_kinetics(potential[i], E0_ap, inter_k0, "forwards")*coeff_1*coeff_2
            dthetadt=(BV_kinetics(potential[i],E0_ap, inter_k0, "forwards"))*coeff_1*((np.exp(theta_e*G*(1-fr))*fr*np.exp(potential[i]-E0))-((1-fr)*np.exp(theta_e*G*fr)))
            theta=theta+dt*dthetadt
            current[i]=dthetadt


            if j==0:
                BV_current[i]=(bv_kb*(1-theta_1)-bv_kf*theta_1)
                dthetadt_1=bv_kb*(1-theta_1)-bv_kf*theta_1
                theta_1=theta_1+dt*dthetadt_1
            
    
        if j==0:
            ax[z].plot(potential, BV_current, label="BV", linestyle="--")
        ax[z].plot(potential, current, label="{0}={1}".format(parameter, incidental_vals[j]))
    ax[z].legend()

plt.show()"""