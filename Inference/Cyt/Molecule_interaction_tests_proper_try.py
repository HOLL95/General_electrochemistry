import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
Hz=10
num_osc=1
time_end=num_osc/Hz
sf=1/(200000*Hz)
times=np.arange(0, time_end, sf)
phase=3*math.pi/2

maxiter=1000
"""def marcus_kinetics(E, E0, lambda_1, integral_0, flag):
    y=E-E0
    if flag=="forward":
        coefficient=25*1000*96485*1.26e-12*np.exp(-0.5*42.51*y)
    elif flag=="backward":
        coefficient=25*1000*96485*1.26e-12*np.exp(0.5*42.51*y)
    integral_1=scipy.integrate.quadrature(I_theta, a=-10000, b=10000, args=(lambda_1, y), maxiter=maxiter)
    return coefficient*(integral_1/integral_0)

def I_theta(x, lambda_1, y):
    return np.exp(-((x-42.51*y)**2)/(170.04*lambda_1))/(2*np.cosh(x/2))

lambda_1=0.65
integral_0=scipy.integrate.romberg(I_theta, a=-10000, b=10000, args=(lambda_1, 0), divmax=100)
spread=np.linspace(-10000, 10000, int(1e5))
cosh=np.cosh(spread)
cosh[np.isinf(cosh)]=0
print(cosh)"""
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

incidental_vals=[-1, -0.5, 0.5, 1]
orig_interaction_dict={"arr":0, "aoo":0, "aor":0}
fig, ax=plt.subplots(1,3)
keys=list(orig_interaction_dict.keys())
for z in range(0, len(keys)):
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

plt.show()