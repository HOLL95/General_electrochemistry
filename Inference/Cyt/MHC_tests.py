import scipy
import numpy as np
import matplotlib.pyplot as plt
import math
Hz=10
num_osc=1
time_end=num_osc/Hz
sf=1/(2000*Hz)
times=np.arange(0, time_end, sf)
phase=3*math.pi/2
potential=0.3*np.sin(2*math.pi*Hz*times+phase)
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
theta=0
dt=sf
k0=0.1              
BV_current=np.zeros(len(potential))
def BV_kinetics(E, E0, k0,flag):
    T=(273+25)
    F=96485.3328959
    R=8.314459848
    FRT=F/(R*T)
    y=FRT*(E-E0)
    if flag=="forwards":
        return k0*np.exp(-0.5*y)
    else:
        return k0*np.exp((0.5)*y)
for i in range(1, len(potential)):
    #z=np.linspace(-100, 100, int(1e3))
    #plots=[I_theta(x,lambda_1, potential[i],inverse_v, "backwards") for x in z]
    #plt.plot(z, plots)
    bv_kf=BV_kinetics(potential[i], E0, k0, "forwards")
    bv_kb=BV_kinetics(potential[i], E0, k0, "backwards")
    kf=marcus_kinetics(potential[i], E0, lambda_1, integral_0_kf, "forwards", k0, inverse_v)
    kb=marcus_kinetics(potential[i], E0, lambda_1, integral_0_kb, "backwards", k0, inverse_v)
    
    dthetadt=kb*(1-theta)-kf*theta
    current[i]=dthetadt
    BV_current[i]=bv_kb*(1-theta)-bv_kf*theta
    theta=theta+dt*dthetadt

plt.plot(potential, current)
plt.plot(potential, BV_current)
plt.show()