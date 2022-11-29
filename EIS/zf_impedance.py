# -*- coding: utf-8 -*-
"""
@author: Tim Tichter
"""


import numpy as np
import matplotlib.pyplot as plt
from math import floor
from cmath import *
from scipy.interpolate import InterpolatedUnivariateSpline
#from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy.optimize import fsolve
import scipy


#=================================================================================
#=================================================================================
#Define some plotting specific Parameters - make plots look nice :)
#=================================================================================
#=================================================================================

#font = {'family': 'Times New Roman', 'color':  'black','weight': 'normal','size': 15,}
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
#plt.rcParams['font.sans-serif'] = ['Times new Roman']



#=======================================================================
#Everything to Talbot
#=======================================================================

def cot(phi):
    return 1.0/tan(phi)

def csc(phi):
    return 1.0/sin(phi)

def coth(x):
    return 1/tanh(x)

def ContourIntegral(t, N, LapFunc):
    h = 2*np.pi/N;
    shift = 0.0;
    ans   = 0.0;
    if t == 0:
        print ("ERROR:   Inverse transform can not be calculated for t=0")
        return ("Error");
    for k in range(0,N):
        theta  = -np.pi + (k+1./2)*h;
        z      = shift + N/t*(0.5017*theta*cot(0.6407*theta) - 0.6122 + 0.2645j*theta);
        dz     = N/t*(-0.5017*0.6407*theta*(csc(0.6407*theta)**2)+0.5017*cot(0.6407*theta)+0.2645j);
        ans    = ans + exp(z*t)*(LapFunc(z))*dz;
    return ((h/(2j*np.pi))*ans).real


def Talbot(LapFunc):
    t_Array             = np.logspace(-6,6,400)    #np.logspace(-8,5,300)
    ft_Array            = np.empty(len(t_Array))
    for i in range(len(t_Array)):
        ft_Array[i]     = ContourIntegral(float(t_Array[i]), 24, LapFunc)
    TimeDomainFunc      = InterpolatedUnivariateSpline(t_Array, ft_Array, k=3)
    return TimeDomainFunc

#=================================================================================
#=================================================================================
#Define Impedance function for comparison - here planar semi-infinite
#=================================================================================
#=================================================================================


def PlanarSemiinfImpCalculator(freqs, Ru, EZero, Eeq, Dred, Dox, Cap, kp, kmp, kf, kmf, gamma):
    Freq    = 2*np.pi*freqs
    c_tot   = c
    R_u     = Ru
    kfp     = kp
    kbp     = kmp
    Kp      = kfp/kbp
    lp      = kfp+kbp
    kfs     = kf
    kbs     = kmf
    Ks      = kfs/kbs
    ls      = kfs+kbs
    cox     = c_tot /((1 + 1/Kp)*np.exp(-(n*F/(R*T))*(Eeq-EZero))+(1+Ks))        #Bewirkt, dass Immer die Gleichgewkonz genopmmen
    cred    = c_tot /((1 + 1/Kp) + (1+Ks)*np.exp((n*F/(R*T))*(Eeq-EZero)))       #Bewirkt, dass Immer die Gleichgewkonz genopmmen
    kf_el   = kzero*np.exp(alpha*n*F*(Eeq-EZero)/(R*T))
    kb_el   = kzero*np.exp(-(1-alpha)*n*F*(Eeq-EZero)/(R*T))
    print(kf_el, kb_el)
    print(cred, cox)
    ZFar    = (R*T/(A*(n*F)**2))*(1/(alpha*kf_el*cred+(1-alpha)*kb_el*cox)) #* (1  +kf_el/((1+Kp)*((1j*Freq+lp)*Dred)**0.5)   +kf_el*Kp/((1+Kp)*((1j*Freq)*Dred)**0.5)       +kb_el*Ks/((1+Ks)*((1j*Freq+ls)*Dox)**0.5)   +kb_el/((1+Ks)*((1j*Freq)*Dox)**0.5)      )
    print(ZFar)
    #----------------------------------------------------------------------------------------------
    Z  = R_u + 1/(1.0/ZFar + ((1j*Freq)**gamma)*Cap)
    return Z
def NodiffImpCalculator(freqs, Ru,gamma, Cap, alpha, kf_el, kb_el, ctot):
    print(kf_el, kb_el)
    Freq    = 2*np.pi*freqs
    R_u     = Ru
    coeff=-(ctot*A*F**2)/(R*T)
    ZFar    =1/(coeff*(-alpha*kf_el-((1-alpha)*kb_el))
    print(ZFar)
    #----------------------------------------------------------------------------------------------
    Z  = R_u + 1/((1.0/ZFar) + (((1j*Freq)**gamma)*Cap))
    return Z

def V_Finder(gamma, omega, Tau):
    Periods = 10
    PPP     = 2500
    def Fs(x):
        return 1/(x*(x**(-gamma) + Tau))
    t     = np.linspace(1e-20,(Periods/freq),PPP*Periods)
    ft_TA = Talbot(Fs)(t)
    ft_AN = np.cos(omega*t)
    V_t   = np.zeros(len(t))
    for i in range(len(V_t)-1):
        Integrand = ft_TA[i::-1]*ft_AN[:i+1:]
        V_t[i]    = (t[1]-t[0])*(0.5*Integrand[0] + np.sum(Integrand[1:-1:1]) + 0.5*Integrand[-1])
    V_t[1]   = V_t[2] - (-11*V_t[3] + 18*V_t[4] -9*V_t[5] + 2*V_t[6])/6.0
    V_t[0]   = 0
    V_t[-1]  = V_t[-2] + (-2*V_t[-6] + 9*V_t[-5] -18*V_t[-4] + 11*V_t[-3])/6.0
    V_Interpol = InterpolatedUnivariateSpline(t, V_t)
    return V_Interpol




#=================================================================================
#=================================================================================
#Provide Parameters for Simulation
#=================================================================================
#=================================================================================
n     =  1         ;    F    =  96485        ;    R     =  8.314  ;   T     =  296        ;    D     =  1e-6
A     =  0.07       ;    c    =  0.001        ;    nu    =  0.1    ;   alpha =  0.5        ;    kzero =  3e-3
E_i   = -0.05      ;    E_up =  0.0          ;    E_low = -0.0    ;   E_f   = -0.0        ;    E_0   =  0
R_ohm = 10        ;    K_dl =  0.0005      ;    gamma = 0.6     ;   kp    = 1000        ;    kmp   =  0.001
kf    = 0.001      ;    kmf  =  1000         ;    kmax  = 1e6     ;   D_f   = D           ;    D_b   =  D
#_________________________________________________________________________________
U_amp     = 0.005          # Excitation amplitude for EIS signal
Tau       = R_ohm*K_dl     # Time constant
dXi       = 0.005          # Parameter for numerical resolution BE CAREFUL HERE!!!!
#=================================================================================
#=================================================================================
#Define Frequencies and an empty array to strore Z upon calculation
#=================================================================================
#=================================================================================
freqs     = np.logspace(-3,6,600)
FT_Z      = np.zeros(len(freqs), dtype = 'complex')
#=================================================================================
#=================================================================================
#Begin the actual calculations
#=================================================================================
#=================================================================================

Z_classical = PlanarSemiinfImpCalculator(freqs, Ru = R_ohm, EZero = E_0, Eeq = E_i, Dred = D_f, Dox = D_b, Cap = K_dl, kp =kp, kmp=kmp, kf=kf, kmf=kmf, gamma = gamma)
theta_0=1

params=["k_0", "alpha", "gamma", "E_0", "Cap"]
param_ranges=dict(zip(params, [
                            [1e-3, 3e-3, 5e-3, 9e-3],
                            [0.4, 0.5, 0.6, 0.7],
                            [0.1, 0.3, 0.7, 0.9],
                            [-0.0025, 0, 0.0025],
                            [1e-6, 1e-4, 1e-3]
                            ]))
fig, ax=plt.subplots(2,3)
for i in range(0, len(params)):
    axis=ax[i//3, i%3]
    key=params[i]
    param_vals=dict(zip(params, [kzero, alpha, gamma, E_0, K_dl]))
    for j in range(0,len(param_ranges[key])):
        param_vals[key]=param_ranges[key][j]
        kf=param_vals["k_0"]*np.exp((1-param_vals["alpha"])*n*F*(E_i-param_vals["E_0"])/(R*T))
        kb=param_vals["k_0"]*np.exp(-param_vals["alpha"]*n*F*(E_i-param_vals["E_0"])/(R*T)) 
        Z_classical=NodiffImpCalculator(freqs, Ru=R_ohm,gamma=param_vals["gamma"], Cap=param_vals["Cap"], alpha=param_vals["alpha"], 
                                        kf_el=kf, 
                                        kb_el=kb,
                                        ctot=1)
        axis.plot(Z_classical.real, -Z_classical.imag, label="{1}={0}".format(param_ranges[key][j], key))
    axis.legend()
ax[1, 2].set_axis_off()
plt.show()
