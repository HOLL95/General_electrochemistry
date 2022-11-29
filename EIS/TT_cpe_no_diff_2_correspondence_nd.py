# -*- coding: utf-8 -*-
"""
@author: Tim Tichter
"""


import numpy as np
import matplotlib.pyplot as plt
from math import floor
from cmath import *
from scipy.interpolate import InterpolatedUnivariateSpline
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)
from scipy.optimize import fsolve
import scipy
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
import math
from collections import deque
from params_class import params
from single_e_class_unified import single_electron
import isolver_martin_brent
from convolutive_modelling_class import conv_model

#=================================================================================
#=================================================================================
#Define some plotting specific Parameters - make plots look nice :)
#=================================================================================
#=================================================================================

font = {'family': 'Times New Roman', 'color':  'black','weight': 'normal','size': 15,}
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.sans-serif'] = ['Times new Roman']



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
    ZFar    = (R*T/(A*(n*F)**2))*(1/(alpha*kf_el*cred+(1-alpha)*kb_el*cox)) * (1  +kf_el/((1+Kp)*((1j*Freq+lp)*Dred)**0.5)   +kf_el*Kp/((1+Kp)*((1j*Freq)*Dred)**0.5)       +kb_el*Ks/((1+Ks)*((1j*Freq+ls)*Dox)**0.5)   +kb_el/((1+Ks)*((1j*Freq)*Dox)**0.5)      )
    #----------------------------------------------------------------------------------------------
    Z  = R_u + 1/(1.0/ZFar + ((1j*Freq)**gamma)*Cap)
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
A     =  0.1       ;    c    =  0.001        ;    nu    =  0.1    ;   alpha =  0.5        ;    kzero =  0.00001
E_i   = -0.05      ;    E_up =  0.0          ;    E_low = -0.0    ;   E_f   = -0.0        ;    E_0   =  0
R_ohm = 1.0       ;    K_dl =  0.0005    ;    gamma = 0.6     ;   kp    = 1000        ;    kmp   =  0.001
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
freqs     = np.logspace(-1,5,60)
FT_Z      = np.zeros(len(freqs), dtype = 'complex')
#=================================================================================
#=================================================================================
#Begin the actual calculations
#=================================================================================
#=================================================================================
plt.figure(figsize=(11,5), dpi = 100)
plot1 = plt.subplot(121)
plot2 = plt.subplot(122)


for kk in range(len(freqs)):
    param_list={
        "E_0":0.0,
        'E_start':  -0.05-U_amp, #(starting dc voltage - V)
        'E_reverse':0.0,
        'omega':10, #8.88480830076,  #    (frequency Hz)
        "original_omega":10,
        'd_E': U_amp,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru':R_ohm,  #     (uncompensated resistance ohms)
        'Cdl': K_dl, #(capacitance parameters)
        'CdlE1': 0,#0.000653657774506,
        'CdlE2': 0,#0.000245772700637,
        "CdlE3":0,
        'gamma': 1e-11,
        "psi":0.5,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': kzero    , #(reaction rate s-1)
        'alpha': 0.5,
        "cap_phase":0,
        'sampling_freq' : (1.0/400),
        'phase' :0,
        "num_peaks":5
    }
    import copy
    orig_param_list=copy.deepcopy(param_list)
    sim_options={
        "method":"sinusoidal",
        "experimental_fitting":False,
        "likelihood":"timeseries"
    }



    param_list["original_omega"]=freqs[kk]
    param_list["omega"]=freqs[kk]
    sim_options["no_transient"]=2/param_list["omega"]
    eis_test=single_electron(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
    current=eis_test.i_nondim(eis_test.test_vals([],"timeseries"))
    #omega=eis_test.nd_param.nd_param_dict["nd_omega"]
    #kzero=eis_test.nd_param.nd_param_dict["k_0"]
    #R_ohm=eis_test.nd_param.nd_param_dict["Ru"]
    #E_0=eis_test.nd_param.nd_param_dict["E_0"]
    #E_i=eis_test.nd_param.nd_param_dict["E_start"]
    #U_amp=eis_test.nd_param.nd_param_dict["d_E"]
    #K_dl=eis_test.nd_param.nd_param_dict["Cdl"]
    test_class=conv_model(None, dim_parameter_dictionary=param_list, simulation_options=sim_options)
    freq      = freqs[kk]
    omega     = 2*np.pi*freq
    Periods   = 5
    PPP       = 500
    t         = np.linspace(1e-20,(Periods/freq),PPP*Periods)
    dt        = t[1]-t[0]
    E         = E_i + U_amp*np.sin(omega*t)
    Kn        = np.exp(n*F*(E-E_0)/(R*T))
    timescale = np.logspace(-20,6,10000)
    p         = kp + kmp
    Kp        = kp/kmp
    Kf        = kf/kmf
    f         = kf + kmf
    Preced    = np.exp(-p*t)
    Follow    = np.exp(-f*t)
    Xi_in     = n*F*E_i/(R*T)
    cR_in     = c/(1 + 1/Kp + np.exp(Xi_in)*(1+Kf))
    cO_in     = cR_in*np.exp(Xi_in)
    print("cR_in = ", cR_in, "cO_in = ", cO_in, "Ein=")
    #=================================================================================
    #=================================================================================
    #Define the lineary capacitive CPE-part to zero here since no ramp is driven (unlike ACCV)
    #=================================================================================
    Pure_Idl_t  = np.zeros(len(t))
    #=================================================================================
    #=================================================================================
    def W_Lap(s):
        return 1/(s*(1 + Tau*s**(gamma)))
    W_timescale     = Talbot(W_Lap)(timescale)
    W_interpol      = InterpolatedUnivariateSpline(timescale, W_timescale, k = 3)
    W0              = W_timescale[0]
    #=================================================================================
    # Now, go for the oscillationg part and choose
    #=================================================================================
    #------------------------------------
    # the analytical solution - no CPE
    #------------------------------------
    #V_t_ana      =  (np.cos(omega*t) + omega*Tau*np.sin(omega*t) - np.exp(-t/Tau))/(1+omega**2 *Tau**2)
    #V_t_num=V_t_ana
    #------------------------------------
    # or the numerical solution - for CPE
    #------------------------------------
    V_t_num = V_Finder(gamma, omega, Tau)(t)
    #=================================================================================
    #=================================================================================
    M_t   = 2*(t/np.pi)**0.5
    M_R   = 2*(t/np.pi)**0.5
    M_O   = 2*(t/np.pi)**0.5
    W_t   = W_interpol(t)
    #=================================================================================
    #=================================================================================
    #Solve Convolution Integrals - this is the heart of the computations
    #=================================================================================
    #=================================================================================
    I_f    = np.zeros(len(E))
    I_ges  = np.zeros(len(E))
    I_dl   = np.zeros(len(E))
    Iconvm = 0
    theta_0=1
    theta=theta_0

    for m in range(len(E)-1):
        E_t=E[m]
        if m > 0:
            if m > 1:
                Iconvm    = np.sum(W_t[0:m-2:]*(I_f[m-1:1:-1] - I_f[m-2:0:-1]))
            if m == 1:
                def Rekur(x):
                    kf=kzero*np.exp((1-alpha)*n*F*(E_t-E_0-R_ohm*x))
                    kb=kzero*np.exp(-alpha*n*F*(E_t-E_0-R_ohm*x))
                    theta_top=kf+np.exp(-t[m]*(kf+kb))*(theta_0*kf+theta_0*kb-kf)
                    theta=theta_top/(kf+kb)
                    return (  (  ( kf*(1-theta) -kb*(theta)  )
                                - I_f[m-1])*W_t[1] + Pure_Idl_t[m] + omega*U_amp*K_dl*V_t_num[m] + Iconvm - x  )
            I_ges[m]  = fsolve(Rekur, I_ges[m-1])
            #theta=(theta+dt*kzero*np.exp((1-alpha)*(E_t-E_0-R_ohm*I_ges[m])))/(1+dt*kzero*np.exp(1-alpha*(E_t-E_0-R_ohm*I_ges[m]))+dt*kzero*np.exp(-alpha*(E_t-E_0-R_ohm*I_ges[m])))
            I_f[m]    = (I_ges[m] - Pure_Idl_t[m] - omega*U_amp*K_dl*V_t_num[m] - Iconvm)/W_t[1] + I_f[m-1]
            I_dl[m]   = I_ges[m] - I_f[m]
    #=================================================================================
    #=================================================================================
    #Fouriertransform the current and the potential to calculate the impedance
    # Maybe a dircet DNFT will be better than a FFT, though FFT does the job as well.
    #=================================================================================
    #=================================================================================
    FT_E            = np.fft.fft(E[PPP*2::])
    FT_I            = np.fft.fft(I_ges[PPP*2::])
    Z_of_omega      = FT_E/FT_I
    fft_freq=np.fft.fftfreq(len(I_ges[PPP*2::]), t[1]-t[0])
    #plt.plot(fft_freq, FT_I)
    #plt.axvline(freq)
    #plt.axvline(fft_freq[3], color="red", linestyle="--")
    #plt.show()
    ft_idx=np.abs(np.subtract(fft_freq, freq))
    peak_loc=np.where(ft_idx==min(ft_idx))[0][0]

    FT_Z[kk]        = Z_of_omega[peak_loc]

    conv_current=test_class.simulate_current(CPE=False)
    potential=test_class.e_nondim(test_class.define_voltages())
    times=eis_test.time_vec
    fft_freq_2=np.fft.fftfreq(len(conv_current), times[1]-times[0])
    plt.plot(t, I_ges)
    #plt.plot(times, conv_current)
    plt.show()
    #plt.subplot(1,2,1)
    #plt.plot(t[PPP*2:], E[PPP*2::])
    #plt.plot(times, potential)
    #plt.subplot(1,2,2)
    #plt.plot(t[PPP*2:],  I_ges[PPP*2::])
    #plt.plot(times, conv_current)
    #plt.show()
        #plt.plot(fft_freq, i_fft)
        #plt.axvline(fft_freq[peak_loc])
        #plt.show()

    #plt.plot(I_f)
    #plt.show()
    #=================================================================================
    #=================================================================================
    #Plot everything
    #=================================================================================
    #=================================================================================
    print("ITERATION", kk+1)
    if kk == 25:
        print("Frequency of Plot = ", freqs[kk])
        #plot1.plot(PPP*2*t[0:-1], PPP*2*I_ges[0:-1], color = 'black', linewidth = 1, linestyle = '-' , label = '$I_{\mathrm{T}}$')
        plot1.plot(PPP*2*t[0:-1], PPP*2*I_f[0:-1], color = 'blue', linewidth = 1, linestyle = '-' , label = '$I_{\mathrm{F}}$')
        plot1.plot(PPP*2*t[0:-1], PPP*2*I_dl[0:-1], color = 'red', linewidth = 1, linestyle = '-' , label = '$I_{\mathrm{DL}}$')
        plot1.set_ylabel('Currents in mA', fontsize = 20)
        plot1.set_xlabel('$10^{3}\,t\,/\,\mathrm{s}$', fontsize = 20)
        plot1.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 20)
        #plot1.set_ylim(-0.58, 0.79)
        plot1.annotate("C)", xy = (0.05,0.9), xycoords = 'axes fraction', fontsize = 25)
        ax2 = plot1.twinx()
        #ax2.plot(PPP*2*t[0:-1], PPP*2*(E[0:-1]-E_i), color = 'magenta', linewidth = 1, linestyle = '-' , label = '$U(t)$')
        ax2.set_ylabel(r'$U(t)$ / mV', fontsize = 20)
        ax2.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 20)
        ax2.set_ylim(-5.5, 8.9)
        plot1.legend(frameon = False, ncol = 2, fontsize = 15)
        ax2.legend(frameon = False, fontsize = 15, loc='lower right', bbox_to_anchor=(1.027, 0.77))


Z_classical = PlanarSemiinfImpCalculator(freqs, Ru = R_ohm, EZero = E_0, Eeq = E_i, Dred = D_f, Dox = D_b, Cap = K_dl, kp =kp, kmp=kmp, kf=kf, kmf=kmf, gamma = gamma)
plot2.plot(Z_classical.real,  -Z_classical.imag, marker = '.', label = 'EIS-normal', color ='black', linestyle = '')
plot2.plot(FT_Z.real, -FT_Z.imag, color ='black', linewidth=1, linestyle ='-')
plot2.set_xlabel(r'$\mathfrak{Re}(Z(\omega))$', fontsize = 20)
plot2.set_ylabel(r'$-\mathfrak{IM}(Z(\omega))$', fontsize = 20)
plot2.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 20)
plot2.annotate("D)", xy = (0.05,0.9), xycoords = 'axes fraction', fontsize = 25)
plot2.axvline(Z_classical[25].real, color = 'black', linewidth = 0.5, linestyle = ':')
plot2.axhline(-Z_classical[25].imag, color = 'black', linewidth = 0.5, linestyle = ':')


plt.tight_layout()
plt.savefig("Impedance_from_TimeDat_Num_V_Dagger.png", dpi = 100)
plt.show()
