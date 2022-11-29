import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser, EIS_genetics
from circuit_drawer import circuit_artist
from Fit_explorer import explore_fit
from single_e_class_unified import single_electron
import math


k0=200
F=96485.3321
R=8.3145
T=298
circuit_1={"z0":"R0", "z1":{"p1":"C1", "p2":["R1","C2"]}}
circuit_2={"z0":"R0", "z1":{"p1":"C1", "p2":"R1"}}
R0=5
fig, axes=plt.subplots(1,3)
param_variations={"E_0":[-0.01, 0, 0.02],"k0":[100, 200, 300], "alpha":[0.4, 0.5, 0.6]}
labels={
        "E0_mean":{"name":"$E^0\\mu$", "unit":"mV"}, "E0_std":{"name":"$E^0\\sigma$", "unit":"mV"},
        "k0":{"name":"$k_0$", "unit":"$s^{-1}$"}, "alpha":{"name":"$\\alpha$", "unit":""},
        "E_0":{"name":"$E^0$", "unit":"mV"}, }
param_keys=list(param_variations.keys())
param_vals={key:param_variations[key][1] for key in param_keys}
orig_param_vals=copy.deepcopy(param_vals)
circuit_list=[circuit_1, circuit_2]
frequency_powers=np.arange(2.5, 5, 0.1)
frequencies=[10**x for x in frequency_powers]
for i in range(0, len(param_keys)):

    ax=axes[i]
    circuit=circuit_1
    current_key=param_keys[i]
    param_vals=orig_param_vals
    for j in range(0, len(param_variations[current_key])):
        param_vals[current_key]=param_variations[current_key][j]
        k0=param_vals["k0"]
        #dc_pot=0
        gamma=4e-10
        alpha=param_vals["alpha"]



        dc_pot=0
        area=5e-2
        FRT=F/(R*T)

        e0=param_vals["E_0"]
        ratio=np.exp(FRT*(e0-dc_pot))
        red=gamma/(ratio+1)
        ox=gamma-red
        #ox=1.3340954705306548e-10
        #red=ox
        #print(red, ox)
        Ra_coeff=(R*T)/((F**2)*area*k0)
        #print(Ra_coeff)
        nu_1_alpha=np.exp((1-alpha)*FRT*(dc_pot-e0))
        nu_alpha=np.exp((-alpha)*FRT*(dc_pot-e0))
        Ra=Ra_coeff*((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha))**-1
        #pred_theta_ox=Ra_coeff/(133*(alpha*nu_alpha+(1-alpha)*(nu_1_alpha)))
        #print(pred_theta_ox)
        sigma=k0*Ra*(nu_alpha+nu_1_alpha)
        #print(Ra, sigma)
        Ca=1/46794.65389622453
        Ca=1/sigma
        Cd=1e-6

        params={"R0":R0, "C1":1e-6, "R1":Ra,  "C2":Ca}
        test=EIS(circuit=circuit)
        spectra=test.test_vals(params, frequencies)


        test.nyquist(spectra, orthonormal=False, scatter=1,ax=ax,
                    label=labels[current_key]["name"]+"="+str(param_variations[current_key][j])+" "+labels[current_key]["unit"])
    ax.legend()
plt.show()


#explore_fit(circuit, params, frequencies=frequencies, )
for e0 in [0e-3, 10e-3, 20e-3, 30e-3]:


    ratio=np.exp(FRT*(e0-dc_pot))
    red=gamma/(ratio+1)
    ox=gamma-red
    Ra_coeff=(R*T)/((F**2)*area*k0)
    nu_1_alpha=np.exp((1-alpha)*FRT*(dc_pot-e0))
    nu_alpha=np.exp((-alpha)*FRT*(dc_pot-e0))
    Ra=Ra_coeff*((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha))
    sigma=k0*Ra*(nu_alpha+nu_1_alpha)
    Ca=1/sigma
    Cd=1e-6
    print(nu_1_alpha, nu_alpha)
    #print(red, ox)
    #print(((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha)), Ra_coeff)
    #print(Ca, Ra)

    frequencies=np.power(10, np.arange(0, 3, 0.1))
    phase=np.arctan(sigma/(frequencies*Ra))
    plt.plot(frequencies/k0, phase*(180/np.pi))
plt.show()





B=1+sigma*Cd
#print(red, ox)
#print(((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha)), Ra_coeff)
#print(Ca, Ra)
params={"R0":R0, "C1":Cd, "R1":Ra, "C2":Ca}

#phase=np.arctan(sigma/(frequencies*Ra))
#plt.plot(frequencies/k0, phase*(180/np.pi))
Ra=88
sigma=5.33*10**4
Delta=(1+sigma*Cd)**2+(np.square(frequencies)*sigma**2*Cd**2)
Z_r=R0+Ra/Delta
Z_im=((sigma/frequencies)+(frequencies*sigma*Cd)+ (sigma**2*Cd/frequencies))/Delta
plt.plot(Z_r, Z_im)
plt.show()
def z_func(R0, R1, C1, C2, frequencies):
    z_c1=1/(1j*frequencies*C1)
    z_c2_r1=R1+(1/(1j*frequencies*C2))
    z_3=1/((1/z_c1)+(1/z_c2_r1))
    return z_3
z_3=z_func(R0, 88, 1e-6, 1e-5, frequencies)
plt.plot(z_3.real, -z_3.imag)
plt.show()
