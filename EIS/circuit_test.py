import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
import time
test_dict={"z1":{"p1":{"p1_1":"R18", "p1_2":"C13"}, "p2":{"p2_1":"R8", "p2_2":"Cl3"},
        "p3":{"p3_1":["R14", "R18"], "p3_2":{"p3_2_1":"R2", "p3_2_2":"Cdl1"}}},
        "z2":"R1", "z3":{"p1":"R3", "p2":{"p2_1":"R4", "p2_2":"Cdl2"}}}
#Randles={"z1":"R1","z2":{"p_1":"R2", "p_2":"C1"},  "z3":{"p_1":"R3", "p_2":"C2"}}#
randles={"z1":"R1", "z2":{"p_1":("Q1", "alpha1"), "p_2":["R2", "W1"]}, "z4":{"p_1":"R3", "p_2":"C2"} }
#debeye={"z1":{"p_1":"C1", "p_2":["R1", ("Q1", "alpha1")]}}
test=EIS(circuit=randles)
#test.write_model(circuit=Randles)
frequency_powers=np.arange(-1.5, 5, 0.01)
frequencies=[10**x for x in frequency_powers]
spectra=np.zeros((2, len(frequencies)), dtype="complex")
def double_randles(**kwargs):
    R1=kwargs["R1"]
    R2=kwargs["R2"]
    R3=kwargs["R3"]
    jfC1=1j*kwargs["omega"]*kwargs["C1"]
    jfC2=1j*kwargs["omega"]*kwargs["C2"]
    return R1+1/(1/(R2)+(jfC1))+1/(1/(R3)+(jfC2))
def double_randles_W_f(**kwargs):
    R1=kwargs["R1"]
    R2=kwargs["R2"]
    R3=kwargs["R3"]
    jfC1=1j*kwargs["omega"]*kwargs["C1"]
    jfC2=1j*kwargs["omega"]*kwargs["C2"]
    return R1+1/(1/(R2)+(jfC1))+1/(kwargs["Q1"]*np.power(1j*kwargs["omega"], kwargs["alpha1"]))+((1/kwargs["Q2"])*np.tanh((kwargs["delta1"]/np.sqrt(kwargs["D1"]))*np.sqrt(1j*params["omega"])))#+1/(1/(R3)+(jfC2))
def double_randles_W_i(**kwargs):
    R1=kwargs["R1"]
    R2=kwargs["R2"]
    R3=kwargs["R3"]
    jfC1=1j*kwargs["omega"]*kwargs["C1"]
    jfC2=1j*kwargs["omega"]*kwargs["C2"]
    return R1+1/(1/(R2)+(jfC1))+1/(kwargs["Q1"]*np.power(1j*kwargs["omega"], kwargs["alpha1"]))+(1/(kwargs["Q2"]*np.sqrt(1j*kwargs["omega"])))#+1/(1/(R3)+(jfC2))
def single_randles(**kwargs):
    R1=kwargs["R1"]
    R2=kwargs["R2"]
    R3=kwargs["R3"]
    jfC1=1j*kwargs["omega"]*kwargs["C1"]
    return 1/((1/(R3+R2))+jfC1)
def gamry_randles(**kwargs):
    R1=kwargs["R1"]
    R2=kwargs["R2"]
    jfC1=1j*kwargs["omega"]*kwargs["C1"]
    Q1=1/(np.sqrt(1j*kwargs["omega"]))
    denom=np.sqrt(kwargs["omega"])
    W1=(kwargs["W1"]/denom)-1j*(kwargs["W1"]/(denom))
    #W1=(1j*kwargs["W1"]/(np.sqrt(kwargs["omega"])))
    return R1+1/(jfC1+1/(R2+W1))#(1/(R2+W1)))
times=[0,0]
fig, ax=plt.subplots(2, 4)
for R3 in [10, 50, 100, 150, 200, 250]:
    for i in range(0, len(frequencies)):
        params={"R1":10, "R3":R3, "C2":1e-3,"R2":250, "Q1":2e-3, "omega":frequencies[i],  "C1":1e-6, "W1":40, "alpha1":1}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[0, -1].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=R3)
    ax[0,-1].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    plt.legend()
ax[0, -1].set_title("R3")
for C2 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
    for i in range(0, len(frequencies)):
        params={"R1":10, "R3":100, "C2":C2,"R2":250, "Q1":2e-3, "omega":frequencies[i],  "C1":1e-6, "W1":40, "alpha1":1}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[1,-2].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=C2)
    ax[1,-2].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    ax[1,-2].legend()
ax[1,-2].set_title("C2")
for Rct in [10, 50, 100, 150, 200, 250]:
    for i in range(0, len(frequencies)):
        params={"R1":10, "R3":100, "C2":1e-3,"R2":Rct, "Q1":2e-3, "omega":frequencies[i],  "C1":1e-6, "W1":40, "alpha1":1}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[0,0].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=Rct)
    ax[0,0].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    ax[0,0].legend()
ax[0,0].set_title("R2")
for Cdl in [1e-4, 5e-4, 1e-3, 5e-3, 1e-3]:
    for i in range(0, len(frequencies)):
        params={"R1":10, "R3":100, "C2":1e-3,"R2":250, "Q1":Cdl, "omega":frequencies[i],  "C1":1e-6, "W1":40, "alpha1":1}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[1,0].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=Cdl)
    ax[1,0].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    ax[1,0].legend()
ax[1,0].set_title("Q1")
for alpha in [0.1, 0.3, 0.5, 0.7, 1]:
    for i in range(0, len(frequencies)):
        params={"R1":10, "R3":100, "C2":1e-3,"R2":250, "Q1":2e-3, "omega":frequencies[i],  "C1":1e-6, "W1":40, "alpha1":alpha}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[0,1].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=alpha)
    ax[0,1].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    ax[0,1].legend()
ax[0, 1].set_title("alpha")
for w in [1, 10, 50, 100, 200]:
    for i in range(0, len(frequencies)):
        params={"R1":10, "R3":100, "C2":1e-3,"R2":250, "Q1":2e-3, "omega":frequencies[i],  "C1":1e-6, "W1":w, "alpha1":alpha}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[1,1].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=w)
    ax[1,1].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    ax[1,1].legend()
ax[1,1].set_title("W")
for rs in [1, 10, 20, 50, 100]:
    for i in range(0, len(frequencies)):
        params={"R1":rs, "R3":100, "C2":1e-3,"R2":250, "Q1":2e-3, "omega":frequencies[i],  "C1":1e-6, "W1":40, "alpha1":alpha}
        #spectra[0][i]=gamry_randles(**params)
        start=time.time()
        spectra[1][i]=test.simulate(**params)
        times[1]=time.time()-start
    print(times)
    #plt.plot(np.real(spectra[0,:]), -np.imag(spectra[0,:]), label="hard")
    ax[0,2].plot(np.real(spectra[1,:]), -np.imag(spectra[1,:]), label=rs)
    ax[0,2].scatter(np.real(spectra[1,:])[0::8], -np.imag(spectra[1,:])[0::8])
    ax[0, 2].legend()
ax[0,2].set_title("R1")
ax[1, -1].set_axis_off()
plt.show()
