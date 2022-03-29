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
import copy
test_dict={"z1":{"p1":{"p1_1":"R18", "p1_2":"C13"}, "p2":{"p2_1":"R8", "p2_2":"Cl3"},
        "p3":{"p3_1":["R14", "R18"], "p3_2":{"p3_2_1":"R2", "p3_2_2":"Cdl1"}}},
        "z2":"R1", "z3":{"p1":"R3", "p2":{"p2_1":"R4", "p2_2":"Cdl2"}}}
#Randles={"z1":"R1","z2":{"p_1":"R2", "p_2":"C1"},  "z3":{"p_1":"R3", "p_2":"C2"}}#
randles={"z1":"R1", "z2":{"p_1":("Q1", "alpha1"), "p_2":["R2", "W1"]}}#, "z4":{"p_1":"R3", "p_2":"C2"} }
#debeye={"z1":{"p_1":"C1", "p_2":["R1", ("Q1", "alpha1")]}}

test=EIS(circuit=randles)
#test.write_model(circuit=Randles)
frequency_powers=np.arange(-1.5, 5, 0.1)
frequencies=[10**x for x in frequency_powers]
spectra=np.zeros((2, len(frequencies)), dtype="complex")
times=[0,0]
fig, ax=plt.subplots(2, 4)
plt.rcParams.update({'font.size': 16})
param_dict={"R1": [10, 50, 100, 150, 200, 250], "R2":[10, 50, 100, 150, 200, 250], "R3":[10, 50, 100, 150, 200, 250],
"C2":[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "W1":[1, 10, 50, 100, 200], "Q1":[1e-4, 5e-4, 1e-3, 5e-3, 1e-2], "alpha1":[0.1, 0.3, 0.5, 0.7, 1]}
params={"R1":10, "R3":100, "C2":0.01,"R2":250, "Q1":2e-3, "omega":0,  "C1":1e-6, "W1":40, "alpha1":1}
spectra=np.zeros(len(frequencies), dtype="complex")
param_keys=list(param_dict.keys())
for j in range(0, len(param_keys)):
    for q in range(0, len(param_dict[param_keys[j]]), 2):
        sens_params=copy.deepcopy(params)
        sens_params[param_keys[j]]=param_dict[param_keys[j]][q]
        start=time.time()
        for i in range(0, len(frequencies)):
            sens_params["omega"]=frequencies[i]
            spectra[i]=test.freq_simulate(**sens_params)
        print(time.time()-start)
        idx1=j//4
        idx2=j%4
        axes=ax[idx1, idx2]
        axes.plot(np.real(spectra), -np.imag(spectra), label=param_dict[param_keys[j]][q])
        axes.scatter(np.real(spectra)[0::8], -np.imag(spectra)[0::8])
    axes.legend()
    axes.set_title(param_keys[j])
    axes.set_xlabel("$Z_r (\\omega$)")
    axes.set_ylabel("$-Z_i (\\omega)$")
ax[1, -1].set_axis_off()
#plt.tight_layout()
fig.set_size_inches(7, 4.5)
plt.show()
