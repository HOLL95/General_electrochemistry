import os
import sys
import math
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.image as mpimg
from scipy import special
from circuit_drawer import circuit_artist
thesis_circuits=[{"z1":"R1"}, {"z1":"C1"}, {"z1":("Q1", "alpha1")}]
faradaic_circuit={"z1":"R1", "z2":{"p1":"C1", "p2":["R2", ("Q1", "alpha1")]}, "z3":{"p1":"R3", "p2":"C2"}}
blank_circuit={"z1":{"p1":"R1", "p2":"C1"}, "z2":"R0", "z3":{"p1":["R3",("Q1", "alpha1") ], "p2":"C2"},}
bc_1={"z1":"R1", "z2":"R0", "z3":{"p1":["R3",("Q1", "alpha1") ], "p2":"C2"},}
bc_2={"z1":"C1", "z2":"R0", "z3":{"p1":["R3",("Q1", "alpha1") ], "p2":"C2"},}
bc_3={"z1":{"p1":"R1", "p2":"C1"},  "z3":{"p1":["R3",("Q1", "alpha1") ], "p2":"C2"},}
bc_4={"z1":{"p1":"R1", "p2":"C1"}, "z2":"R0", "z3":{"p1":("Q1", "alpha1"), "p2":"C2"},}
bc_5={"z1":{"p1":"R1", "p2":"C1"}, "z2":"R0", "z3":{"p1":"R3" , "p2":"C2"},}
bc_6={"z1":{"p1":"R1", "p2":"C1"}, "z2":"R0", "z3":["R3",("Q1", "alpha1") ],}
full_circ=[bc_1, bc_2, bc_3, bc_4, bc_5, bc_6, blank_circuit]
fig,ax=plt.subplots(4, 4)
#debeye={"z1":{"p1":"C1", "p2":["R1", ("Q1", "alpha1")]}}
params={"R2":1000, "R1":1500,"C1":1e-3, "Q1":1e-3, "alpha1":0.4, "C2":50e-3, "R3":500, "C3":20e-3}
params=k0_1={'R0': 103.42115051106593, 'R1': 10314.65622036976, 'C1': 0.0003923123750362509, 'C2': 4.8714659815844195e-05, 'R3': 702860.1095345016, 'Q1': 4.0932950630502546e-05, 'alpha1': 0.9859474865077266}
#variables={"R1":{"vals":[5, 10, 20], "sym":"R", "unit":"$\\Omega$"}, "C1":{"vals":np.flip([1e-1, 2.5e-1, 3e-1]), "sym":"C", "unit":"F"},  "alpha1":{"vals":[0.9, 0.8, 0.7], "sym":"$\\alpha$", "unit":""}}
#variable_key=list(variables.keys())
#test.write_model(circuit=Randles)

for i in range(0, len(full_circ)):
#params["R2"]=R
    frequency_powers=np.arange(-3, 9, 0.1)
    frequencies=[10**x for x in frequency_powers]
    sim_class=EIS(circuit=full_circ[i])
    spectra=np.zeros((2, len(frequencies)), dtype="complex")
    spectra=sim_class.test_vals(params, frequencies)
    axis=ax[2*(i//4)+1, i%4]
   
    sim_class.bode(spectra, frequencies, ax=axis, compact_labels=True)
    circ_ax=ax[2*(i//4), i%4]
    print(i)
    circuit_artist(full_circ[i], ax=circ_ax)
    circ_ax.set_axis_off()
ax[-1,-1].set_axis_off()
ax[2, -1].set_axis_off()
plt.show()
