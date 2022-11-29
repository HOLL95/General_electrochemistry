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
thesis_circuits=[{"z1":"R1"}, {"z1":"C1"}, {"z1":("Q1", "alpha1")}]
faradaic_circuit={"z1":"R1", "z2":{"p_1":"R2", "p_2":("Q1", "alpha1")}}
blank_circuit={"z1":"R1", "z2":"C1"}
#debeye={"z1":{"p_1":"C1", "p_2":["R1", ("Q1", "alpha1")]}}
params={"R2":1000, "R1":10,"C1":1e-5, "Q1":1e-2, "alpha1":0.5}
variables={"R1":{"vals":[5, 10, 20], "sym":"R", "unit":"$\\Omega$"}, "C1":{"vals":np.flip([1e-1, 2.5e-1, 3e-1]), "sym":"C", "unit":"F"},  "alpha1":{"vals":[0.9, 0.8, 0.7], "sym":"$\\alpha$", "unit":""}}
variable_key=list(variables.keys())
#test.write_model(circuit=Randles)

frequency_powers=np.arange(-1, 5, 0.1)
frequencies=[10**x for x in frequency_powers]
sim_class=EIS(circuit=blank_circuit)
spectra=np.zeros((2, len(frequencies)), dtype="complex")
spectra=sim_class.test_vals(params, frequencies)
sim_class.nyquist(spectra, scatter=2,)
sim_class.bode(spectra, frequencies)
plt.show()
