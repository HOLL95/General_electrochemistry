import numpy as np
import matplotlib.pyplot as plt
import os
import sys

dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
randles={"z1":"R0", "z2":{"p1":["R2", ("Q2", "alpha2")], "p2":("Q1", "alpha1")}, "z3":{"p1":"R1", "p2":"C2"}}
randles={"z1":"R0","z2":{"p1":"R1", "p2":"C2"}}
translator=EIS(circuit=randles)
frequency_powers=np.arange(2, 10, 0.01)
frequencies=[10**x for x in frequency_powers]
fig, ax=plt.subplots()
for R_val in [1, 1.5, 2]:
    true_params={ "Q1":1e-3, "alpha1":0.5, "R2":1, "Q2":1/100, "alpha2":0.5, "R0":10, "R1":R_val, "C2":1e-6}
    spectra=translator.test_vals(true_params,frequencies )
    translator.nyquist(spectra, ax=ax, label="$R_{ct}="+str(R_val)+" \\Omega$")
plt.legend()
plt.show()
