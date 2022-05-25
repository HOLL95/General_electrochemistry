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
from EIS_optimiser import EIS_optimiser, EIS_genetics
from circuit_drawer import circuit_artist
from time_domain_simulator_class import time_domain
frequency_powers=np.arange(1, 6, 0.1)
frequencies=[10**x for x in frequency_powers]

translated_circuit={"z1":{"p1":"R1", "p2":("Q1", "alpha1")}, "z2":"R0"}
sim_dict={"R1":1, "R0":1, "Q1":1e-3, "alpha1":0.5}
td=time_domain(translated_circuit, params=sim_dict)
c, t, p=td.simulate()
plt.plot(p, c)
plt.show()
