import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
proposed_circ_1={'z1': 'R0', 'z2': {'p1': 'C1', 'p2': ['R1', 'C2'], 'p3': ['R2', 'C3']}}
proposed_circ_2={'z1': 'R0', 'z2': {'p1': 'C1', 'p2': ['R1', 'C2']}}

params={"R0":10, "C1":1e-7, "R1":40000, "C2":1e-7}
param_names=list(params.keys())
test_class=EIS(circuit=proposed_circ_2, parameter_names=param_names)
freq_powers=np.linspace(-2, 6, 80)
frequencies=[10**x for x in freq_powers]
vals=test_class.simulate([params[x] for x in param_names], frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(vals, frequencies, ax=ax, twinx=twinx)

params={"R0":10, "C1":1e-7, "R1":40000/0.5, "C2":1e-7*0.5, "R2":40000/0.5, "C3":1e-7*0.5, }
param_names=list(params.keys())
test_class=EIS(circuit=proposed_circ_1, parameter_names=param_names)
vals=test_class.simulate([params[x] for x in param_names], frequencies)
EIS().bode(vals, frequencies, ax=ax, twinx=twinx)


plt.show()