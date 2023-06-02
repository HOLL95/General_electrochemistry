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
from circuit_drawer import circuit_artist
file_name="Best_candidates_BV_sim_1.npy"
data_files=np.load("BV_sim.npy")

freq=data_files[:,0]
results=np.load(file_name, allow_pickle=True).item()
print(results.keys())

plotter=EIS()
fig, ax=plt.subplots(2, len(results["models"]))
for i in range(0, len(results["models"])):
    translated_circuit=plotter.translate_tree(results["models"][i])
    circuit_artist(translated_circuit, ax=ax[0, i])
    sim_class=EIS(circuit=translated_circuit)
    print(type(results["parameters"][i]))
    simulation=sim_class.test_vals(results["parameters"][i], freq)
    twinx=ax[1, i].twinx()
    plotter.bode(simulation, freq, ax=ax[1, i], twinx=twinx, label="sim", compact_labels=True)
    plotter.bode(results["data"], freq, ax=ax[1, i],data_type="phase_mag", twinx=twinx, label="data", compact_labels=True)

plt.show()

