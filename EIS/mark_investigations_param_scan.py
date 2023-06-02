import os
import sys
import copy
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
from pandas import read_csv

files=np.load("BV_sim.npy")

freq=files[:,0]
phase=files[:,1]
magnitude=files[:,2]
fit_data=np.column_stack((phase, np.log10(magnitude)))



mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
params={"R2":1000, "R1":1500,"C1":1e-3,  "C2":50e-3, "R3":500, "Q1":20e-3, "alpha1":0.6}
values={'R0': 104.82889735474738, 'R1': 1525.2234882630044, 
        'C1': 0.00011241593575305592, 'C2': 6.770333588293525e-05, 
        'R3': 3350.5992740459615, 'Q1': 2.1311350351085948e-05, 
        'alpha1': 0.9710356115630412} 
EIS_test=EIS(circuit=mark_circuit)
test=EIS_test.test_vals(values, freq)
fig, ax=plt.subplots(1,1)
twinx=ax.twinx()
EIS_test.bode(test, freq, ax=ax, twinx=twinx)
EIS_test.bode(fit_data, freq, ax=ax, twinx=twinx,data_type="phase_mag")
plt.show()

params=list(values.keys())
multiply_vals=[0.5, 0.75, 1.25, 1.5]
param_scan_dict={key:np.multiply(multiply_vals, values[key]) for key in params}
param_scan_dict["alpha1"]=[0.5, 0.7, 0.9, 1.0]
fig, ax=plt.subplots(2, 4)
twinx=[ax[i//4, i%4].twinx() for i in range(0, len(params))]
EIS_test=EIS(circuit=mark_circuit)
for i in range(0, len(params)):
    axis=ax[i//4, i%4]
    axis.set_title(params[i])
    #twixis=twinx[i]
    copy_dict=copy.deepcopy(values)
    key=params[i]
    #EIS_test.bode(fit_data, freq, data_type="phase_mag", ax=ax[i], twinx=twinx[i])

    for j in range(0, len(multiply_vals)):
        copy_dict[key]=param_scan_dict[key][j]
        test=EIS_test.test_vals(copy_dict, freq)
        EIS_test.bode(test, freq, ax=axis, twinx=twinx[i], compact_labels=True)
ax[-1, -1].set_axis_off()
plt.show()
plt.show()



for methods in ["AIC"]:
    results_list=[]
    results_circuit=[]
    true_score=[]
    for i in range(0, 5):

        #print(sigma, optim.get_std(noisy_data, sim), optim.optimise(noisy_data, sigma,"minimisation", [true_params[x] for x in param_names]))
        #gene_test=EIS_genetics(generation_size=8, generation_test=True, selection="bayes_factors")#, generation_test_save="Generation_images/Bayes_factor_randles_test/round_1/")
        #gene_test.evolve(freq, noisy_data)
        gene_test=EIS_genetics(generation_size=12, generation_test=True, individual_test=True,
                                selection=methods, initial_tree_size=1, 
                                best_record=True, num_top_circuits=6, num_optim_runs=1, data_representation="bode", construction_elements=["R", "C", "CPE"])
        value,return_params, sim_data=gene_test.assess_score(mark_circuit, list(params.keys()), freq, fit_data, score_func=methods, data_representation="bode")
        print(dict(zip(params.keys(), return_params)), value, "HELLO")
        translator.bode(fit_data, freq, data_type="phase_mag", label="data", ax=ax, twinx=twinx)
        translator.bode(EIS_test.test_vals(return_params, freq), freq, data_type="phase_mag", label="sim",ax=ax, twinx=twinx)
        plt.show()

        
        #plt.plot(range(0, num_generations),scaled_score)
#plt.show()
