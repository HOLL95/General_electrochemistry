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
from pandas import read_csv

files=np.load("BV_sim.npy")

freq=files[:,0]
phase=files[:,1]
magnitude=files[:,2]
fit_data=np.column_stack((phase, np.log10(magnitude)))

translator=EIS()
translator.bode(fit_data, freq, data_type="phase_mag")
plt.show()


mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R2", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
params={"R2":1000, "R1":1500,"C1":1e-3,  "C2":50e-3, "R3":500, "Q1":20e-3, "alpha1":0.6}
EIS_test=EIS(circuit=mark_circuit)
test=EIS_test.test_vals(params, freq)
EIS_test.bode(test, freq)
plt.show()
fig,ax=plt.subplots(1,1)
twinx=ax.twinx()


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
        print(return_params, value, "HELLO")
        translator.bode(fit_data, freq, data_type="phase_mag", label="data", ax=ax, twinx=twinx)
        translator.bode(EIS_test.test_vals(return_params, freq), freq, data_type="phase_mag", label="sim",ax=ax, twinx=twinx)
        plt.show()

        
        #plt.plot(range(0, num_generations),scaled_score)
#plt.show()
