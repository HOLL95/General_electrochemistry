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
print(files)
freq=files[:,0]
phase=files[:,1]
magnitude=files[:,2]
fit_data=np.column_stack((phase, np.log10(magnitude)))

translator=EIS()
translator.bode(fit_data, freq, data_type="phase_mag")
plt.show()
EIS()





for methods in ["AIC"]:
    results_list=[]
    results_circuit=[]
    true_score=[]
    for i in range(0, 5):

        #print(sigma, optim.get_std(noisy_data, sim), optim.optimise(noisy_data, sigma,"minimisation", [true_params[x] for x in param_names]))
        #gene_test=EIS_genetics(generation_size=8, generation_test=True, selection="bayes_factors")#, generation_test_save="Generation_images/Bayes_factor_randles_test/round_1/")
        #gene_test.evolve(freq, noisy_data)
        gene_test=EIS_genetics(generation_size=12, generation_test=False, individual_test=False,
                                selection=methods, initial_tree_size=1, 
                                best_record=True, num_top_circuits=6, num_optim_runs=5, data_representation="bode", construction_elements=["R", "C"])
        
       
        num_generations=5
        gene_test.evolve(freq, fit_data, num_generations=num_generations)
        np.save("Best_candidates_BV_sim_{0}".format(i), gene_test.best_candidates)
        scaled_score=np.divide(value, gene_test.best_array)
        results_list.append(gene_test.best_array)
        results_circuit.append(gene_test.best_circuits)
        #plt.plot(range(0, num_generations),scaled_score)
#plt.show()
