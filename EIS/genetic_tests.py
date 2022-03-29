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
randles={"z1":"R0", "z2":{"p1":("Q1", "alpha1"), "p2":["R2", "W1"]}}
translator=EIS()
circuit_artist(randles)
ax=plt.gca()
ax.set_axis_off()
plt.show()
frequency_powers=np.arange(1, 6, 0.1)
frequencies=[10**x for x in frequency_powers]
bounds={
"R1":[0, 1000],
"Q1":[0, 1e-2],
"alpha1":[0.1, 0.9],
"R2":[0, 1000],
"W1":[1, 200]
}
true_params={"R1":100, "Q1":1e-3, "alpha1":0.5, "R2":10, "W1":40, "R0":10}
param_names=["R1", "Q1", "alpha1", "R2", "W1", "R0"]
optim=EIS_optimiser(circuit=randles, parameter_bounds=bounds, frequency_range=frequencies, param_names=param_names, test=False)
sim=optim.test_vals(true_params, frequencies)
sigma=0.001*np.sum(sim)/(2*len(sim))

#plt.plot(sim[:,0], -sim[:,1])
#plt.plot(noisy_data[:,0], -noisy_data[:,1])
#plt.show()
for methods in ["AIC", "BIC"]:
    results_list=[]
    results_circuit=[]
    true_score=[]
    for i in range(0, 5):
        noisy_data=np.column_stack((optim.add_noise(sim[:,0], sigma), optim.add_noise(sim[:,1], sigma)))
        #print(sigma, optim.get_std(noisy_data, sim), optim.optimise(noisy_data, sigma,"minimisation", [true_params[x] for x in param_names]))
        #gene_test=EIS_genetics(generation_size=8, generation_test=True, selection="bayes_factors")#, generation_test_save="Generation_images/Bayes_factor_randles_test/round_1/")
        #gene_test.evolve(frequencies, noisy_data)
        gene_test=EIS_genetics(generation_size=12, generation_test=False, selection=methods, initial_tree_size=1, best_record=True, num_top_circuits=6)
        value,_=gene_test.assess_score(randles, param_names, frequencies, noisy_data, score_func=methods)
        print(value, "value")
        true_score.append(value)
        num_generations=5
        gene_test.evolve(frequencies, noisy_data, num_generations=num_generations)
        np.save("Best_candidates/round_2/{0}/Best_candidates_dict_{1}_12_gen.npy".format(methods, i+1), gene_test.best_candidates)
        print(gene_test.best_array, value)
        scaled_score=np.divide(value, gene_test.best_array)
        results_list.append(gene_test.best_array)
        results_circuit.append(gene_test.best_circuits)
        #plt.plot(range(0, num_generations),scaled_score)
    np.save("{0}_best_scores.npy".format(methods), {"scores":results_list, "circuits":results_circuit, "true_score":true_score})
#plt.show()
