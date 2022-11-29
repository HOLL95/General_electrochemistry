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
data_loc="Experimental_data/"
files=os.listdir(data_loc)
experiments=["DBCO", "Bare"]
for exp in experiments:
    for file in files:
        if exp in file:
            data=read_csv(data_loc+file, sep=",", encoding="utf-16", engine="python", skiprows=5, skipfooter=1)
    print(data.columns)
    numpy_data=data.to_numpy(copy=True, dtype='float')
    print(list(numpy_data[:, 0]))
    frequencies=np.flip(numpy_data[:, 0])
    sim=np.column_stack((np.flip(numpy_data[:, 4]),-1*np.flip(numpy_data[:, 5])))
    print(sim)
    plt.scatter(sim[:,0], sim[:,1])
    randles={"z1":"R0", "z2":{"p1":[("Q1", "alpha1")], "p2":["R2", "W1"]}}
    translator=EIS()
    frequency_powers=np.arange(1, 6, 0.1)
    plt.show()
    bounds={
    "R1":[0, 1000],
    "Q1":[0, 1e-2],
    "alpha1":[0.1, 0.9],
    "R2":[0, 1000],
    "W1":[1, 200]
    }
    true_params={"R1":100, "Q1":1e-3, "alpha1":0.5, "R2":10, "W1":40, "R0":10}
    param_names=["R1", "Q1", "alpha1", "R2", "W1", "R0"]
    optim=EIS_optimiser(circuit=randles, parameter_bounds=bounds, frequency_range=frequencies, param_names=param_names, test=False, generation_test=True)

    sigma=0.001*np.sum(sim)/(2*len(sim))

    #plt.plot(sim[:,0], -sim[:,1])
    #plt.plot(noisy_data[:,0], -noisy_data[:,1])
    #plt.show()
    for methods in ["AIC"]:
        results_list=[]
        results_circuit=[]
        true_score=[]
        for i in range(0, 5):
            noisy_data=sim
            #print(sigma, optim.get_std(noisy_data, sim), optim.optimise(noisy_data, sigma,"minimisation", [true_params[x] for x in param_names]))
            #gene_test=EIS_genetics(generation_size=8, generation_test=True, selection="bayes_factors")#, generation_test_save="Generation_images/Bayes_factor_randles_test/round_1/")
            #gene_test.evolve(frequencies, noisy_data)
            value=-1e15
            gene_test=EIS_genetics(generation_size=12,  selection=methods, initial_tree_size=1, best_record=True, num_top_circuits=6, generation_test=True)

            #value,params, data=gene_test.assess_score(randles, param_names, frequencies, noisy_data, score_func=methods)
            print(value)
            fig, ax=plt.subplots()
            #print(params)
            #gene_test.plot_data(data, ax)
            #gene_test.plot_data(sim, ax)
            #plt.show()
            params=[2.41386492e+04, 3.70130541e-06, 9.13516388e-01, 1.25974558e+04,1.94427239e+02, 1.79239225e-14]
            test_sim=gene_test.dict_simulation(randles, frequencies, noisy_data, params, param_names)
            gene_test.plot_data(noisy_data,ax, scatter=True)
            gene_test.plot_data(test_sim, ax,colour="orange")
            ax.set_xlabel("Zr")
            ax.set_ylabel("-Zi")
            plt.show()
