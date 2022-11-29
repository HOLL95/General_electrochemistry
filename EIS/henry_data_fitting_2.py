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
data_loc="Experimental_data/5_7_22/"
files=os.listdir(data_loc)
exp_type="EIS"
plot_1="blank"
plot_2="t_eq"
plot_3="0.05"
get_color=plt.rcParams['axes.prop_cycle'].by_key()['color']
labels=["WT", "blank"]
get_color[2]="red"
c_idx=0

xaxis=["Potential(V)", "$Z_{re}$", "$log_{10}$(Freq)", "Potential(V)"]
yaxis=["Current($\\mu A$)", "$Z_{im}$", "$Z_{mag}$", "Current($\\mu A$)"]
files=[
"DCV_WT_pre_EIS.csv",
"DCV_blank_post_EIS.csv",
"EIS_blank_0.005V.csv",
"DCV_blank_pre_EIS.csv",
"EIS_WT_0.005V.csv",
"EIS_WT_0.2V.csv",
"EIS_blank_0.005V_wide_window.csv",
"EIS_WT_-0.3V.csv",
]
#fig, ax=plt.subplots(1,1)
file_nums=[6, 4, 5, 7]
folders=["Blank", "5mV", "200mV", "minus_300mV"]
for z in range(0, len(file_nums)):
    desired_files=[files[x] for x in file_nums]
    file=files[file_nums[z]]
    print(file)
    data=read_csv(data_loc+file, sep=",", encoding="unicode_escape", engine="python", skiprows=2, skipfooter=1)
    numpy_data=data.to_numpy(copy=True, dtype='float')
    truncate=-1
    real=np.flip(numpy_data[:truncate, 6])
    imag=-np.flip(numpy_data[:truncate,7])
    plot_freq=np.flip(np.log10(numpy_data[:,0]))
    freq=np.flip(numpy_data[:truncate,0])
    phase=np.flip(numpy_data[:,2])

    sim=np.column_stack((real,imag))

    randles={"z1":"R0", "z2":{"p1":[("Q1", "alpha1"), "W1"], "p2":["R2"]}}
    translator=EIS()
    print(len(freq), len(real), len(imag))

    #translator.nyquist(sim, scatter=1, ax=ax)

    bounds={
    "R1":[0, 1000],
    "Q1":[0, 1e-2],
    "alpha1":[0.1, 0.9],
    "R2":[0, 1000],
    "W1":[1, 200]
    }
    true_params={"R1":100, "Q1":1e-3, "alpha1":0.5, "R2":10, "W1":40, "R0":10}
    param_names=["R1", "Q1", "alpha1", "R2", "W1", "R0"]
    optim=EIS_optimiser(circuit=randles, parameter_bounds=bounds, frequency_range=freq, param_names=param_names, test=False, generation_test=True)

    sigma=0.001*np.sum(sim)/(2*len(sim))

    #plt.plot(sim[:,0], -sim[:,1])
    #plt.plot(noisy_data[:,0], -noisy_data[:,1])
    #plt.show()
    exp="Cjx183"
    voltage=folders[z]
    for methods in ["AIC"]:
        results_list=[]
        results_circuit=[]
        true_score=[]
        for i in range(0, 5):
            noisy_data=sim
            #print(sigma, optim.get_std(noisy_data, sim), optim.optimise(noisy_data, sigma,"minimisation", [true_params[x] for x in param_names]))
            #gene_test=EIS_genetics(generation_size=8, generation_test=True, selection="bayes_factors")#, generation_test_save="Generation_images/Bayes_factor_randles_test/round_1/")
            #gene_test.evolve(freq, noisy_data)
            print(len(sim[:,0]), len(sim[:,1]))
            np.save("Best_candidates/{2}/{0}/{4}/testy".format(methods, i+1, exp, abs(truncate), voltage),{})
            gene_test=EIS_genetics(generation_size=12, generation_test=False, selection=methods, initial_tree_size=1, best_record=True, num_top_circuits=6, num_optim_runs=5)


            value,_=gene_test.assess_score(randles, param_names, freq, noisy_data, score_func=methods)
            print(value, "value")
            true_score.append(value)
            num_generations=5
            gene_test.evolve(freq, noisy_data, num_generations=num_generations)
            np.save("Best_candidates/{2}/{0}/{4}/Best_candidates_dict_{1}_12_gen_truncated_{3}_scaled_1.npy".format(methods, i+1, exp, abs(truncate), voltage), gene_test.best_candidates)
            scaled_score=np.divide(value, gene_test.best_array)
            results_list.append(gene_test.best_array)
            results_circuit.append(gene_test.best_circuits)
            #plt.plot(range(0, num_generations),scaled_score)
        np.save("{1}_{0}_best_scores.npy".format(methods, exp), {"scores":results_list, "circuits":results_circuit, "true_score":true_score})
    #plt.show()
