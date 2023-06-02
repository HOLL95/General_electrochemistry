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

files=np.load("BV_param_scans.npy", allow_pickle=True).item()
values={'R0': 104.82889735474738, 'R1': 1525.2234882630044, 
        'C1': 0.00011241593575305592, 'C2': 6.770333588293525e-05, 
        'R3': 3350.5992740459615, 'Q1': 2.1311350351085948e-05, 
        'alpha1': 0.9710356115630412, "C3":2.1311350351085948e-05, } 
freq=files["freq"]
keys=["k_0","gamma","Cdl",
            "Ru",
            "alpha"]

circuit_1={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}

circuit_2={ "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}
circuit_3={ "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
circuits=[circuit_1, circuit_2, circuit_3, mark_circuit]
names=[EIS(circuit=x).param_names for x in circuits]
print(names)

for z in range(3, len(circuits)):
        new_dict={}
        for i in range(0, len(keys)):
                current_key=keys[i]
                new_dict[current_key]={}
                exp_keys=list(files[current_key].keys())
                for j in range(0, len(files[current_key])):
                        
                        phase=files[current_key][exp_keys[j]][:,0]
                        magnitude=files[current_key][exp_keys[j]][:,1]

                        fit_data=np.column_stack((phase, np.log10(magnitude)))
                        #EIS().bode(np.column_stack((phase, np.log10(magnitude))), freq, data_type="phase_mag")
                        #plt.show()


                        mark_circuit=circuits[z]
                        current_names=names[z]

                        gene_test=EIS_genetics(generation_size=12, generation_test=True, individual_test=False,
                                                selection="AIC", initial_tree_size=1, 
                                                best_record=True, num_top_circuits=6, num_optim_runs=20, data_representation="bode", construction_elements=["R", "C", "CPE"])
                        value,return_params, sim_data=gene_test.assess_score(mark_circuit, current_names, freq, fit_data, score_func="AIC", data_representation="bode")
                        #return_params=np.random.rand(len(current_names))
                        results=dict(zip(current_names, return_params))
                        #print(results)
                        #files[current_key][exp_keys[j]+"+results"]=results
                        new_dict[current_key][exp_keys[j]+"_results"]=results
        save_dict={"data":files, "results":new_dict}
        np.save( "BV_param_scans_best_fits_circuit_{0}_a.npy".format(z), save_dict)

