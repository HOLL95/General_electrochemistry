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
        'alpha1': 0.9710356115630412} 
freq=files["freq"]
keys=["k_0","gamma","Cdl",
            "Ru",
            "alpha",
            "phase",
            "cap_phase"]
set_keys=set(keys)
k0_1={'R0': 103.42115051106593, 'R1': 10314.65622036976, 'C1': 0.0003923123750362509, 'C2': 4.8714659815844195e-05, 'R3': 702860.1095345016, 'Q1': 4.0932950630502546e-05, 'alpha1': 0.9859474865077266}
k0_2={'R0': 104.75927753909005, 'R1': 87999720951.73364, 'C1': 2.4415039062500004, 'C2': 4.294705609558665e-05, 'R3': 6149.528835220614, 'Q1': 4.51804970694856e-05, 'alpha1': 0.9830845139965573}
k0_3={'R0': 104.43423223973389, 'R1': 173.2882141113281, 'C1': 9.381079778390987e-05, 'C2': 8.864907405852452e-05, 'R3': 512845741.8921814, 'Q1': 0.07956542968750002, 'alpha1': 0.8037089824676513}
k0_4={'R0': 104.8167259302378, 'R1': 30762994992.39471, 'C1': 8.812011924130398e-05, 'C2': 9.536579165611838e-05, 'R3': 139.49623929071842, 'Q1': 0.04332950413227081, 'alpha1': 0.7402420043945312}

gamma_1={'R0': 102.62637826993334, 'R1': 78901750253.24878, 'C1': 2.18115234375, 'C2': 4.370517845082303e-05, 'R3': 759060.375809127, 'Q1': 3.119879912865191e-05, 'alpha1': 0.9745833545246042}
gamma_2={'R0': 102.4479373365617, 'R1': 521534352.29333496, 'C1': 8.572101337972233e-05, 'C2': 8.95027932373879e-05, 'R3': 161905.77020874026, 'Q1': 0.0021285899449139827, 'alpha1': 0.378851318359375}
gamma_3={'R0': 102.67382784246583, 'R1': 457938180.90186155, 'C1': 9.48503606767742e-05, 'C2': 8.09844794508612e-05, 'R3': 161386.6519165039, 'Q1': 0.2139453887939453, 'alpha1': 0.4961761474609374}
gamma_4={'R0': 103.74273972093124, 'R1': 2624284876.2144527, 'C1': 0.00010631399359766767, 'C2': 7.267845956918804e-05, 'R3': 3.1237758615946716e-11, 'Q1': 7.667224557631505e-06, 'alpha1': 0.07779954388038812}
results_dict={"k_0":[k0_1, k0_2, k0_3, k0_4],
                "gamma":[gamma_1, gamma_2, gamma_3, gamma_4]}
results_keys=results_dict.keys()
other_results_dict=np.load( "BV_param_scans_best_fits.npy", allow_pickle=True).item()
other_results_keys=list((set(keys)^set(results_keys)))
desired_ones=["gamma", "k_0", "Cdl", "alpha", "Ru"]

fig, ax=plt.subplots(7,4)
extra_dict={"data":files, "results":{}}
for i in range(0, len(keys)):
    
    current_key=keys[i]
    
    if current_key in results_keys:
        current_dict=results_dict
        exp_keys=list(files[current_key].keys())
        data_dict=files[current_key]
    else:
        current_dict=other_results_dict
        exp_keys=[x for x in other_results_dict[current_key].keys() if "results" not in x]
        data_dict=other_results_dict[current_key]
    print(exp_keys)
    if current_key in desired_ones:
        extra_dict["results"][current_key]={}
    for j in range(0, len(exp_keys)):
        
        phase=data_dict[exp_keys[j]][:,0]
        magnitude=data_dict[exp_keys[j]][:,1]

        fit_data=np.column_stack((phase, np.log10(magnitude)))
        #EIS().bode(np.column_stack((phase, np.log10(magnitude))), freq, data_type="phase_mag")
        #plt.show()


        mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
        simulator=EIS(circuit=mark_circuit)
        if current_key in results_keys:
            results=current_dict[current_key][j]
        else:
            results=current_dict[current_key][exp_keys[j]+"+results"]
        if current_key in desired_ones:
            extra_dict["results"][current_key][exp_keys[j]+"_results"]=results
        axis=ax[i, j]#
        twinxis=axis.twinx()
        #axis.set_title(current_key+"="+str(exp_keys[j]))
        EIS().bode(fit_data, freq, data_type="phase_mag", ax=axis, twinx=twinxis, compact_labels=True)
        EIS().bode(simulator.test_vals(results, freq), freq,  ax=axis, twinx=twinxis, compact_labels=True)
np.save("BV_param_scans_best_fits_circuit_3_a.npy", extra_dict)
plt.show()
   
