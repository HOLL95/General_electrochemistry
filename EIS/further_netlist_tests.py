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
frequency_powers=np.arange(1, 6, 0.1)
frequencies=[10**x for x in frequency_powers]

for i in range(1, 6):
    file_name="Best_candidates/round_2/{1}/Best_candidates_dict_{0}_12_gen.npy".format(i, "AIC")
    results_dict=np.load(file_name, allow_pickle=True).item()
    #print(results_dict[0]["scores"])
    for j in range(0, len(results_dict["scores"])):
        translator=EIS()
        simulator=EIS_genetics()


        translated_circuit, params=translator.translate_tree(results_dict["models"][j], get_param_list=True)

        #translated_circuit={'z1': {'p1': {'p1': [[{'p1': {'p1': ['C1', 'R1'], 'p2': {'p1': 'R2', 'p2': 'C2'}}, 'p2': {'p1': 'W1', 'p2': 'R3'}}, {'p1': 'W2', 'p2': 'C3'}], ["C12", "C13"]], 'p2': ['W3', 'C4']}, 'p2': {'p1': ('Q1', 'alpha1'), 'p2': 'R4'}}, 'z2': {'p1': 'R5', 'p2': {'p1': "C14", 'p2': 'W4'}}, 'z0': 'R0'}

        print(len(results_dict["parameters"][j]))
        print(len(params))
        circuit_artist(translated_circuit)
        netlist=EIS(circuit=translated_circuit, construct_netlist=True)


        netlist_keys=list(netlist.netlist_dict.keys())
        net_dict=netlist.netlist_dict
        cpe_num=0
        for key in netlist_keys:
            if "Q" in key:
                cpe_num+=1
        if "gamma1" in netlist_keys:
            continue
        f=open("temp_netlist.net", "w")
        f.write("V1 0 1 1\n")
        sim_dict=dict(zip(params, results_dict["parameters"][j]))
        for key in netlist_keys:
            if "R" in key or "C" in key:
                if key != "R0":
                    write_string="{0} {1} {2} 1 \n".format(key, net_dict[key]["left"],net_dict[key]["right"] )
                else:
                    write_string="R0 {1} 0 1 \n".format(key, net_dict[key]["left"])
                f.write(write_string)
            elif "Q" in key:
                write_string="H{0} {1} {2} V1 1\n".format(key[1], net_dict[key]["left"],net_dict[key]["right"])
                f.write(write_string)
            elif "W" in key:
                current_num=int(key[1])
                warburg_num=cpe_num+current_num

                w_val=sim_dict[key]
                del sim_dict[key]
                sim_dict["alpha"+str(warburg_num)]=0.5
                sim_dict["Q"+str(warburg_num)]=1/(np.sqrt(2)*w_val)
                print(sim_dict)
                write_string="H{0} {1} {2} V1 1\n".format(warburg_num, net_dict[key]["left"],net_dict[key]["right"])
                f.write(write_string)
        f.close()
        from model_creator_class import model_creator
        param_dict_str="{"
        for key in sim_dict.keys():
            param_dict_str+="\""+key+"\":"+str(sim_dict[key])+","
        param_dict_str+="}"
        print(param_dict_str)
        model_creator(0.001, "dt*10000", "0.03*np.sin(math.pi*2*8*t)", param_dict_str)
        import temp_model
        current=temp_model.external_simulate()
        plt.show()
