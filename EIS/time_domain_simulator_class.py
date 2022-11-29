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
from model_creator_class import model_creator
class time_domain:
    def __init__(self, circuit, **kwargs):
        if "params" not in kwargs:
            raise ValueError("Need parameter values!")
        if isinstance(kwargs["params"], list):
            if "parameter_names" not in kwargs:
                raise ValueError("Need list of parameter names")
            else:
                sim_dict=dict(zip(kwargs["params"], kwargs["parameter_names"]))
        elif isinstance(kwargs["params"], dict):
            sim_dict=kwargs["params"]
        if "potential_func" not in kwargs:
            kwargs["potential_func"]="0.3*np.sin(math.pi*2*8*t)"
        netlist_maker=EIS()
        netlist=EIS(circuit=circuit, construct_netlist=True)
        netlist_keys=list(netlist.netlist_dict.keys())
        net_dict=netlist.netlist_dict
        cpe_num=0
        for key in netlist_keys:
            if "Q" in key:
                cpe_num+=1
        if "gamma1" in netlist_keys:
            raise ValueError("Not valid for finite Warburg element")
        f=open("temp_netlist.net", "w")
        f.write("V1 0 1 1\n")
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
        param_dict_str="{"
        for key in sim_dict.keys():
            param_dict_str+="\""+key+"\":"+str(sim_dict[key])+","
        param_dict_str+="}"
        print(param_dict_str)
        model_creator(0.0001, "dt*1000", kwargs["potential_func"], param_dict_str)
    def simulate(self):
        import temp_model
        current, time, potential=temp_model.external_simulate()
        return current, time, potential
