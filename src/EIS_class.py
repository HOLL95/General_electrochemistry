import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import collections
import math
import random
import warnings
from collections import deque
from uuid import uuid4

#TODO->paralell in series+ gerischer
class EIS:
    def __init__(self,  **kwargs,):
        self.options={}
        self.options_checker(**kwargs)
        
        self.all_nodes=["paralell", "series",]#"element"]
        self.all_elems=self.options["construction_elements"]
      
        self.num_nodes=len(self.all_nodes)-1
        self.num_elems=len(self.all_elems)-1
        if "circuit" not in kwargs:
            return
        if self.options["fitting"]==True:
            if kwargs["parameter_bounds"] is None:
                raise ValueError("Need to define parameter bounds when fitting")
            else:
                self.param_bounds=kwargs["parameter_bounds"]
                if "parameter_names" not in kwargs:
                    warnings.warn("Setting the keys of the boundary dictionary as names - be wary!")
                    self.param_names=self.param_bounds.keys()
                else:
                    self.param_names=kwargs["parameter_names"]
        elif "parameter_names" not in kwargs:
            warnings.warn("Trying to extract parameter names from provided circuit - behaviour will be more consistent if these names are manually inputted")
            self.param_names=[]
            circuit=copy.deepcopy(kwargs["circuit"])
            self.construct_circuit(circuit, func=self.get_param_names_from_dict)

        if self.options["integrated_circuit"]==True:
            circuit=copy.deepcopy(kwargs["circuit"])
            self.circuit=self.construct_circuit(circuit, func=self.define_z_func)
        if self.options["construct_netlist"]==True:
            self.netlist_dict=dict(zip(self.param_names, [{"left":None, "right":None} for x in range(0, len(self.param_names))]))
            self.node_counter=1
            circuit=copy.deepcopy(kwargs["circuit"])
            self.identifier_list={}
            self.uuid=None
            prev_elem_is_dict=False
            for key in circuit.keys():
                if prev_elem_is_dict==True:
                    self.node_counter=self.identifier_list[prev_key]
                z=self.construct_netlist(circuit[key], self.node_counter, key)
                self.identifier_list[key]=self.node_counter+1
                prev_key=key
                if isinstance(circuit[key], dict):
                    prev_elem_is_dict=True
                else:
                    prev_elem_is_dict=False
            #print(self.netlist_dict)
            identity_keys=self.identifier_list.keys()
            for key in self.netlist_dict.keys():
                if self.netlist_dict[key]["right"] in identity_keys:
                    self.netlist_dict[key]["right"]=self.identifier_list[self.netlist_dict[key]["right"]]


    def construct_circuit(self, circuit, func):

        if isinstance(circuit, dict):
            in_root=True
            for key in circuit.keys():
                if isinstance(circuit[key], dict):
                    in_root=False
                    self.construct_circuit(circuit[key], func)
                elif  isinstance(circuit[key], list):
                    circuit[key]=func(circuit[key])
                elif "z" in key:
                    circuit[key]=func(circuit[key], "series")
                elif isinstance(circuit[key], tuple):
                    circuit[key]=func(circuit[key])
                elif in_root==False and isinstance(circuit[key], str):
                    circuit[key]=func(circuit[key], "series")
            if in_root==True:
                for key in circuit.keys():
                    if not callable(circuit[key]):
                        circuit[key]=func(circuit[key])
            else:
                for key in circuit.keys():
                    if isinstance(circuit[key], str):
                        if "p" in key:
                            circuit[key]=func(circuit[key],"paralell")
                        else:
                            circuit[key]=func(circuit[key],"series")
        return circuit
    def construct_netlist(self, circuit, previous_node, current_z_key):
        print(self.node_counter, previous_node, circuit)
        if isinstance(circuit, dict):
            in_root=True
            for key in circuit.keys():
                if isinstance(circuit[key], dict):
                    self.construct_netlist(circuit[key], previous_node, current_z_key)
                elif isinstance(circuit[key], list):
                    self.construct_netlist(circuit[key], previous_node, current_z_key)
                elif isinstance(circuit[key], str):
                    self.netlist_dict[circuit[key]]["left"]=previous_node
                    self.netlist_dict[circuit[key]]["right"]=current_z_key
                elif isinstance(circuit[key], tuple):
                    self.netlist_dict[circuit[key][0]]["left"]=previous_node
                    self.netlist_dict[circuit[key][0]]["right"]=current_z_key
        elif isinstance(circuit, list):
            circuit_list=self.list_flatten(circuit)
            series_len=len(circuit_list)
            previous_elem_is_dict=False
            for i in range(0, series_len):
                if previous_elem_is_dict==True:
                    print(identifier)
                    print(circuit_list[i])
                    self.identifier_list[identifier]=self.node_counter
                    previous_node=self.node_counter
                element = circuit_list[i]
                if isinstance(element, str) or isinstance(element, tuple):

                    if isinstance(element, tuple):
                        element=element[0]
                    print("here", element, previous_node)
                    self.netlist_dict[element]["left"]=previous_node

                    if i!=series_len-1:
                        self.netlist_dict[element]["right"]=self.node_counter+1
                        self.node_counter+=1

                    else:
                        self.netlist_dict[element]["right"]=current_z_key
                    previous_node=self.node_counter
                    previous_elem_is_dict=False
                if isinstance(element, dict):
                    if i!=series_len-1:

                        identifier=str(uuid4())
                        print(self.node_counter, previous_node, "#")
                        self.identifier_list[identifier]=identifier
                        self.construct_netlist(element, previous_node, identifier)
                        self.node_counter+=1
                        previous_node+=1
                        previous_elem_is_dict=True
                    else:
                        print(self.node_counter, previous_node, "~")
                        identifier=current_z_key
                        self.construct_netlist(element, previous_node, identifier)
                        previous_node+=1


        elif isinstance(circuit, str):
            self.netlist_dict[circuit]["left"]=previous_node
            self.netlist_dict[circuit]["right"]=self.node_counter+1
            self.node_counter+=1
            previous_node+=1
        elif isinstance(circuit, tuple):
            circuit=circuit[0]
            self.netlist_dict[circuit]["left"]=previous_node
            self.netlist_dict[circuit]["right"]=self.node_counter+1
            self.node_counter+=1
            previous_node+=1

    def flatten(self, d, parent_key='', sep='-'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(self.flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    def list_flatten(self, list_of_lists):
        if len(list_of_lists) == 0:
            return list_of_lists
        if isinstance(list_of_lists[0], list):
            return self.list_flatten(list_of_lists[0]) + self.list_flatten(list_of_lists[1:])
        return list_of_lists[:1] + self.list_flatten(list_of_lists[1:])



    def dict_compress(self, d, parent_key='', sep='_', optional_escape=None, extract_lists=False):
        items = []
        for k in d.keys():
            if optional_escape is not None:
                #print(d)
                if optional_escape in k:
                    return {parent_key:d}
        for k, v in d.items():

            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.dict_compress(v, new_key, sep=sep, optional_escape=optional_escape, extract_lists=extract_lists).items())
            elif extract_lists==True:
                if isinstance(v, list):
                    for i in range(0, len(v)):
                        series_key=new_key+sep+"series"+sep+str(i+1)
                        items.extend(self.dict_compress(v[i], series_key, sep=sep, optional_escape=optional_escape, extract_lists=extract_lists).items())
            else:
                items.append((new_key, v))
        return dict(items)
    def define_z_func(self, param, flag="paralell"):
        if  isinstance(param, list):
            fun_list=[]
            paralell=False
            param=self.list_flatten(param)
            for parameter in param:
                if isinstance(parameter, dict):
                    func_dict=self.construct_circuit(parameter, self.define_z_func)
                    flattened_list=self.flatten(func_dict).values()
                    def f(**kwargs):
                        return_arg=[x(**kwargs) for x in flattened_list]
                        return 1/np.sum(return_arg)
                elif isinstance(parameter, tuple):
                    if len(parameter)==2:
                        if "Q" in parameter[0] and "alpha" in parameter[1]:
                            f=self.z_functions("CPE", parameter, paralell=paralell)
                        else:
                            raise KeyError("parameters not in right order for CPE or finite warburg")
                    elif len(parameter)==3:
                        if "gamma" in parameter[0] and "delta" in parameter[1] and "D" in parameter[2]:
                            f=self.z_functions("warburg_finite", parameter, paralell=paralell)
                        else:
                            raise KeyError("parameters not in right order for CPE or finite warburg")
                elif "C" in parameter:
                    f=self.z_functions("capacitor", parameter, paralell=paralell)
                elif "R" in parameter:
                    f=self.z_functions("resistor", parameter, paralell=paralell)
                elif "W" in parameter:
                    f=self.z_functions("warburg_inf", parameter, paralell=paralell)
                fun_list.append(f)
            def F(**kwargs):
                return_arg=[x(**kwargs) for x in fun_list]
                return 1/np.sum(return_arg)
            return F
        elif flag=="series":
            paralell=False
        else:
            paralell=True
        if isinstance(param, tuple):
            if len(param)==2:
                if "Q" in param[0] and "alpha" in param[1]:
                    F=self.z_functions("CPE", param, paralell=paralell)
                else:
                    raise KeyError("parameters not in right order for CPE or finite warburg")
            elif len(param)==3:
                if "gamma" in param[0] and "delta" in param[1] and "D" in param[2]:
                    F=self.z_functions("warburg_finite", param, paralell=paralell)
                else:
                    raise KeyError("parameters not in right order for CPE or finite warburg")
        else:
            if "C" in param:
                F=self.z_functions("capacitor", param, paralell=paralell)
            elif "R" in param:
                F=self.z_functions("resistor", param, paralell=paralell)
            elif "W" in param:
                F=self.z_functions("warburg_inf", param, paralell=paralell)
        return F
    def get_param_names_from_dict(self, param, flag="paralell"):
        if  isinstance(param, list):
            fun_list=[]
            paralell=False
            param=self.list_flatten(param)
            for parameter in param:
                if isinstance(parameter, dict):
                    func_dict=self.construct_circuit(parameter, self.get_param_names_from_dict)
                elif isinstance(parameter, tuple):
                    for i in range(0, len(parameter)):
                        self.param_names.append(parameter[i])
                elif isinstance(parameter, str):
                    self.param_names.append(parameter)
        elif isinstance(param, tuple):
            for i in range(0, len(param)):
                self.param_names.append(param[i])
        elif isinstance(param, str):
            self.param_names.append(param)
    def z_functions(self, circuit_type,param, paralell):
        if circuit_type=="capacitor":
            if paralell==False: 
                def F(**kwargs):
                 return 1/(1j*kwargs["omega"]*kwargs[param])
            else:
                def F(**kwargs):
                 return 1j*kwargs["omega"]*kwargs[param]
        elif circuit_type=="resistor":
            if paralell==False:
                def F(**kwargs):
                    return kwargs[param]#
            else:
                #print("fire2")
                def F(**kwargs):
                    return 1/kwargs[param]#
        elif circuit_type=="warburg_inf":

            if paralell==False:
                def F(**kwargs):
                    denom=np.sqrt(kwargs["omega"])
                    common_term=kwargs[param]/denom
                    return  common_term-1j*common_term
            else:
                def F(**kwargs):

                    denom=np.sqrt(1j*kwargs["omega"])
                    return  denom/kwargs[param]
        elif circuit_type=="warburg_finite":
            if paralell==False:
                def F(**kwargs):
                    B=kwargs[param[1]]/np.sqrt(kwargs[param[2]])
                    return (1/kwargs[param[0]])*np.tanh(B*np.sqrt(1j*kwargs["omega"]))
            else:
                def F(**kwargs):

                    B=kwargs[param[1]]/np.sqrt(kwargs[param[2]])
                    return 1/((1/kwargs[param[0]])*np.tanh(B*np.sqrt(1j*kwargs["omega"])))
        elif circuit_type=="CPE":
            if paralell==False:
                def F(**kwargs):
                    #print(kwargs[param[0]]*np.power(kwargs["omega"]*1j, kwargs[param[1]]))
                    return 1/(kwargs[param[0]]*np.power(kwargs["omega"]*1j, kwargs[param[1]]))
            else:
                def F(**kwargs):
                    #print(param[0], param[1], kwargs["omega"], np.float64(kwargs["omega"])*1j)
                    #print(kwargs[param[0]], kwargs[param[1]], np.power(kwargs["omega"]*1j, kwargs[param[1]]))
                    return  (kwargs[param[0]]*np.power(kwargs["omega"]*1j, kwargs[param[1]]))

        return F


    def define_model(self, param, flag="paralell"):

        if  isinstance(param, list):
            fun_list=[]
            for parameter in param:
                if "C" in parameter:
                    f="(1/jf"+parameter+")"
                elif "R" in parameter:
                    f="("+parameter+")"
            fun_list.append(f)
            F=("+").join(fun_list)
        elif flag=="series":
            if "C" in param:
                F="(1/jf"+param+")"
            elif "R" in param:
                F="("+param+")"
        else:
            if "C" in param:
                #as its paralell it's the inverse!
                F="(jf"+param+")"
            elif "R" in param:
                F="1/("+param+")"
        return F
    def write_model(self, circuit):
        print(circuit)
        self.construct_circuit(circuit, self.define_model)
        print(circuit)
        func_dict={}
        for key in circuit.keys():
            if isinstance(circuit[key], dict):
                flattened_list=self.flatten(circuit[key]).values()
                write_list="1/("+("+").join(flattened_list)+")"
                func_dict[key]=write_list
            else:
                func_dict[key]=circuit[key]
        for key in func_dict:
            final_list=[func_dict[key] for key in func_dict.keys()]
            final_list=("+").join(final_list)
        print(final_list)
    def freq_simulate(self, **kwargs):

        z=0j
        for key in self.circuit.keys():
            if isinstance(self.circuit[key], dict):
                flattened_list=self.flatten(self.circuit[key]).values()

                vals=[x(**kwargs) for x in flattened_list]
                z+=1/np.sum(vals)
            else:
                z+=self.circuit[key](**kwargs)

        return z
    def add_noise(self, series, sd):
        return np.add(series, np.random.normal(0, sd, len(series)))
    def normalise(self, norm, boundaries):
        return  (norm-boundaries[0])/(boundaries[1]-boundaries[0])
    def un_normalise(self, norm, boundaries):
        return (norm*(boundaries[1]-boundaries[0]))+boundaries[0]
    def RMSE(self, y, y_data):
        return np.mean(np.sqrt(np.square(np.subtract(y, y_data))))
    def n_outputs(self):
        return 2
    def n_parameters(self):
        return len(self.param_bounds.keys())
    def change_norm_group(self, param_list, method, return_type="list"):
        if method=="un_norm":
            if return_type=="dict":
                normed_params={x:self.un_normalise(param_list[x], self.param_bounds[x]) for x in self.param_names}
            elif return_type=="list":
                normed_params=[self.un_normalise(param_list[x], self.param_bounds[self.param_names[x]]) for x in range(0, len(self.param_names))]
        return normed_params
    def simulate(self, parameters, frequencies):
        sim_params=dict(zip(self.param_names, parameters))

        if self.options["normalise"]==True:
            normed_params=self.change_norm_group(sim_params, "un_norm", return_type="dict")
        else:
            normed_params=sim_params
        #print(sim_params)
        #print(normed_params)
        #print("++"*20)
        spectra=np.zeros(len(frequencies), dtype="complex")
        for i in range(0, len(frequencies)):
            normed_params["omega"]=frequencies[i]
            spectra[i]=self.freq_simulate(**normed_params)
        if self.options["test"]==True:
            print(normed_params, sim_params, self.options["normalise"])
            plt.plot(np.real(spectra), -np.imag(spectra))
            plt.show()
        
        
        if self.options["data_representation"]=="nyquist":
            if self.options["invert_imaginary"]==True:
                return np.column_stack((np.real(spectra), -np.imag(spectra)))
            else:
                return np.column_stack((np.real(spectra), np.imag(spectra)))
        else:

            real=np.real(spectra)
            imag=np.imag(spectra)
            #plt.plot(real, -imag)
            #plt.show()
            phase=np.angle(spectra, deg=True)#np.arctan(np.divide(-imag, real))*(180/math.pi)
            magnitude=np.log10(np.abs(spectra))#np.divide(np.add(np.square(real), np.square(imag)), 1000)
            return_arg=np.column_stack((phase, magnitude))
            #self.bode(return_arg, frequencies,data_type="phase_mag")
            #ax=plt.gca()
            #ax.set_title("test")
            #plt.show()
            #if self.options["test"]==True:
            #self.bode(np.column_stack((real, imag)), frequencies)
            #plt.show()
            return return_arg
    def convert_to_bode(self,spectra):
        spectra=[complex(x, y) for x,y in zip(spectra[:,0], spectra[:,1])]
        phase=np.angle(spectra, deg=True)#np.arctan(np.divide(-spectra[:,1], spectra[:,0]))*(180/math.pi)
        #print(np.divide(spectra[:,1], spectra[:,0]))
        magnitude=np.log10(np.abs(spectra))
        return np.column_stack((phase,magnitude))
    def nyquist(self, spectra, **kwargs):
        if "ax" not in kwargs:
            _,kwargs["ax"]=plt.subplots(1,1)
        if "scatter" not in kwargs:
            kwargs["scatter"]=0
        if "label" not in kwargs:
            kwargs["label"]=None
        if "linestyle" not in kwargs:
            kwargs["linestyle"]="-"
        if "marker" not in kwargs:
            kwargs["marker"]="o"
        if "colour" not in kwargs:
            kwargs["colour"]=None
        if "orthonormal" not in kwargs:
            kwargs["orthonormal"]=True

        ax=kwargs["ax"]
        imag_spectra_mean=np.mean(spectra[:,1])
        if imag_spectra_mean<0:

            ax.plot(spectra[:,0], -spectra[:,1], label=kwargs["label"], linestyle=kwargs["linestyle"], color=kwargs["colour"])
        else:
            ax.plot(spectra[:,0], spectra[:,1], label=kwargs["label"], linestyle=kwargs["linestyle"], color=kwargs["colour"])
        ax.set_xlabel("$Z_{Re}$ ($\\Omega$)")
        ax.set_ylabel("$-Z_{Im}$ ($\\Omega$)")
        total_max=max(np.max(spectra[:,0]), np.max(-spectra[:,1]))
        if kwargs["orthonormal"]==True:
            ax.set_xlim([0, total_max+0.1*total_max])
            ax.set_ylim([0, total_max+0.1*total_max])
        if kwargs["scatter"]!=0:
            if imag_spectra_mean<0:
                ax.scatter(spectra[:,0][0::kwargs["scatter"]], -spectra[:,1][0::kwargs["scatter"]], marker=kwargs["marker"], color=kwargs["colour"])
            else:
                ax.scatter(spectra[:,0][0::kwargs["scatter"]], spectra[:,1][0::kwargs["scatter"]], marker=kwargs["marker"], color=kwargs["colour"])

    def bode(self, spectra,frequency, **kwargs):
        if "ax" not in kwargs:
            _,kwargs["ax"]=plt.subplots(1,1)
        if "label" not in kwargs:
            kwargs["label"]=None
        if "type" not in kwargs:
            kwargs["type"]="both"
        if "twinx" not in kwargs:
            kwargs["twinx"]=kwargs["ax"].twinx()
        if "data_type" not in kwargs:
            kwargs["data_type"]="complex"
        if "compact_labels" not in kwargs:
            kwargs["compact_labels"]=False
        if "lw" not in kwargs:
            kwargs["lw"]=1.5
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        if "scatter" not in kwargs:
            kwargs["scatter"]=False
        if kwargs["data_type"]=="complex":
            spectra=[complex(x, y) for x,y in zip(spectra[:,0], spectra[:,1])]
            phase=np.angle(spectra, deg=True)#np.arctan(np.divide(-spectra[:,1], spectra[:,0]))*(180/math.pi)
            #print(np.divide(spectra[:,1], spectra[:,0]))
            magnitude=np.log10(np.abs(spectra))#np.add(np.square(spectra[:,0]), np.square(spectra[:,1]))
        elif kwargs["data_type"]=="phase_mag":
            phase=spectra[:,0]
            magnitude=spectra[:,1]
            if "data_is_log" not in kwargs:
                kwargs["data_is_log"]=True
            if kwargs["data_is_log"]==False:
                magnitude=np.log10(magnitude)
            
            
        ax=kwargs["ax"]
        ax.set_xlabel("$\\log_{10}$(Frequency)")
        x_freqs=np.log10(frequency)
        if kwargs["type"]=="both":
            twinx=kwargs["twinx"]
            ax.plot(x_freqs, -phase, label=kwargs["label"], lw=kwargs["lw"], alpha=kwargs["alpha"])
            
            if kwargs["compact_labels"]==False:
                ax.set_ylabel("-Phase")
                twinx.set_ylabel("Magnitude")
            else:
                ax.text(x=-0.05, y=1.05, s="$-\\psi$", fontsize=12, transform=ax.transAxes)
                ax.text(x=0.96, y=1.05, s="$\\log_{10}(|Z|) $", fontsize=12, transform=ax.transAxes)
            twinx.plot(x_freqs, magnitude, linestyle="--", lw=kwargs["lw"], alpha=kwargs["alpha"])
            if kwargs["scatter"] is not False:
                ax.scatter(x_freqs, -phase)
                twinx.scatter(x_freqs, magnitude, marker="v")
            
        elif kwargs["type"]=="phase":
            if kwargs["compact_labels"]==False:
                ax.set_ylabel("Phase")
            else:
                 ax.text(x=-0.05, y=1.05, s="$\\psi$", fontsize=12, transform=ax.transAxes)
            ax.plot(x_freqs, -phase, label=kwargs["label"], lw=kwargs["lw"], alpha=kwargs["alpha"])

        elif kwargs["type"]=="magnitude":
            
            ax.plot(x_freqs, magnitude, label=kwargs["label"], lw=kwargs["lw"], alpha=kwargs["alpha"])
            if kwargs["compact_labels"]==False:
                ax.set_ylabel("Magnitude")
            else:
                 ax.text(x=-0.05, y=1.05, s="$|Z|$", fontsize=12, transform=ax.transAxes)
        if kwargs["label"]!=None:
            kwargs["ax"].legend()

    def test_vals(self, parameters, frequencies):
        normalise=self.options["normalise"]
        self.options["normalise"]=False
        if isinstance(parameters, (list, np.ndarray)):
            list_params=parameters
        elif isinstance(parameters, dict):
            list_params=[parameters[x] for x in self.param_names]
        results=self.simulate(list_params, frequencies)
        self.options["normalise"]=normalise
        return results
    def find_random_node(self, tree, n, parent):
        if isinstance(tree, str):
            return parent
        if n==0:
            return tree
        direction=random.randint(0, 1)
        for key in tree.keys():
            if "left" in key:
                left_key=key
            elif "right" in key:
                right_key=key#
        if direction==0:
            next_key=left_key
        else:
            next_key=right_key
        return self.find_random_node(tree[next_key], n-1, tree)
    def replace_random_node(self, tree, n, replacing_node, parent, parent_key):
        if n==0 or isinstance(tree, str):
            parent[parent_key]=replacing_node
        else:
            direction=random.randint(0, 1)
            for key in tree.keys():
                if "left" in key:
                    left_key=key
                elif "right" in key:
                    right_key=key#
            if direction==0:
                next_key=left_key
            else:
                next_key=right_key
            parent[parent_key]=self.replace_random_node(tree[next_key], n-1, replacing_node, tree, next_key)
        return parent
    #def tree_depth(self, d):
    #    try:
    #         if isinstance(d, dict):
    #             return 1 + (max(map(self.tree_depth, d.values())) if d else 0)
    #         return 0
    #    except:
    #        return 10
    def tree_depth(self, d):
        queue = deque([(id(d), d, 1)])
        memo = set()
        while queue:
            id_, o, level = queue.popleft()
            if id_ in memo:
                continue
            memo.add(id_)
            if isinstance(o, dict):
                queue += ((id(v), v, level + 1) for v in o.values())
        return level
    def crossover(self, original_tree, target_tree, min_depth=1):
        target_tree_depth=self.tree_depth(target_tree)
        orig_tree_depth=self.tree_depth(original_tree)
        orig_tree_key=self.get_top_key(original_tree)
        rand_val=random.randint(min_depth, orig_tree_depth)
        crossover_node=self.find_random_node(original_tree[orig_tree_key], rand_val, original_tree)
        target_tree_key=self.get_top_key(target_tree)
        mutated_tree=self.replace_random_node(target_tree[target_tree_key], random.randint(min_depth, target_tree_depth), crossover_node, target_tree, target_tree_key)
        return mutated_tree
    def get_top_key(self, d):
        return list(d)[0]
    def mutation(self, target_tree, max_mutant_size=3,min_depth=1):
        target_tree_depth=self.tree_depth(target_tree)
        mutant_addition=self.generate_random_node(max_mutant_size)
        target_tree_key=self.get_top_key(target_tree)
        mutated_tree=self.replace_random_node(target_tree[target_tree_key], random.randint(min_depth, target_tree_depth), mutant_addition, target_tree, target_tree_key)
        return mutated_tree
    def random_circuit_tree(self, max_depth):
        nodes=["paralell", "series"]
        root_idx=random.randint(0,1)
        root_node="root_"+nodes[root_idx]
        random_tree={root_node:self.generate_random_node(max_depth)}
        return random_tree

        #for i in range(0, len(depths)):

    def generate_random_node(self, n):
        if n<1:
            return {"left_element":self.all_elems[random.randint(0, self.num_elems)], "right_element":self.all_elems[random.randint(0, self.num_elems)]}
        left_n=random.randint(0, n)
        keys=[x+y for x, y in zip(["left_", "right_"], [self.all_nodes[random.randint(0, self.num_nodes)] for x in range(0,2)])]
        return {keys[0]:self.generate_random_node(left_n), keys[1]:self.generate_random_node(n-left_n-1)}


    def construct_dict_from_tree(self, tree, parent_key):
        if "root" in parent_key:
            self.element_counter={"R":0, "C":0, "W_inf":0, "W_fin":0, "CPE":0}
            self.param_list=["R0"]
        if isinstance(tree, str):
            self.element_counter[tree]+=1
            if tree=="R" or tree=="C":
                parameter="{0}{1}".format(tree, self.element_counter[tree])
                self.param_list.append(parameter)
                return parameter
            elif tree=="W_inf":
                parameter="{0}{1}".format("W", self.element_counter[tree])
                self.param_list.append(parameter)
                return parameter
            elif tree=="W_fin":
                parameter=tuple(["{0}{1}".format(x, self.element_counter[tree]) for x in ["gamma", "delta", "D"]])
                for element in parameter:
                    self.param_list.append(element)
                return parameter
            elif tree=="CPE":
                parameter=tuple(["{0}{1}".format(x, self.element_counter[tree]) for x in ["Q", "alpha"]])
                for element in parameter:
                    self.param_list.append(element)
                return parameter
            elif "series" in tree or "paralell" in tree:
                return None
        for key in tree.keys():
            #print(parent_key, tree.keys())
            if "left" in key:
                left_key=key
            elif "right" in key:
                right_key=key#
        if "series" in parent_key:
            return self.series_combinator(self.construct_dict_from_tree(tree[left_key], left_key), self.construct_dict_from_tree(tree[right_key], right_key))
        elif "paralell" in parent_key:
            return self.paralell_combinator(self.construct_dict_from_tree(tree[left_key], left_key), self.construct_dict_from_tree(tree[right_key], right_key))
        elif "element" in parent_key:
            return None

    def series_combinator(self, elem_1, elem_2):
        if elem_1 is None:
            return elem_2
        elif elem_2 is None:
            return elem_1
        return [elem_1, elem_2]
    def paralell_combinator(self, elem_1, elem_2):
        if elem_1 is None:
            return elem_2
        elif elem_2 is None:
            return elem_1
        return {"p1":elem_1, "p2":elem_2}
    def construct_circuit_from_tree_dict(self,tree_dict):
        if isinstance(tree_dict, list):
            pass_list=self.list_flatten(tree_dict)
            z_counter=0
            tree_dict={}
            for i in range(0, len(pass_list)):
                if pass_list[i] is None:
                    continue
                else:
                    z_counter+=1
                    tree_dict["z{0}".format(z_counter)]=pass_list[i]
        else:
            tree_dict={"z1":tree_dict}
        tree_dict["z0"]="R0"
        return tree_dict
    def translate_tree(self, circuit_tree, get_param_list=False):
        top_key=self.get_top_key(circuit_tree)
        circuit_dictionary=self.construct_dict_from_tree(circuit_tree[top_key], top_key)
        functional_dictionary=self.construct_circuit_from_tree_dict(circuit_dictionary)
        if get_param_list==False:
            return functional_dictionary
        else:
            return functional_dictionary, self.param_list

    def options_checker(self, **kwargs):
        if "circuit" in kwargs:
            self.options["integrated_circuit"]=True
        else:
            self.options["integrated_circuit"]=False
        if "normalise" in kwargs:
            self.options["normalise"]=kwargs["normalise"]
        else:
            self.options["normalise"]=False
        if "fitting" in kwargs:
            self.options["fitting"]=kwargs["fitting"]
        else:
            self.options["fitting"]=False
        if "test" in kwargs:
            self.options["test"]=kwargs["test"]
        else:
            self.options["test"]=False
        if "construct_netlist" in kwargs:
            self.options["construct_netlist"]=kwargs["construct_netlist"]
        else:
            self.options["construct_netlist"]=False
        if "data_representation" in kwargs:
            self.options["data_representation"]=kwargs["data_representation"]
        else:
            self.options["data_representation"]="nyquist"
        if "invert_imaginary" in kwargs:
            self.options["invert_imaginary"]=kwargs["invert_imaginary"]
        else:
            self.options["invert_imaginary"]=False
        if "construction_elements" not in kwargs:
            self.options["construction_elements"]=[ "R", "C", "W_inf", "CPE"]
        else:
            self.options["construction_elements"]=kwargs["construction_elements"]
