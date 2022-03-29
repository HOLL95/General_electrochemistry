import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import collections
import random
import warnings
from collections import deque


#TODO->paralell in series+ gerischer
class EIS:
    def __init__(self,  **kwargs):
        self.options={}
        self.options_checker(**kwargs)
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

        if self.options["integrated_circuit"]==True:
            circuit=copy.deepcopy(kwargs["circuit"])
            self.circuit=self.construct_circuit(circuit, func=self.define_z_func)
        self.all_nodes=["paralell", "series",]#"element"]
        self.all_elems=[ "R", "C", "W_inf", "W_fin", "CPE"]
        self.num_nodes=len(self.all_nodes)-1
        self.num_elems=len(self.all_elems)-1

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
    def flatten(self, d, parent_key='', sep='-'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
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
        #print(param, flag)
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
    def z_functions(self, circuit_type,param, paralell):
        if circuit_type=="capacitor":
            if paralell==False:
                def F(**kwargs):
                 return 1/1j*kwargs["omega"]*kwargs[param]
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
                    return 1/(1/kwargs[param[0]])*np.tanh(B*np.sqrt(1j*kwargs["omega"]))
        elif circuit_type=="CPE":
            if paralell==False:
                def F(**kwargs):
                    #print(kwargs[param[0]]*np.power(kwargs["omega"]*1j, kwargs[param[1]]))
                    return 1/(kwargs[param[0]]*np.power(kwargs["omega"]*1j, kwargs[param[1]]))
            else:
                def F(**kwargs):
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
        #print(parameters, self.param_names)
        sim_params=dict(zip(self.param_names, parameters))
        if self.options["normalise"]==True:
            normed_params=self.change_norm_group(sim_params, "un_norm")
        else:
            normed_params=sim_params
        #print(normed_params)
        spectra=np.zeros(len(frequencies), dtype="complex")
        for i in range(0, len(frequencies)):

            normed_params["omega"]=frequencies[i]
            spectra[i]=self.freq_simulate(**normed_params)
        if self.options["test"]==True:
            plt.plot(np.real(spectra), -np.imag(spectra))
            plt.show()

        return np.column_stack((np.real(spectra), np.imag(spectra)))

    def test_vals(self, parameters, frequencies):
        normalise=self.options["normalise"]
        self.options["normalise"]=False
        if isinstance(parameters, list):
            list_params=parameters
        elif isinstance(parameters, dict):
            list_params=[parameters[x] for x in self.param_names]
        print(list_params)
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
