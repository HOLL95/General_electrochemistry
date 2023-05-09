import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
import re
import copy
import math
from params_class import params
class k_functions:
    def __init__(self, key, number, sign):
        
        
        if "k_" not in key:
            self.keys=["{0}_{1}".format(x, number) for x in ["k0", "E0", "alpha"]]
            if "ox" in key:
                if sign=="-":
                    self.function_handle=self.oxidation_func_neg
                elif sign =="+":
                    self.function_handle=self.oxidation_func
                
            elif "red" in key:
                if sign=="-":
                    self.function_handle=self.reduction_func_neg
                elif sign =="+":
                    self.function_handle=self.reduction_func
        else:
            self.k="k_{0}".format(number)
            if sign=="-":
                self.function_handle=self.catalytic_func_neg
            elif sign =="+":
                self.function_handle=self.catalytic_func
    def reduction_func(self, params, E, IR):
        keys=self.keys
        return params[keys[0]]*np.exp((-params[keys[2]])*(E-params[keys[1]]-IR))
    def oxidation_func(self,params, E, IR):
        keys=self.keys
        return params[keys[0]]*np.exp((1-params[keys[2]])*(E-params[keys[1]]-IR))
    def catalytic_func(self, params, E, IR):
        return params[self.k]
    def reduction_func_neg(self, params, E, IR):
        keys=self.keys
        return -params[keys[0]]*np.exp((-params[keys[2]])*(E-params[keys[1]]-IR))
    def oxidation_func_neg(self,params, E, IR):
        keys=self.keys
        return -params[keys[0]]*np.exp((1-params[keys[2]])*(E-params[keys[1]]-IR))
    def catalytic_func_neg(self, params, E, IR):
        return -params[self.k]
class square_scheme(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "linkage_dict" not in self.simulation_options:
            raise ValueError("Need to define a linkage dictionary")
        self.square_scheme=self.simulation_options["linkage_dict"]
        print(self.square_scheme)
        for key in self.square_scheme.keys():
            for second_key in self.square_scheme[key].keys():
                if "group" not in self.square_scheme[key][second_key]:
                    self.square_scheme[key][second_key]["group"]=None
                if "type" not in self.square_scheme[key][second_key]:
                    raise ValueError("Need to define reaction type for {0}->{1}".format(key, second_key))
        if "subtracted_species" not in self.simulation_options:
            raise ValueError("Need a subtracted species in simulation_options")
        element_list=list(self.square_scheme.keys())
        if element_list[-1]!=self.simulation_options["subtracted_species"]:
            element_list[-2], element_list[-1]=element_list[-1], element_list[-2]
        farad_params=self.create_model(element_list, self.simulation_options["subtracted_species"], self.square_scheme, "model_writing")
        self.farad_params=farad_params
        self.K, self.dedt, self.c, self.dedt_add=self.create_model(element_list, self.simulation_options["subtracted_species"], self.square_scheme, "model_construction")
        for i in range(0, len(farad_params)):
            self.dim_dict[farad_params[i]]=0
            if "k" in farad_params[i]:
                self.param_bounds[farad_params[i]]=self.param_bounds["k_0"]
            elif "E0" in farad_params[i]:
                self.param_bounds[farad_params[i]]=self.param_bounds["E_0"]
            elif "alpha" in farad_params[i]:
                self.param_bounds[farad_params[i]]=self.param_bounds["alpha"]
    def create_model(self, elements,subtracted_elem,square_scheme, mode):
        stoich_len=len(elements)-1
        self.stoich_len=stoich_len
        indexes=dict(zip(elements, (range(0, len(elements)))))      
        neg_link_dict=square_scheme
        reaction_set_sorted=[]
        all_mechanisms={}
        farad_params=[]
        BV_counter=0
        cat_counter=0
        if mode=="model_writing":
            stochiometry_array=[["" for i in range(0, stoich_len)] for j in range(0, stoich_len)]
        elif mode=="model_construction":
            stochiometry_array=[[[] for i in range(0, stoich_len)] for j in range(0, stoich_len)]
        oxred={"ox":"red", "red":"ox"}
        plusminus={"+":"-", "-":"+"}
        group_dict={}
        for i in range(0, len(elements)):
            key=elements[i]
            current_keys=list(neg_link_dict[key].keys())
            for j in range(0, len(current_keys)):
                negative_key=current_keys[j]
                reaction_mechanism=neg_link_dict[key][negative_key]["type"]
                counterpart= neg_link_dict[negative_key][key]["type"]
                
                if "BV" in reaction_mechanism:
                    end =reaction_mechanism[reaction_mechanism.index("_")+1:]
                    counter_end=counterpart[counterpart.index("_")+1:]
                    if oxred[end]!=counter_end:
                        raise ValueError("Unsupported reaction - {0}->{2}:{1}, {2}->{0}:{3}".format(key,reaction_mechanism, negative_key, couterpart))
                    custom_key=key+"->"+negative_key+" (BV)"
                    ref_key=custom_key[:-1]+" "+end+")"
                    if neg_link_dict[key][negative_key]["group"]==None:
                        group_flag=False 
                    elif neg_link_dict[key][negative_key]["group"] in group_dict:
                        custom_key=group_dict[neg_link_dict[key][negative_key]["group"]]["name"]
                        group_flag="repeat"
                    else:
                        if neg_link_dict[key][negative_key]["group"] != neg_link_dict[negative_key][key]["group"]:
                            raise ValueError("Unsupported grouping - {0}->{2}:{1}, {2}->{0}:{3}".format(key,neg_link_dict[key][negative_key]["group"] , negative_key, neg_link_dict[negative_key][key]["group"]))
                        custom_key=key+"->"+negative_key+" (BV)"
                        group_dict[neg_link_dict[key][negative_key]["group"]]={"name":custom_key}
                        group_flag="first"
                    anagram=sorted(custom_key)
                    if anagram not in reaction_set_sorted:
                        BV_counter+=1
                        f_params=["{0}_{1}".format(f_param, BV_counter) for f_param in ["k0", "E0", "alpha"]]
                        farad_params+=f_params
                        reaction_set_sorted.append(anagram)
                    if group_flag==False:
                        all_mechanisms[ref_key]=f_params[0]+"_"+end
                    
                    elif group_flag=="first":
                        all_mechanisms[ref_key]=f_params[0]+"_"+end
                        group_dict[neg_link_dict[key][negative_key]["group"]]["rate_constant"]=f_params[0]+"_"+end
                    elif group_flag=="repeat":
                        rate_constant=group_dict[neg_link_dict[key][negative_key]["group"]]["rate_constant"]
                        try:
                            rate_constant_name=re.findall("k0_[0-9]+", rate_constant)[0]
                        except:
                            raise ValueError("Check you are grouped with only electron-hopping reactions")
                        all_mechanisms[ref_key]=rate_constant_name+"_"+end

                elif neg_link_dict[negative_key][key]["type"]!=reaction_mechanism:
                
                    raise ValueError("Unsupported reaction - {0}->{2}:{1}, {2}->{0}:{3}".format(key,reaction_mechanism, negative_key, couterpart))
                elif neg_link_dict[negative_key][key]["type"]=="Cat":
                    custom_key=key+"->"+negative_key+" (Cat)"
                    ref_key=custom_key
                    if neg_link_dict[key][negative_key]["group"]==None:
                        cat_counter+=1
                        farad_params+=["k_{0}".format(cat_counter)]
                        all_mechanisms[ref_key]=farad_params[-1]
                    elif neg_link_dict[key][negative_key]["group"] in group_dict:
                        rate_constant=group_dict[neg_link_dict[key][negative_key]["group"]]["rate_constant"]
                        if "k0" in rate_constant:
                            raise ValueError("Catalytic rate cannot be identical to BV rate ({0}->{1})".format(key, negative_key))
                        all_mechanisms[ref_key]=rate_constant
                    else:

                        cat_counter+=1
                        k="k_{0}".format(cat_counter)
                        farad_params+=[k]
                        group_dict[neg_link_dict[key][negative_key]["group"]]={"rate_constant":k}
                        all_mechanisms[ref_key]=k
                if elements[i]==subtracted_elem:
                    continue
                else:
                    mat_idx=indexes[elements[i]]
                if mode=="model_writing":
                    stochiometry_array[i][i]+="-"+all_mechanisms[ref_key] #only the subtracted species
                elif mode=="model_construction":
                    if "k0" not in all_mechanisms[ref_key]:
                        temp_class=k_functions(all_mechanisms[ref_key], cat_counter, "-")
                    else:
                        temp_class=k_functions(all_mechanisms[ref_key], BV_counter, "-")
                    stochiometry_array[i][i].append(temp_class.function_handle)
                if mode=="model_writing":
                    addition_vector=[[""] for x in range(0, stoich_len)]
                    charge_vector=[[""] for x in range(0, stoich_len)]
                    charge_vector_add=""
                elif mode=="model_construction": 
                    addition_vector=[[] for x in range(0, stoich_len)]
                    charge_vector=[[] for x in range(0, stoich_len)]
                    charge_vector_add=[]
        self.all_mechanisms=all_mechanisms
        for key in all_mechanisms.keys():
            formed_species=re.findall("(?<=->).*?(?=\s\()", key)[0]
            subtracted_species=re.findall(".+?(?=->)", key)[0]
            row_idx=indexes[formed_species]
            col_idx=indexes[subtracted_species]
            rate=all_mechanisms[key]
            if "k0" not in all_mechanisms[key]:    
                num=rate[rate.index("_")+1:]
            else:
                num=re.findall("(?<=k0_)[0-9]+", rate)[0]
            if subtracted_elem not in key:
                if mode=="model_writing":
                    stochiometry_array[row_idx][col_idx]+="+"+all_mechanisms[key] #only adding
                elif mode=="model_construction":
                    temp_class=k_functions(rate, num, "+")
                    stochiometry_array[row_idx][col_idx].append(temp_class.function_handle) #only adding
            else:
                if subtracted_species==subtracted_elem:
                
                    for i in range(0, stoich_len):
                        if mode=="model_writing":
                            stochiometry_array[row_idx][i]+="-"+all_mechanisms[key]
                        elif mode=="model_construction":
                            temp_class=k_functions(rate, num, "-")
                            stochiometry_array[row_idx][i].append(temp_class.function_handle)
                    if mode=="model_writing":
                        addition_vector[row_idx][0]+=all_mechanisms[key]
                    elif mode=="model_construction":
                        temp_class=k_functions(rate, num, "+")
                        addition_vector[row_idx].append(temp_class.function_handle)
            if "BV" in key:
                if "ox" in all_mechanisms[key]:
                    if mode=="model_writing":
                        rate_constant="+"+all_mechanisms[key]
                    elif mode=="model_construction":
                        temp_class=k_functions(rate, num, "+")
                        rate_constant=temp_class.function_handle
                        sign="+"
                elif "red" in all_mechanisms[key]:
                    if mode=="model_writing":
                        rate_constant="-"+all_mechanisms[key]
                    elif mode=="model_construction":
                        temp_class=k_functions(rate, num, "-")
                        rate_constant=temp_class.function_handle
                        sign="-"
                    else:            
                        raise ValueError("BV needs to be red or ox ({0})".format(all_mechanisms[key]))
                if subtracted_elem not in subtracted_species:
                    if mode=="model_writing":
                        charge_vector[col_idx][0]+=rate_constant
                    elif mode=="model_construction":
                        
                        charge_vector[col_idx].append(rate_constant)
                elif subtracted_species==subtracted_elem:
                    if mode=="model_writing":
                        charge_vector_add+=rate_constant
                        rate_constant=list(rate_constant)
                        rate_constant[0]=plusminus[rate_constant[0]]
                        rate_constant=("").join(rate_constant)
                        for i in range(0, stoich_len):
                            charge_vector[i][0]+=rate_constant
                    elif mode=="model_construction":
                        if sign=="+":
                            temp_class=k_functions(rate, num, "-")
                        elif sign=="-":
                            temp_class=k_functions(rate, num, "+")
                        charge_vector_add.append(rate_constant)
                        for i in range(0, stoich_len):
                            charge_vector[i].append(temp_class.function_handle)
        if mode=="model_writing":
            print("all_reactions")
            print("-----------")
            for key in all_mechanisms:
                print(key, ":", all_mechanisms[key])
            print("K")
            print("-----------")
            for line in stochiometry_array:
                for i in range(0, len(line)):
                    if line[i]=="":
                        line[i]="0"
                print(line)
            print("c")
            print("-----------")
            for line in addition_vector:
                for i in range(0, len(line)):
                    if line[i]=="":
                        line[i]="0"
                print(line[0])
            print("dedt")
            print("-----------")
            for i in range(0, stoich_len):
                print(charge_vector[i],)
            print("+")
            print("-----------")
            print(charge_vector_add)
            return farad_params
        elif mode=="model_construction":
            return stochiometry_array, charge_vector, addition_vector, charge_vector_add
    def print_parameter_dict(self,):
        mechanism_dict=self.all_mechanisms
        farad_params=self.farad_params
        print_dict=dict(zip(farad_params, [[] for x in range(0, len(farad_params))]))
       
        for key in mechanism_dict.keys():
            write_key=key[:key.index("(")-1]
            loc=mechanism_dict[key]
            
            if "k0" in loc:
                num=re.findall("(?<=k0_)[0-9]+", loc)[0]
                print_dict["k0_{0}".format(num)].append(write_key)
                print_dict["E0_{0}".format(num)].append(write_key)
                print_dict["alpha_{0}".format(num)].append(write_key)
            else:
                print_dict[loc].append(write_key)
        print(r"{")
        for key in print_dict.keys():
            print(r"'{0}':None, #{1}".format(key, (", ").join(print_dict[key])))
        print(r"}")

    def et(self, E_start, omega, phase, delta_E, t):
        E_t = (E_start + delta_E) + delta_E * math.sin((omega * t) + phase)
        return E_t

    def dEdt(self, omega, phase, delta_E, t):
        dedt = delta_E * omega * math.cos(omega * t + phase)
        return dedt

    def c_et(self, E_start, E_reverse, tr, omega, phase, v, delta_E, t):
        if t < tr:
            E_dc = E_start + (v * t)
        else:
            E_dc = E_reverse - (v * (t - tr))

        E_t = E_dc + (delta_E * math.sin((omega * t) + phase))
        return E_t
    def c_dEdt(self, tr, omega, phase, v, delta_E, t):
        if t < tr:
            E_dc = v
        else:
            E_dc = -v

        dedt = E_dc + (delta_E * omega * math.cos(omega * t + phase))
        return dedt
    def dcv_et(self, E_start, E_reverse, tr, v, t):
        if t < tr:
            E_dc = E_start + (v * t)
        else:
            E_dc = E_reverse - (v * (t - tr))

        return E_dc
    def dcv_dEdt(self, tr, v, t):
        if t < tr:
            E_dc = v
        else:
            E_dc = -v

        dedt = E_dc
        return dedt
    def voltage_query(self, time):
        if self.simulation_options["method"]=="sinusoidal":
                Et=self.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
                dEdt=self.dEdt(self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
        elif self.simulation_options["method"]=="ramped":
                Et=self.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
                dEdt=self.c_dEdt(self.nd_param.nd_param_dict["tr"] ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
        elif self.simulation_options["method"]=="dcv":
                Et=self.dcv_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) , 1,time)
                dEdt=self.dcv_dEdt(self.nd_param.nd_param_dict["tr"],1,time)
        elif self.simulation_options["method"]=="sum_of_sinusoids":
                Et=self.sum_of_sinusoids_E(self.nd_param.nd_param_dict["amp_array"],self.nd_param.nd_param_dict["freq_array"],
                                                                    self.nd_param.nd_param_dict["phase_array"],self.nd_param.nd_param_dict["num_frequencies"], time)
                dEdt=self.sum_of_sinusoids_dE(self.nd_param.nd_param_dict["amp_array"],self.nd_param.nd_param_dict["freq_array"],
                                                                    self.nd_param.nd_param_dict["phase_array"],self.nd_param.nd_param_dict["num_frequencies"], time)
        return Et, dEdt
    def define_voltages(self, **kwargs):
        if "transient" not in kwargs:
            transient=False
        else:
            transient=kwargs["transient"]
        voltages=np.zeros(len(self.time_vec))
        if self.simulation_options["method"]=="sinusoidal":
            for i in range(0, len(self.time_vec)):
                voltages[i]=self.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], (self.time_vec[i]))
        elif self.simulation_options["method"]=="ramped":
            for i in range(0, len(self.time_vec)):
                voltages[i]=self.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],(self.time_vec[i]))
        elif self.simulation_options["method"]=="dcv":
            for i in range(0, len(self.time_vec)):
                voltages[i]=self.dcv_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) , 1,(self.time_vec[i]))
        return voltages
    def simulate(self, parameters, times):
        if isinstance(parameters, dict):
            parameters=[parameters[key] for key in self.optim_list]
        if len(parameters)!= len(self.optim_list):
            print(self.optim_list)
            print(parameters)
            raise ValueError('Wrong number of parameters')
        if self.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(parameters, "un_norm")
        else:
            normed_params=copy.deepcopy(parameters)
        for i in range(0, len(self.optim_list)):
            self.dim_dict[self.optim_list[i]]=normed_params[i]  
        self.nd_param=params(self.dim_dict, multi_flag=True)
        E, dEdt=self.voltage_query(0)
        Cdl, CdlE, CdlE2, CdlE3=[self.nd_param.nd_param_dict[x] for x in ["Cdl", "CdlE1", "CdlE2", "CdlE3"]]
        self.Cdlp= Cdl * (1.0 + CdlE * E + CdlE2 * E**2 + CdlE3 * E**3)
        current=np.zeros(len(self.time_vec))
        current[0]=self.Cdlp
        identity=np.identity(self.stoich_len)
        dt=self.time_vec[1]-self.time_vec[0]
        K_mat=np.zeros((self.stoich_len, self.stoich_len))
        addition_vector=np.zeros((self.stoich_len, 1))
        current_vector=np.zeros((self.stoich_len, 1))
        u0=np.zeros((self.stoich_len, 1))
        u0[0]=1
        Ru=self.nd_param.nd_param_dict["Ru"]
        length=len(self.time_vec)
        print(length)
        interval=30
        load=length//interval
        thetas=np.zeros((self.stoich_len, len(self.time_vec)))
        thetas[:,0]=u0.T
        koxs=np.zeros((self.stoich_len, len(self.time_vec)))
        kreds=np.zeros((self.stoich_len, len(self.time_vec)))
        nd_params=self.nd_param.nd_param_dict
        for i in range(1, len(self.time_vec)):
           
            I=current[i-1]
            ndparams=self.nd_param.nd_param_dict
            IR=I*Ru
            t=self.time_vec[i]
            Er=E-IR
            E, dEdt=self.voltage_query(t)
            """for m in range(0, self.stoich_len):
                keys=["{0}_{1}".format(x, m+1) for x in ["k0", "E0", "alpha"]]
                koxs[m, i]=nd_params[keys[0]]*np.exp((-nd_params[keys[2]])*(E-nd_params[keys[1]]-IR))
                kreds[m, i]=nd_params[keys[0]]*np.exp((1-nd_params[keys[2]])*(E-nd_params[keys[1]]-IR))"""
            for j in range(0, self.stoich_len):
                for k in range(0, self.stoich_len):
                    if self.K[j][k]!=[]:
                        K_mat[j, k]=np.sum([x(self.nd_param.nd_param_dict, E, IR) for x in self.K[j][k]])
                if self.c[j]!=[]:
                    addition_vector[j,0]=np.sum([x(self.nd_param.nd_param_dict, E, IR) for x in self.c[j]])
                if self.dedt[j]!=[]:
                    current_vector[j,0]=np.sum([x(self.nd_param.nd_param_dict, E, IR) for x in self.dedt[j]])
            LHSQ, LHSR=np.linalg.qr(identity-(dt*K_mat))
            RHS=np.array(u0+(dt*addition_vector))
            Qb=np.matmul(LHSQ.T, RHS)
            u1=np.linalg.solve(LHSR, Qb)

            dedt=np.matmul(np.transpose(current_vector)[0], u1)
            dedt+=np.sum([x(self.nd_param.nd_param_dict, E, IR) for x in self.dedt_add])
            #thetas[:,i]=u1.T
            u0=u1
            Cdlp= Cdl * (1.0 + CdlE * Er + CdlE2 * Er**2 + CdlE3 * Er**3)
            In1 = Cdlp * (dEdt + Ru * I / dt)
            In1+=self.nd_param.nd_param_dict["gamma"]*dedt
            In1/=(1+Cdlp*Ru/dt)
            current[i]=In1
        return self.i_nondim(current)
        


   