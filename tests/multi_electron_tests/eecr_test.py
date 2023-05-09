import numpy as np
import re
elements=["AoRo", "AoRr", "AiRo", "AiRr", "ArRo", "ArRr"]
linked_list=[   
            {"AoRr":{"type":"BV_red", "group":None},"ArRo" :{"type":"Cat", "group":None}},
            {"AoRo":{"type":"BV_ox", "group":None} ,"AiRo":{"type":"Cat", "group":None}, "ArRr":{"type":"Cat", "group":None}},
            {"AoRr":{"type":"Cat", "group":None},"AiRr":{"type":"BV_red", "group":None}},
            {"AiRo":{"type":"BV_ox", "group":None},"ArRo":{"type":"Cat", "group":None}},
            {"AoRo":{"type":"Cat", "group":None},"AiRr":{"type":"Cat", "group":None},"ArRr":{"type":"BV_red", "group":None}},
            {"ArRo":{"type":"BV_ox", "group":None},"AoRr":{"type":"Cat", "group":None}},
            ]
linked_list_with_groups=[   
            {"AoRr":{"type":"BV_red", "group":"a"},"ArRo" :{"type":"Cat", "group":None}},
            {"AoRo":{"type":"BV_ox", "group":"a"} ,"AiRo":{"type":"Cat", "group":None}, "ArRr":{"type":"Cat", "group":None}},
            {"AoRr":{"type":"Cat", "group":None},"AiRr":{"type":"BV_red", "group":"a"}},
            {"AiRo":{"type":"BV_ox", "group":"a"},"ArRo":{"type":"Cat", "group":None}},
            {"AoRo":{"type":"Cat", "group":None},"AiRr":{"type":"Cat", "group":None},"ArRr":{"type":"BV_red", "group":"a"}},
            {"ArRo":{"type":"BV_ox", "group":"a"},"AoRr":{"type":"Cat", "group":None}},
            ]
mode="model_writing"
mode="model_construction"
stoich_len=len(elements)-1
indexes=dict(zip(elements, (range(0, len(elements)))))
subtracted_elem="ArRr"
neg_link_dict=dict(zip(elements, linked_list))
reaction_set_sorted=[]
all_mechanisms={}
farad_params=[]
BV_counter=0
cat_counter=0
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
                print(group_dict)
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
        if mode=="model_writing":
            stochiometry_array[i][i]+="-"+all_mechanisms[ref_key] #only the subtracted species
        elif mode=="model_construction":
            if "k0" not in all_mechanisms[ref_key]:
                print(all_mechanisms[ref_key], cat_counter)
                temp_class=k_functions(all_mechanisms[ref_key], cat_counter, "-")
            else:
                temp_class=k_functions(all_mechanisms[ref_key], BV_counter, "-")
            stochiometry_array[i][i].append(temp_class.function_handle)



        

print(farad_params)
if mode=="model_writing":
    addition_vector=[[""] for x in range(0, stoich_len)]
    charge_vector=[[""] for x in range(0, stoich_len)]
    charge_vector_add=""
elif mode=="model_construction": 
    addition_vector=[[] for x in range(0, stoich_len)]
    charge_vector=[[] for x in range(0, stoich_len)]
    charge_vector_add=[]
for key in all_mechanisms:
    
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
zero_func=lambda params, E, IR:0

if mode=="model_construction":
   
   
    param_vals=np.ones(len(farad_params))
    params=dict(zip(farad_params, param_vals))
    for key in params.keys():
        if "alpha" in key:
            params[key]=0.5
        
    IR=0.1
    E=1
    vals=np.zeros((stoich_len, stoich_len))
    charge_vals=np.zeros((stoich_len))
    add_vals=np.zeros(stoich_len)
    for i in range(0, stoich_len):
        #print(stochiometry_array[i])
        for j in range(0, stoich_len):
            if stochiometry_array[i][j]!=[]:
                vals[i][j]=np.sum([x(params, E, IR) for x in stochiometry_array[i][j]])
        if charge_vector[i] !=[]:   
            charge_vals[i]=np.sum([x(params, E, IR) for x in charge_vector[i]])
        if addition_vector[i] !=[]:
            add_vals[i]=np.sum([x(params, E, IR) for x in addition_vector[i]])
    print(vals)
    print(charge_vals)
    print(add_vals)
    
