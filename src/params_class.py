#!/usr/bin/env python
import math
import warnings
import copy
import re
from numpy import multiply, divide
class params:
    def __init__(self,param_dict, multi_flag=False):
        SWV_set={"deltaE", "sampling_factor", "SW_amplitude", "scan_increment"}
        if len(set(param_dict.keys()).intersection(SWV_set))>0:
            warnings.warn("Using square-wave nondimensionalisation")
            self.sw_class=SW_params(param_dict)
            self.nd_param_dict=self.sw_class.nd_param_dict
            self.c_E0=self.sw_class.c_E0
            self.c_T0=self.sw_class.c_T0
            self.c_I0=self.sw_class.c_I0
        else:
            self.param_dict=copy.deepcopy(param_dict)
            self.T=(273+25)
            self.F=96485.3328959
            self.R=8.314459848
            self.c_E0=(self.R*self.T)/self.F
            self.c_Gamma=self.param_dict["original_gamma"]
            if "v" not in self.param_dict or "original_omega" in self.param_dict:
                if "v" in self.param_dict and "original_omega" in self.param_dict:
                    warnings.warn("Both FTACV/DCV and PSV detected - NDing by omega")

                self.param_dict["v"]=self.c_E0*self.param_dict["original_omega"]
            self.c_T0=abs(self.c_E0/self.param_dict["v"])
            self.c_I0=(self.F*self.param_dict["area"]*self.c_Gamma)/self.c_T0
            self.method_switch={
                                'e_0':self.e0,
                                'cdl':self.cdl,
                                'e_start' :self.estart,
                                'e_reverse': self.erev,
                                'omega':self.omega_d,
                                'd_e' :self.de,
                                'ru':self.ru,
                                'gamma':self.Gamma,
                                'sampling_freq':self.sf,
                                "dcv_sep":self.d_sep
                                }
            keys=sorted(param_dict.keys())
            
            k_p=re.compile("^k(?:0|_0)?_[0-9]*(?:_scale)?$")
            if multi_flag==True:
                e_match=re.compile("^E(?:0|_0|0_mean|0_std)_[0-9]*$")
            for i in range(0, len(keys)):
                if keys[i].lower() in self.method_switch:
                    self.non_dimensionalise(keys[i], param_dict[keys[i]])
                elif k_p.match(keys[i])!=None or keys[i]=="k0_scale":
                    
                    self.generic_k(keys[i], param_dict[keys[i]])
                elif multi_flag==True and e_match.match(keys[i])!=None:
                    
                    self.generic_e(keys[i], param_dict[keys[i]])

                
            self.nd_param_dict=self.param_dict


    def non_dimensionalise(self, name,name_value):
        function = self.method_switch[name.lower()]
        function(name_value, 'non_dim')
    def re_dimensionalise(self,name, name_value):
        if name.lower() in self.method_switch:
                function = self.method_switch[name.lower()]
                function(name_value, 're_dim')
        else:
            raise ValueError(name + " not in param list!")
    def generic_k(self, key, value):
        self.param_dict[key]=value*self.c_T0
    def generic_e(self, key, value):
        self.param_dict[key]=value/self.c_E0
    def e0(self, value, flag):
        if flag=='re_dim':
            self.param_dict["E_0"]=value*self.c_E0
            if "E0_std" in self.param_dict:
                self.param_dict["E0_std"]=self.param_dict["E0_std"]*self.c_E0
                self.param_dict["E0_mean"]=self.param_dict["E0_mean"]*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["E_0"]=value/self.c_E0
            if "E0_std" in self.param_dict:
                self.param_dict["E0_std"]=self.param_dict["E0_std"]/self.c_E0
                self.param_dict["E0_mean"]=self.param_dict["E0_mean"]/self.c_E0
    def k0(self, value, flag):
        if flag=='re_dim':
            self.param_dict["k_0"]=value/self.c_T0
            if "k0_scale" in self.param_dict:
                self.param_dict["k0_scale"]=self.param_dict["k0_scale"]/self.c_T0
        elif flag == 'non_dim':
            self.param_dict["k_0"]=value*self.c_T0
            if "k0_scale" in self.param_dict:
                self.param_dict["k0_scale"]=self.param_dict["k0_scale"]*self.c_T0
    def cdl(self, value, flag):
        if flag=='re_dim':
            self.param_dict["Cdl"]=value*self.c_I0*self.c_T0/(self.param_dict["area"]*self.c_E0)
        elif flag == 'non_dim':
            self.param_dict["Cdl"]=value/self.c_I0/self.c_T0*(self.param_dict["area"]*self.c_E0)
    def estart(self, value, flag):
        if flag=='re_dim':
            self.param_dict["E_start"]=value*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["E_start"]=value/self.c_E0
    def erev(self, value, flag):
        if flag=='re_dim':
            self.param_dict["E_reverse"]=value*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["E_reverse"]=value/self.c_E0
    def omega_d(self, value, flag):

        if flag=='re_dim':
            self.param_dict["omega"]=value/(2*math.pi*self.c_T0)
        elif flag == 'non_dim':
            self.param_dict["nd_omega"]=value*(2*math.pi*self.c_T0)
            self.param_dict["freq_array"]=multiply(self.param_dict["freq_array"], 2*math.pi*self.c_T0)
    def de(self, value, flag):
        if flag=='re_dim':
            self.param_dict["d_E"]=value*self.c_E0
        elif flag == 'non_dim':
            self.param_dict["d_E"]=value/self.c_E0
            self.param_dict["amp_array"]=divide(self.param_dict["amp_array"], self.c_E0)
    def ru(self, value, flag):
        if flag=='re_dim':
            self.param_dict["Ru"]=value*self.c_E0/self.c_I0
        elif flag == 'non_dim':
            self.param_dict["Ru"]=value/self.c_E0*self.c_I0
    def Gamma(self, value, flag):
        if flag=='re_dim':
            self.param_dict["gamma"]=value*self.c_Gamma
        elif flag == 'non_dim':
            self.param_dict["gamma"]=value/self.c_Gamma
    def sf(self, value, flag):

        if flag=='re_dim':
            self.param_dict["sampling_freq"]=value/((2*math.pi)/self.param_dict["nd_omega"])
        elif flag == 'non_dim':

            if self.param_dict["omega"]==0:
                #print("Hi")
                self.param_dict["sampling_freq"]=value
            else:
                self.param_dict["sampling_freq"]=value*((2*math.pi)/self.param_dict["nd_omega"])
    def d_sep(self, value, flag):
        if flag == 'non_dim':
            self.param_dict["dcv_sep"]=value/self.c_E0
class SW_params:
    def __init__(self,param_dict):
        self.param_dict=copy.deepcopy(param_dict)
        self.nd_param_dict=copy.deepcopy(param_dict)
        self.T=(273+25)
        self.F=96485.3328959
        self.R=8.314459848
        self.c_E0=(self.R*self.T)/self.F
        self.c_Gamma=self.param_dict["original_gamma"]
        self.c_I0=(self.F*self.param_dict["area"]*self.c_Gamma*self.param_dict["omega"])
        self.c_T0=1/self.param_dict["omega"]
        self.nd_param_dict["k_0"]=self.param_dict["k_0"]/self.param_dict["omega"]
        if "Ru" in param_dict.keys():
            self.nd_param_dict["Ru"]=self.param_dict["Ru"]/self.c_E0*self.c_I0
        if "Cdl" in param_dict.keys():
            self.nd_param_dict["Cdl"]=self.param_dict["Cdl"]/self.c_I0/self.c_T0*(self.param_dict["area"]*self.c_E0)
        if "gamma" in param_dict.keys():
            self.nd_param_dict["gamma"]=self.param_dict["gamma"]/self.c_Gamma
        #self.nd_param_dict["E_0"]=self.nd_param_dict["E_0"]/self.c_E0
        #self.nd_param_dict["E_start"]=self.nd_param_dict["E_start"]/self.c_E0
        #self.nd_param_dict["scan_increment"]=self.nd_param_dict["scan_increment"]/self.c_E0
        #self.nd_param_dict["deltaE"]=self.nd_param_dict["deltaE"]/self.c_E0
        #self.nd_param_dict["SW_amplitude"]=self.nd_param_dict["SW_amplitude"]/self.c_E0
