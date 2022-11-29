import isolver_martin_brent
import SWV_surface
#import isolver_martin_NR
from scipy.stats import norm, lognorm
import math
import numpy as np
import itertools
from params_class import params
#from pybamm_solve import pybamm_solver
from dispersion_class import dispersion
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
#from sklearn import linear_model, datasets
from decimal import Decimal
from scipy.optimize import curve_fit
from scipy_solver_class import scipy_funcs
from scipy.stats import pearsonr
import multiprocessing
import copy
import time
import warnings
import re
import matplotlib.pyplot as plt
class single_electron:
    def __init__(self,file_name="", dim_parameter_dictionary={}, simulation_options={}, other_values={}, param_bounds={}, results_flag=True):
        if type(file_name) is dict:
            raise TypeError("Need to define a filename - this is currently a dictionary!")

        if len(dim_parameter_dictionary)==0 and len(simulation_options)==0 and len(other_values)==0:
            self.file_init=True
            file=open(file_name, "rb")
            save_dict=pickle.load(file, encoding="latin1")
            dim_parameter_dictionary=save_dict["param_dict"]
            simulation_options=save_dict["simulation_opts"]
            other_values=save_dict["other_vals"]
            param_bounds=save_dict["bounds"]
            self.save_dict=save_dict
        else:
            self.file_init=False
        simulation_options=self.options_checker(simulation_options)
        dim_parameter_dictionary=self.param_checker(dim_parameter_dictionary)
        #required_params=set(["E_0", "k_0", "alpha", "gamma", "Ru", "Cdl", "CdlE1","CdlE2","CdlE3", "E_start", \
        #                    "E_reverse", "d_E"])
        #param_set=set(dim_parameter_dictionary.keys())
        #req_union=required_params.intersection(param_set)
        #if len(req_union)!=len(required_params):
        #    missing_params=required_params-req_union
        #    raise KeyError("Essential parameter(s) mising:",missing_params)
        key_list=list(dim_parameter_dictionary.keys())
        if simulation_options["method"]=="ramped" or simulation_options["method"]=="sinusoidal":
            if simulation_options["phase_only"]==False and "cap_phase" not in key_list:
                warnings.warn("Capacitance phase not define - assuming unified phase values")
                simulation_options["phase_only"]=True
        if simulation_options["method"]=="ramped":
            dim_parameter_dictionary["v_nondim"]=True


        self.simulation_options=simulation_options

        self.other_values=other_values
        self.optim_list=self.simulation_options["optim_list"]
        if not other_values:
            self.harmonic_range=list(range(0, 10))
            self.filter_val=0.5
            self.bounds_val=200
        else:
            self.harmonic_range=other_values["harmonic_range"]
            self.filter_val=other_values["filter_val"]
            self.bounds_val=other_values["bounds_val"]
        self.num_harmonics=len(self.harmonic_range)
        self.nd_param=params(dim_parameter_dictionary)
        self.dim_dict=copy.deepcopy(dim_parameter_dictionary)
        self.def_optim_list(self.simulation_options["optim_list"])
        self.boundaries=None
        if "square_wave" not in self.simulation_options["method"]:
            self.calculate_times()
            frequencies=np.fft.fftfreq(len(self.time_vec), self.time_vec[1]-self.time_vec[0])
            self.frequencies=frequencies[np.where(frequencies>0)]
            last_point= (self.harmonic_range[-1]*self.nd_param.nd_param_dict["omega"])+(self.nd_param.nd_param_dict["omega"]*self.filter_val)
            self.test_frequencies=frequencies[np.where(self.frequencies<last_point)]

        else:
            if self.simulation_options["method"]=="square_wave":
                self.SW_end=(abs(self.dim_dict["deltaE"]/self.dim_dict["scan_increment"])*self.dim_dict["sampling_factor"])
                self.time_vec=np.arange(0, self.SW_end, 1)
            elif self.simulation_options["method"]=="square_wave_fourier":
                self.SW_end=abs(self.dim_dict["deltaE"]/self.dim_dict["scan_increment"])
                self.time_vec=np.arange(0, self.SW_end, 1/(self.SW_end*self.dim_dict["sampling_factor"]))
            self.frequencies=[]
            self.SW_sampling()
            ## TODO: Work out how to non-dimensionalise experimental SWV data.
        if param_bounds!={}:
            self.param_bounds=param_bounds
        if self.simulation_options["experimental_fitting"]==True:
            self.secret_data_fourier=self.top_hat_filter(other_values["experiment_current"])
            self.secret_data_time_series=other_values["experiment_current"]
        if self.simulation_options["sample_times"]!=False:
            self.simulation_options["sample_times"]=np.divide(self.simulation_options["sample_times"],self.nd_param.c_T0)
    def calculate_times(self,):
        self.dim_dict["tr"]=self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]
        if self.simulation_options["experimental_fitting"]==True:
            if self.simulation_options["method"]=="sinusoidal":
                time_end=(self.nd_param.nd_param_dict["num_peaks"]/self.nd_param.nd_param_dict["original_omega"])
            elif self.simulation_options["method"]=="ramped":
                time_end=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])*self.nd_param.c_T0
            elif self.simulation_options["method"]=="dcv":
                time_end=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])*self.nd_param.c_T0

            if self.simulation_options["no_transient"]!=False:
                if self.simulation_options["no_transient"]>time_end:
                    warnings.warn("Previous transient removal method detected")
                    time_idx=tuple(np.where(self.other_values["experiment_time"]<=time_end))
                    desired_idx=tuple((range(self.simulation_options["no_transient"],time_idx[0][-1])))
                    self.time_idx=time_idx[:-1]
                else:
                    time_idx=tuple(np.where((self.other_values["experiment_time"]<=time_end) & (self.other_values["experiment_time"]>self.simulation_options["no_transient"])))
                    desired_idx=time_idx
                    self.time_idx=time_idx[:-1]
            else:
                desired_idx=tuple(np.where(self.other_values["experiment_time"]<=time_end))
                time_idx=desired_idx
                self.time_idx=time_idx[:-1]
            if self.file_init==False or results_flag==True:
                self.time_vec=self.other_values["experiment_time"][time_idx]/self.nd_param.c_T0

                self.other_values["experiment_time"]=self.other_values["experiment_time"][desired_idx]/self.nd_param.c_T0
                self.other_values["experiment_current"]=self.other_values["experiment_current"][desired_idx]/self.nd_param.c_I0
                self.other_values["experiment_voltage"]=self.other_values["experiment_voltage"][desired_idx]/self.nd_param.c_E0

            else:
                if self.simulation_options["method"]=="sinusoidal":
                    self.nd_param.nd_param_dict["time_end"]=(self.nd_param.nd_param_dict["num_peaks"])#/self.nd_param.nd_param_dict["omega"])
                else:
                    self.nd_param.nd_param_dict["time_end"]=2*(self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"])/self.nd_param.nd_param_dict["v"]
                self.times()
        else:
            if self.simulation_options["method"]=="sinusoidal":
                self.nd_param.nd_param_dict["time_end"]=(self.nd_param.nd_param_dict["num_peaks"])#/self.nd_param.nd_param_dict["original_omega"])#DIMENSIONAL
            else:
                self.nd_param.nd_param_dict["time_end"]=(2*(self.dim_dict["E_reverse"]-self.dim_dict["E_start"])/self.dim_dict["v"])/self.nd_param.c_T0#DIMENSIONAL
            self.times()
            if self.simulation_options["no_transient"]!=False:
                    transient_time=self.t_nondim(self.time_vec)
                    time_idx=np.where((transient_time<=(self.nd_param.nd_param_dict["time_end"]*self.nd_param.c_T0)) & (transient_time>self.simulation_options["no_transient"]))
                    self.time_idx=time_idx

            else:
                    transient_time=self.t_nondim(self.time_vec)
                    desired_idx=tuple(np.where(transient_time<=(self.nd_param.nd_param_dict["time_end"]*self.nd_param.c_T0)))
                    self.time_idx=desired_idx
    def GH_setup(self):
        """
        We assume here that for n>1 normally dispersed parameters then the order of the integral
        will be the same for bothif self.simulation_options["sample_times"] is not False:
            self.downsample(self.time_vec, )
        """
        try:
            disp_idx=self.simulation_options["dispersion_distributions"].index("normal")
        except:
            raise KeyError("No normal distributions for GH quadrature")
        nodes=self.simulation_options["dispersion_bins"][disp_idx]
        labels=["nodes", "weights", "normal_weights"]
        nodes, weights=np.polynomial.hermite.hermgauss(nodes)
        normal_weights=np.multiply(1/math.sqrt(math.pi), weights)
        self.other_values["GH_dict"]=dict(zip(labels, [nodes, weights, normal_weights]))
    def define_boundaries(self, param_bounds):
        self.param_bounds=param_bounds
    def voltage_query(self, time):
        if self.simulation_options["method"]=="sinusoidal":
                Et=isolver_martin_brent.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
                dEdt=isolver_martin_brent.dEdt(self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], time)
        elif self.simulation_options["method"]=="ramped":
                Et=isolver_martin_brent.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
                dEdt=isolver_martin_brent.c_dEdt(self.nd_param.nd_param_dict["tr"] ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],time)
        elif self.simulation_options["method"]=="dcv":
                Et=isolver_martin_brent.dcv_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) , 1,time)
                dEdt=isolver_martin_brent.dcv_dEdt(self.nd_param.nd_param_dict["tr"],1,time)
        return Et, dEdt
    def current_ode_sys(self, state_vars, time):
        current, theta, potential=state_vars
        Et, dEdt=self.voltage_query(time)
        Er=Et-(self.nd_param.nd_param_dict["Ru"]*current)
        ErE0=Er-self.nd_param.nd_param_dict["E_0"]
        alpha=self.nd_param.nd_param_dict["alpha"]
        self.Cdlp=self.nd_param.nd_param_dict["Cdl"]*(1+self.nd_param.nd_param_dict["CdlE1"]*Er+self.nd_param.nd_param_dict["CdlE2"]*(Er**2)+self.nd_param.nd_param_dict["CdlE3"]*(Er**3))
        d_thetadt=((1-theta)*self.nd_param.nd_param_dict["k_0"]*np.exp((1-alpha)*ErE0))-(theta*self.nd_param.nd_param_dict["k_0"]*np.exp((-alpha)*ErE0))
        if self.Cdlp<1e-6:
            dIdt=self.nd_param.nd_param_dict["gamma"]*d_thetadt
        else:
            dIdt=(dEdt-(current/self.Cdlp)+self.nd_param.nd_param_dict["gamma"]*d_thetadt*(1/self.Cdlp))/self.nd_param.nd_param_dict["Ru"]
        f=[dIdt, d_thetadt, dEdt]
        return f
    def diff_cap(self, state_vars, time):
        current=state_vars
        if time>self.nd_param.nd_param_dict["tr"]:
            dE=-1
        else:
            dE=1
        return (dE/self.nd_param.nd_param_dict["Ru"])-(current/(self.nd_param.nd_param_dict["Ru"]*self.nd_param.nd_param_dict["Cdl"]))
    def likelihood_surfaces(self, parameters, data, **kwargs):
        if "pc" not in kwargs:
            kwargs["pc"]=0.1
        if "size" not in kwargs:
            kwargs["size"]=20
        if "scan_parameters" not in kwargs:
            desired_range=range(0, len(parameters))
        else:
            if type(kwargs["scan_parameters"]) is not list:
                raise TypeError("Parameters needs to be list not "+str(type(kwargs["scan_parameters"])))
            else:
                desired_range=[self.optim_list.index(x) for x in kwargs["scan_parameters"]]
        save_dict={}

        for i in desired_range:
            save_dict={}
            print(self.optim_list[i])
            for j in range(0, len(parameters)):
                if i==j:
                    pass
                if i>j:
                    start=time.time()
                    y_param, x_param=self.optim_list[i],self.optim_list[j]
                    y_idx, x_idx=self.optim_list.index(y_param), self.optim_list.index(x_param)
                    y_val, x_val=parameters[i], parameters[j]
                    y_list=np.linspace(y_val*(1-kwargs["pc"]), y_val*(1+kwargs["pc"]), kwargs["size"])
                    x_list=np.linspace(x_val*(1-kwargs["pc"]), x_val*(1+kwargs["pc"]), kwargs["size"])
                    XX,YY=np.meshgrid(x_list, y_list)
                    param_matrix=[[[0 for x in range(0, len(parameters))] for x in range(0, kwargs["size"])] for y in range(0, kwargs["size"])]
                    for q in range(0, kwargs["size"]):
                        for k in range(0, kwargs["size"]):
                            sim_params=copy.deepcopy(parameters)
                            sim_params[x_idx]=x_list[k]
                            sim_params[y_idx]=y_list[q]
                            param_matrix[q][k]=sim_params
                    param_list=list(itertools.chain(*param_matrix))
                    mp_argument=zip(param_list, ["fourier"]*(kwargs["size"]**2))
                    with multiprocessing.Pool(processes=4) as pool:
                        results = pool.starmap(self.test_vals, mp_argument)
                    errors=[self.RMSE(x, data) for x in results]
                    Z=[errors[i:i+kwargs["size"]] for i in range(0, len(errors), kwargs["size"])]
                    save_dict[x_param+"_"+y_param]={"X":XX, "Y":YY, "Z":Z}
                    print(x_param+"_"+y_param)
                    print(XX)
                    print(YY)
                    print(Z)
            np.save("Likelihood_surfaces_"+self.optim_list[i]+".npy", save_dict)
    def likelihood_curves(self, parameters, data, **kwargs):
        if "pc" not in kwargs:
            kwargs["pc"]=0.1
        if "size" not in kwargs:
            kwargs["size"]=20
        if "scan_parameters" not in kwargs:
            desired_range=range(0, len(parameters))
        else:
            if type(kwargs["scan_parameters"]) is not list:
                raise TypeError("Parameters needs to be list not "+str(type(kwargs["scan_parameters"])))
            else:
                desired_range=[self.optim_list.index(x) for x in kwargs["scan_parameters"]]
        save_dict={}
        for i in desired_range:
            x_param=self.optim_list[i]
            x_idx=self.optim_list.index(x_param)
            x_val=parameters[i]
            x_list=np.linspace(x_val*(1-kwargs["pc"]), x_val*(1+kwargs["pc"]), kwargs["size"])
            param_list=[[0 for x in range(0, len(parameters))] for y in range(0, kwargs["size"])]
            for q in range(0, kwargs["size"]):
                sim_params=copy.deepcopy(parameters)
                sim_params[x_idx]=x_list[q]
                param_list[q]=sim_params

            mp_argument=zip(param_list, ["fourier"]*kwargs["size"])
            with multiprocessing.Pool(processes=4) as pool:
                results = pool.starmap(self.test_vals, mp_argument)
            errors=[self.RMSE(x, data) for x in results]
            save_dict[x_param]={"X":x_list, "Y":errors}
        np.save("Likelihood_curves_high_gamma.npy", save_dict)
    def def_optim_list(self, optim_list):
        keys=list(self.dim_dict.keys())
        for i in range(0, len(optim_list)):
            if optim_list[i] in keys:
                continue
            elif "freq_" in optim_list[i] or "amp_" in optim_list[i] or "phase_" in optim_list[i]:
                continue
            else:
                raise KeyError("Parameter " + optim_list[i]+" not found in model")
        self.optim_list=optim_list
        param_boundaries=np.zeros((2, self.n_parameters()))
        check_for_bounds=vars(self)
        if "param_bounds" in list(check_for_bounds.keys()):
            param_bound_keys=self.param_bounds.keys()
            for i in range(0, self.n_parameters()):
                    if optim_list[i] in self.param_bounds:
                        param_boundaries[0][i]=self.param_bounds[self.optim_list[i]][0]
                        param_boundaries[1][i]=self.param_bounds[self.optim_list[i]][1]
                    elif "freq" in optim_list[i] or "amp" in optim_list[i] or "phase" in optim_list[i]:
                        appropriate_key="all_{0}s".format(optim_list[i][:optim_list[i].index("_")])
                        param_boundaries[0][i]=self.param_bounds[appropriate_key][0]
                        param_boundaries[1][i]=self.param_bounds[appropriate_key][1]
                    else:
                        raise ValueError("Need to define boundaries for "+optim_list[i])

            self.boundaries=param_boundaries

        disp_check_flags=["mean", "scale", "upper", "logupper"]#Must be unique for each distribution!
        disp_check=[[y in x for y in disp_check_flags] for x in self.optim_list]
        if True in [True in x for x in disp_check]:
            self.simulation_options["dispersion"]=True
            disp_flags=[["mean", "std"], ["shape","scale"], ["lower","upper"], ["mean","std", "skew"], ["logupper", "loglower"]]#Set not name must be unique
            all_disp_flags=["mean", "std", "skew", "shape", "scale", "upper", "lower", "logupper", "loglower"]
            distribution_names=["normal", "lognormal", "uniform", "skewed_normal"]
            dist_dict=dict(zip(distribution_names, disp_flags))
            disp_param_dict={}
            for i in range(0, len(self.optim_list)):
                for j in range(0, len(all_disp_flags)):
                    if all_disp_flags[j] in self.optim_list[i]:
                        try:

                            m=re.search('.+?(?=_'+all_disp_flags[j]+')', self.optim_list[i])
                            param=m.group(0)
                            if param in disp_param_dict:
                                disp_param_dict[param].append(all_disp_flags[j])
                            else:
                                disp_param_dict[param]=[all_disp_flags[j]]
                        except:
                            print(self.optim_list[i], all_disp_flags[j])
                            continue

            distribution_names=["normal", "lognormal", "uniform", "skewed_normal", "log_uniform"]
            distribution_dict=dict(zip(distribution_names, disp_flags))

            self.simulation_options["dispersion_parameters"]=list(disp_param_dict.keys())
            self.simulation_options["dispersion_distributions"]=[]
            for param in self.simulation_options["dispersion_parameters"]:
                param_set=set(disp_param_dict[param])

                for key in distribution_dict.keys():

                    if set(distribution_dict[key])==param_set:
                        self.simulation_options["dispersion_distributions"].append(key)
            if type(self.simulation_options["dispersion_bins"])is int:
                if len(self.simulation_options["dispersion_distributions"])>1:
                    num_dists=len(self.simulation_options["dispersion_distributions"])
                    warnings.warn("Only one set of bins defined for multiple distributions. Assuming all distributions discretised using the same number of bins")
                    self.simulation_options["dispersion_bins"]=[self.simulation_options["dispersion_bins"]]*num_dists
                else:
                    raise ValueError("Fewer specified bins than distributions")

                self.GH_setup()
            self.disp_class=dispersion(self.simulation_options, optim_list)
            if self.simulation_options["GH_quadrature"]==True:
                self.GH_setup()
            self.disp_class=dispersion(self.simulation_options, optim_list)
        else:
            self.simulation_options["dispersion"]=False
        if "phase" in optim_list and "cap_phase" not in optim_list:
            self.simulation_options["phase_only"]=True
        if self.simulation_options["method"]=="sum_of_sinusoids":
            if len(optim_list)==0:
                if len(self.dim_dict["freq_array"])==0:
                    raise ValueError("Need to define list of sinusoids either in optim_list or in param_list")
            if "freq_1" in self.optim_list:
                found_flag=True
                omega_count=0
                
                while found_flag==True:
                    omega_count+=1
                    if "freq_{0}".format(omega_count) not in optim_list:
                        found_flag=False
                self.dim_dict["num_frequencies"]=omega_count-1
                self.dict_of_sine_list=dict(zip(["freq_", "amp_", "phase_"],[[0]*self.dim_dict["num_frequencies"] for x in range(0, 3)]))
                sinusoid_positions=[0]*self.dim_dict["num_frequencies"]*3
                sinusoid_count=0
                for i in range(0, self.dim_dict["num_frequencies"]):
                    for key in self.dict_of_sine_list.keys():
                            optim_position=optim_list.index(key+str(i+1))
                            self.dict_of_sine_list[key][i]=optim_position
                            sinusoid_positions[sinusoid_count]=optim_position
                            sinusoid_count+=1

                self.param_positions=list(set(range(0, len(optim_list)))-set(sinusoid_positions))
    def add_noise(self, series, sd):
        return np.add(series, np.random.normal(0, sd, len(series)))
    def normalise(self, norm, boundaries):
        return  (norm-boundaries[0])/(boundaries[1]-boundaries[0])
    def un_normalise(self, norm, boundaries):
        return (norm*(boundaries[1]-boundaries[0]))+boundaries[0]
    def i_nondim(self, current):
        return np.multiply(current, self.nd_param.c_I0)
    def e_nondim(self, potential):
        return np.multiply(potential, self.nd_param.c_E0)
    def t_nondim(self, time):
        return np.multiply(time, self.nd_param.c_T0)
    def RMSE(self, y, y_data):
        return np.mean(np.sqrt(np.square(np.subtract(y, y_data))))
    def square_error(self, y, y_data):
        return np.square(np.subtract(y, y_data))
    def n_outputs(self):
        if self.simulation_options["multi_output"]==True:
            return 2
        else:
            return 1
    def n_parameters(self):
        return len(self.optim_list)
    
        
    def define_voltages(self, transient=False):
        voltages=np.zeros(len(self.time_vec))
        if self.simulation_options["method"]=="sinusoidal":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.et(self.nd_param.nd_param_dict["E_start"],self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"], (self.time_vec[i]))
        elif self.simulation_options["method"]=="ramped":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.c_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) ,self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], 1,self.nd_param.nd_param_dict["d_E"],(self.time_vec[i]))
        elif self.simulation_options["method"]=="dcv":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.dcv_et(self.nd_param.nd_param_dict["E_start"], self.nd_param.nd_param_dict["E_reverse"], (self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]) , 1,(self.time_vec[i]))
        elif self.simulation_options["method"]=="square_wave":
            #self.SW_end=(self.dim_dict["deltaE"]/self.dim_dict["scan_increment"])*self.dim_dict["sampling_factor"]
            for i in self.time_vec:
                i=int(i)
                voltages[i-1]=SWV_surface.potential(i, self.dim_dict["sampling_factor"], self.dim_dict["scan_increment"],self.dim_dict["SW_amplitude"], self.dim_dict["E_start"])
            voltages=voltages/self.nd_param.sw_class.c_E0
        elif self.simulation_options["method"]=="square_wave_fourier":
            for i in range(0, len(self.time_vec)):
                voltages[i]=SWV_surface.fourier_Et(math.pi, self.nd_param.nd_param_dict["scan_increment"], self.nd_param.nd_param_dict["fourier_order"], self.nd_param.nd_param_dict["SW_amplitude"], self.nd_param.nd_param_dict["E_start"], self.time_vec[i])
            voltages=voltages/self.nd_param.sw_class.c_E0
        elif self.simulation_options["method"]=="sum_of_sinusoids":
            for i in range(0, len(self.time_vec)):
                voltages[i]=isolver_martin_brent.sum_of_sinusoids_E(self.nd_param.nd_param_dict["amp_array"],self.nd_param.nd_param_dict["freq_array"],
                                                                    self.nd_param.nd_param_dict["phase_array"],self.nd_param.nd_param_dict["num_frequencies"], self.time_vec[i])
        if transient==True:
            voltages=voltages[self.time_idx]
        return voltages
    def top_hat_filter(self, time_series):

        L=len(time_series)
        window=np.hanning(L)
        if self.simulation_options["hanning"]==True:
            time_series=np.multiply(time_series, window)
        f=np.fft.fftfreq(len(time_series), self.time_vec[1]-self.time_vec[0])
        Y=np.fft.fft(time_series)
        frequencies=f

        #Y_pow=np.power(copy.deepcopy(Y[0:len(frequencies)]),2)
        top_hat=copy.deepcopy(Y)
        scale_flag=False
        true_harm=self.nd_param.nd_param_dict["omega"]*self.nd_param.c_T0

        if self.simulation_options["fourier_scaling"]!=None:
                scale_flag=True
        if sum(np.diff(self.harmonic_range))!=len(self.harmonic_range)-1 or scale_flag==True:
            results=np.zeros(len(top_hat), dtype=complex)
            for i in range(0, self.num_harmonics):
                true_harm_n=true_harm*self.harmonic_range[i]
                index=tuple(np.where((frequencies<(true_harm_n+(true_harm*self.filter_val))) & (frequencies>true_harm_n-(true_harm*self.filter_val))))
                if scale_flag==True:
                    filter_bit=abs(top_hat[index])
                    min_f=min(filter_bit)
                    max_f=max(filter_bit)
                    filter_bit=[self.normalise(x, [min_f, max_f]) for x in filter_bit]
                else:
                    filter_bit=top_hat[index]
                results[index]=filter_bit
        else:
            first_harm=(self.harmonic_range[0]*true_harm)-(true_harm*self.filter_val)
            last_harm=(self.harmonic_range[-1]*true_harm)+(true_harm*self.filter_val)
            freq_idx_1=tuple(np.where((frequencies>first_harm) & (frequencies<last_harm)))
            freq_idx_2=tuple(np.where((frequencies<-first_harm) & (frequencies>-last_harm)))
            likelihood_1=top_hat[freq_idx_1]
            likelihood_2=top_hat[freq_idx_2]
            #self.test_frequencies=frequencies[np.where((frequencies>first_harm) & (frequencies<last_harm))]
            results=np.zeros(len(top_hat), dtype=complex)
            results[freq_idx_1]=likelihood_1
            results[freq_idx_2]=likelihood_2
        #comp_results=np.real(np.fft.ifft(results))
        #plt.plot(self.other_values["experiment_voltage"],comp_results)

        #plt.plot(self.secret_data_time_series)
        #plt.show()
        if self.simulation_options["top_hat_return"]=="abs":
            return abs(results)
        elif self.simulation_options["top_hat_return"]=="imag":
            return np.imag(results)
        elif self.simulation_options["top_hat_return"]=="real":
            return np.real(results)
        elif self.simulation_options["top_hat_return"]=="composite":
            self.comp_results=np.append(np.real(results), np.imag(results))
            return self.comp_results
        elif self.simulation_options["top_hat_return"]=="inverse":
            return np.fft.ifft(results)
    def abs_transform(self, data):
        window=np.hanning(len(data))
        hanning_transform=np.multiply(window, data)
        f_trans=abs(np.fft.fft(hanning_transform[len(data)/2+1:]))
        return f_trans
    def saved_param_simulate(self, params):
        if self.file_init==False:
            raise ValueError('No file provided')
        else:
            self.def_optim_list(self.save_dict["optim_list"])
            type=self.simulation_options["likelihood"]
            return self.test_vals(params,type, test=False)
    def save_state(self, results, filepath, filename, params):
        other_vals_save=self.other_values
        other_vals_save["experiment_time"]=results["experiment_time"]
        other_vals_save["experiment_current"]=results["experiment_current"]
        other_vals_save["experiment_voltage"]=results["experiment_voltage"]
        file=open(filepath+"/"+filename, "wb")
        save_dict={"simulation_opts":self.simulation_options, \
                    "other_vals":other_vals_save, \
                    "bounds":self.param_bounds, \
                    "param_dict":self.dim_dict ,\
                    "params":params, "optim_list":self.optim_list}
        pickle.dump(save_dict, file, pickle.HIGHEST_PROTOCOL)
        file.close()
    def Kalman_capacitance(self, q, error=0.005):
        #error=0.005*max(self.secret_data_time_series)
        my_filter = KalmanFilter(dim_x=2, dim_z=1)
        my_filter.x = np.array([[self.secret_data_time_series[0]],
                        [1.]])       # initial state (location and velocity)
        dt=self.time_vec[1]-self.time_vec[0]
        cdl=self.nd_param.nd_param_dict["Cdl"]
        ru=self.nd_param.nd_param_dict["Ru"]
        dt_rc=dt/(ru*cdl)
        my_filter.F=np.array([[1/(1+dt_rc), (dt/ru)/(1+dt_rc)],[0, 0]])  # state transition matrix
        tr=self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]
        u=np.ones(len(self.time_vec))
        u[np.where(self.time_vec>tr)]*=-1
        my_filter.H = np.array([[1.,0.]])    # Measurement function
        my_filter.P *= error      # covariance matrix
        my_filter.R = error                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(2, dt, q) # process uncertainty
        my_filter.B=np.array([[0], [1]])
        means, _, _, _=my_filter.batch_filter(self.secret_data_time_series, Fs=None, Qs=None, Hs=None, Bs=None, us=u)
        return means[:, 0, 0]
    def kalman_pure_capacitance(self, current, q):
        my_filter = KalmanFilter(dim_x=2, dim_z=1)
        my_filter.x = np.array([[self.nd_param.nd_param_dict["Cdl"]],
                        [0]])       # initial state (location and velocity)
        dt=self.time_vec[1]-self.time_vec[0]

        my_filter.F=np.array([[1, dt],[0, 1]])  # state transition matrix
        tr=self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]
        u=np.ones(len(self.time_vec))
        ru=self.nd_param.nd_param_dict["Ru"]
        if self.simulation_options["method"]=="dcv":
            u[np.where(self.time_vec>tr)]*=-1
            self.kalman_u=u
            pred_cap=self.kalman_u*current
        elif self.simulation_options["method"]=="sinusoidal":
            orignal_phase=self.nd_param.nd_param_dict["phase"]
            self.nd_param.nd_param_dict["phase"]=self.nd_param.nd_param_dict["cap_phase"]
            for i in range(0, len(self.time_vec)):
                u[i]=(self.voltage_query(self.time_vec[i])[1])
            self.kalman_u=u
            pred_cap=np.divide((current), u)
        norm_denom=np.zeros(len(current))
        norm_denom[0]=u[0]

        """
        for i in range(1, len(self.secret_data_time_series)):
            gradient=(self.secret_data_time_series[i]-self.secret_data_time_series[i-1])/dt
            norm_denom[i]=u[i]-ru*gradient
            norm=np.divide(1, norm_denom)
        """
        #H_array=[np.array([[1, 0]]) for x in range(0, len(norm_denom))]
        my_filter.H = np.array([[1.,0.]])
        my_filter.P *= 0.1      # covariance matrix
        my_filter.R = 1                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(2, dt, q) # process uncertainty
        means=np.zeros(len(self.secret_data_time_series))
        for i in range(0, len(self.time_vec)):
            my_filter.predict()
            my_filter.update(z=pred_cap[i], R=0.1)
            means[i]=my_filter.x[0][0]

        means=np.divide(means, self.kalman_u)
        return pred_cap, means
    def rolling_window(self,x, N, arg="same"):
        return np.convolve(x, np.ones(N)/N, mode=arg)
    def kalman_dcv_simulate(self, Faradaic_current, q):

        candidate_current=np.subtract(self.secret_data_time_series, Faradaic_current)
        predicted_capcitance, predicted_current=self.kalman_pure_capacitance(candidate_current, q)
        if self.simulation_options["Kalman_capacitance"]==True:
            print([(x, self.dim_dict[x]) for x in self.optim_list])
            self.pred_cap=predicted_current
            self.farad_current=Faradaic_current
        return np.add(predicted_current, Faradaic_current)
    def times(self):
        self.time_vec=np.arange(0, self.nd_param.nd_param_dict["time_end"], self.nd_param.nd_param_dict["sampling_freq"])
        #print(self.time_vec)
        #self.time_vec=np.linspace(0, self.nd_param.nd_param_dict["time_end"], num_points)
    def change_norm_group(self, param_list, method):
        normed_params=copy.deepcopy(param_list)
        if method=="un_norm":
            for i in range(0,len(param_list)):
                normed_params[i]=self.un_normalise(normed_params[i], [self.boundaries[0][i],self.boundaries[1][i]])
        elif method=="norm":
            for i in range(0,len(param_list)):
                normed_params[i]=self.normalise(normed_params[i], [self.boundaries[0][i],self.boundaries[1][i]])
        return normed_params
    def variable_returner(self):
        variables=self.nd_param.nd_param_dict
        for key in list(variables.keys()):
            if type(variables[key])==int or type(variables[key])==float or type(variables[key])==np.float64:
                print(key, variables[key])
    def test_vals(self, parameters, likelihood, test=False):
        orig_likelihood=self.simulation_options["likelihood"]
        orig_label=self.simulation_options["label"]
        orig_test=self.simulation_options["test"]
        self.simulation_options["likelihood"]=likelihood
        self.simulation_options["label"]="MCMC"
        self.simulation_options["test"]=test
        if self.simulation_options["numerical_debugging"]==False:
            results=self.simulate(parameters, self.frequencies)
            self.simulation_options["likelihood"]=orig_likelihood
            self.simulation_options["label"]=orig_label
            self.simulation_options["test"]=orig_test
            return results
        else:
            current_range, gradient=self.simulate(parameters, self.frequencies)
            self.simulation_options["likelihood"]=orig_likelihood
            self.simulation_options["label"]=orig_label
            self.simulation_options["test"]=orig_test
            return current_range, gradient
    def update_params(self, param_list):
        if len(param_list)!= len(self.optim_list):
            print(self.optim_list)
            print(param_list)
            raise ValueError('Wrong number of parameters')
        if self.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(param_list, "un_norm")
        else:
            normed_params=copy.deepcopy(param_list)
        print(normed_params, self.optim_list)
        for i in range(0, len(self.optim_list)):
            self.dim_dict[self.optim_list[i]]=normed_params[i]
        self.nd_param=params(self.dim_dict)
    def return_distributions(self, bins):
        original_bins=self.simulation_options["dispersion_bins"]
        if type(bins) is not list:
            bins=[bins]
        self.simulation_options["dispersion_bins"]=bins
        if self.simulation_options["GH_quadrature"]==True:
            sim_params, values, weights=self.disp_class.generic_dispersion((self.nd_param.nd_param_dict), self.other_values["GH_dict"])
        else:
            sim_params, values, weights=self.disp_class.generic_dispersion((self.nd_param.nd_param_dict))
        self.simulation_options["dispersion_bins"]=original_bins
        return values, weights
    def SW_sampling(self,):
        #TODO implement setter property if someone changes deltaE part way through
        sampling_factor=self.dim_dict["sampling_factor"]
        self.end=int(abs(self.dim_dict['deltaE']//self.dim_dict['scan_increment']))

        p=np.array(range(1, self.end-1))

        self.b_idx=(sampling_factor*p)+(sampling_factor/2)
        self.f_idx=(p+1)*sampling_factor
        Es=self.dim_dict["E_start"]#-self.dim_dict["E_0"]
        self.E_p=(Es-(p*self.dim_dict['scan_increment']))
    def SW_peak_extractor(self, current, **kwargs):
        if "mean" not in kwargs:
            kwargs["mean"]=0
        if "window_length" not in kwargs:
            kwargs["window_length"]=1

        #TODO implement this in C++ for speed
        j=np.array(range(1, self.end*self.dim_dict["sampling_factor"]))

        if kwargs["mean"]==0:
            forwards=np.zeros(len(self.f_idx))
            backwards=np.zeros(len(self.b_idx))
            forwards=np.array([current[x-1] for x in self.f_idx])
            backwards=np.array([current[int(x)-1] for x in self.b_idx])
        else:
            indexes=[self.f_idx, self.b_idx]
            sampled_currents=[np.zeros(len(self.f_idx)), np.zeros(len(self.b_idx))]
            colours=["red", "green"]
            mean_idx=copy.deepcopy(sampled_currents)
            for i in range(0, len(self.f_idx)):
                for j in range(0, len(sampled_currents)):
                    x=indexes[j][i]
                    data=self.rolling_window(current[int(x-kwargs["mean"]-1):int(x-1)], kwargs["window_length"])
                    #plt.scatter(range(int(x-kwargs["mean"]-1),int(x-1)), data, color=colours[j])
                    #mean_idx[j][i]=np.mean(range(int(x-kwargs["mean"]-1),int(x-1)))
                    #plt.scatter(np.mean(range(int(x-kwargs["mean"]-1),int(x-1))), np.mean(data))
                    sampled_currents[j][i]=np.mean(data)
            #plt.plot(current)
            #plt.scatter(mean_idx[0], sampled_currents[0], color="red", marker="x")
            #plt.scatter(mean_idx[1], sampled_currents[1], color="green", marker="x")
            forwards=np.zeros(len(self.f_idx))
            backwards=np.zeros(len(self.b_idx))
            forwards=np.array([current[x-1] for x in self.f_idx])
            backwards=np.array([current[int(x)-1] for x in self.b_idx])
            #plt.scatter(self.f_idx-1, forwards, color="orange", marker="x")
            #plt.scatter(self.b_idx-1, backwards, color="yellow", marker="x")
            #plt.show()
            #forwards=sampled_currents[0]
            #backwards=sampled_currents[1]
        return forwards, backwards, forwards-backwards, self.E_p

    def paralell_disperse(self, solver):
        time_series=np.zeros(len(self.time_vec))

        if self.simulation_options["GH_quadrature"]==True:
            sim_params, self.values, self.weights=self.disp_class.generic_dispersion((self.nd_param.nd_param_dict), self.other_values["GH_dict"])
        else:
            sim_params, self.values, self.weights=self.disp_class.generic_dispersion((self.nd_param.nd_param_dict))
        self.disp_test=[]
        for i in range(0, len(self.weights)):
            for j in range(0, len(sim_params)):
                self.nd_param.nd_param_dict[sim_params[j]]=self.values[i][j]
            start=time.time()
            time_series_current=solver(self.nd_param.nd_param_dict, self.time_vec,self.simulation_options["method"], -1, self.bounds_val)
            print(time.time()-start)
            if self.simulation_options["dispersion_test"]==True:
                self.disp_test.append(time_series_current)
            time_series=np.add(time_series, np.multiply(time_series_current, np.prod(self.weights[i])))
        return time_series
    def NR_python(self):
        self.def_optim_list([])
        self.test_vals([], "timeseries")
        class_init=python_NR_simulation(self.time_vec, self.nd_param.nd_param_dict, self.voltage_query)
        return class_init.numerical_current
    def r_squared(self, data, prediction):
        residual_sq=np.sum(np.square(np.subtract(data, prediction)))
        mean=np.mean(data)
        ss_tot=np.sum(np.square(np.subtract(data, mean)))
        return 1-(residual_sq/ss_tot)
    def numerical_plots(self, solver):
        self.debug_time=self.simulation_options["numerical_debugging"]
        time_series=solver(self.nd_param.nd_param_dict, self.time_vec,self.simulation_options["method"], self.debug_time, self.bounds_val)
        current=time_series[0]
        residual=time_series[1]
        residual_gradient=time_series[2]

        #plt.subplot(1,2,1)
        #plt.semilogy(current, np.abs(residual))
        bounds_val=self.bounds_val
        middle_index=(len(time_series[0])-1)//2 + 1
        I0=residual[middle_index]
        if  self.simulation_options["numerical_method"]=="Newton-Raphson":
            plt.subplot(1,2,1)
            plt.title("Residual, t="+str(self.debug_time))
            plt.plot(current, residual)
            plt.axvline(time_series[3][1], color="red",linestyle="--")
            plt.axvline(time_series[3][0]+time_series[3][2], color="black", linestyle="--")
            plt.axvline(time_series[3][0]-time_series[3][2], color="black",linestyle="--")
            plt.subplot(1,2,2)
            plt.title("Residual gradient")
            plt.plot(current, ((residual_gradient)))
            plt.show()
        else:
            return current, residual
    def downsample(self, times, data, samples):
        sampled_data=np.interp(samples, times, data)
        return sampled_data
    def simulate(self,parameters, frequencies):
        start=time.time()
        if len(parameters)!= len(self.optim_list):
            print(self.optim_list)
            print(parameters)
            raise ValueError('Wrong number of parameters')
        if self.simulation_options["label"]=="cmaes":
            normed_params=self.change_norm_group(parameters, "un_norm")
        else:
            normed_params=copy.deepcopy(parameters)
        #print(list(normed_params), self.optim_list)
        if self.simulation_options["method"]=="sum_of_sinusoids":
            self.max_freq=0
            self.min_freq=1e9
            if len(self.optim_list)!=0:
                for key in self.dict_of_sine_list.keys():     
                    array_key=key+"array"
                    self.dim_dict[array_key]=np.zeros(self.dim_dict["num_frequencies"])
                    for i in range(0, self.dim_dict["num_frequencies"]):
                        if key=="freq_":
                            if normed_params[self.dict_of_sine_list[key][i]]>self.max_freq:
                                self.max_freq=normed_params[self.dict_of_sine_list[key][i]]
                            if normed_params[self.dict_of_sine_list[key][i]]<self.min_freq:
                                self.min_freq=normed_params[self.dict_of_sine_list[key][i]]
                        self.dim_dict[array_key][i]=normed_params[self.dict_of_sine_list[key][i]]
                self.dim_dict["omega"]=self.max_freq
                self.dim_dict["original_omega"]=self.min_freq
                for i in range(0, len(self.param_positions)):
                    self.dim_dict[self.optim_list[self.param_positions[i]]]=normed_params[self.param_positions[i]]
        for i in range(0, len(self.optim_list)):
            self.dim_dict[self.optim_list[i]]=normed_params[i]
        if self.simulation_options["phase_only"]==True:
            self.dim_dict["cap_phase"]=self.dim_dict["phase"]
       
        self.nd_param=params(self.dim_dict)
        if self.simulation_options["method"]=="sum_of_sinusoids":
            self.nd_param.nd_param_dict["time_end"]=5
            self.times()
        
        
        if self.simulation_options["voltage_only"]==True:
            return self.define_voltages()[self.time_idx]
        if self.simulation_options["adaptive_ru"]==True:
            if self.dim_dict["Ru"]>1000:
                self.simulation_options["numerical_method"]="pybamm"
            else:
                self.simulation_options["numerical_method"]="Brent minimisation"
        if self.simulation_options["numerical_method"]=="Brent minimisation":
            solver=isolver_martin_brent.brent_current_solver
        elif self.simulation_options["numerical_method"]=="Newton-Raphson":
            solver=isolver_martin_NR.NR_current_solver
            if self.simulation_options["method"]=="dcv":
                raise ValueError("Newton-Raphson dcv simulation not implemented")
        elif self.simulation_options["numerical_method"]=="Kalman_simulate":
            if self.simulation_options["method"]=="ramped":
                raise ValueError("Ramped not implemented for Kalman approach")
            else:
                cdl_record=self.nd_param.nd_param_dict["Cdl"]
                self.nd_param.nd_param_dict["Cdl"]=0
                solver=isolver_martin_brent.brent_current_solver

        elif self.simulation_options["numerical_method"]=="pybamm":
            pybamm_sol=pybamm_solver(self)
            solver=pybamm_sol.simulate
        elif self.simulation_options["numerical_method"]=="scipy":
            if self.simulation_options["scipy_type"]==None:
                warnings.warn("No defined reaction mechanism, assuming single electron Faradaic")
                self.simulation_options["scipy_type"]="single_electron"
            scipy_class=scipy_funcs(self)
            self.scipy_class=scipy_class
            solver=scipy_class.simulate_current
        else:
            raise ValueError('Numerical method not defined')
        if self.simulation_options["method"]=="square_wave" and self.simulation_options["numerical_method"]!="scipy":
            solver=SWV_surface.SWV_current
        if self.simulation_options["numerical_debugging"]!=False:
            current_range, gradient=self.numerical_plots(solver)
            return current_range, gradient
        else:
            if self.simulation_options["dispersion"]==True:
                time_series=self.paralell_disperse(solver)
            else:
                if self.simulation_options["numerical_method"]=="pybamm":
                    try:
                        time_series=solver(self.nd_param.nd_param_dict, self.time_vec, self.simulation_options["method"],-1, self.bounds_val)
                    except:

                        time_series=np.zeros(len(self.time_vec))#isolver_martin_brent.brent_current_solver(self.nd_param.nd_param_dict, self.time_vec, self.simulation_options["method"],-1, self.bounds_val)
                else:
                    
                    
                    
                    time_series=solver(self.nd_param.nd_param_dict, self.time_vec, self.simulation_options["method"],-1, self.bounds_val)
        #print(time.time()-start)
        if self.simulation_options["numerical_method"]=="Kalman_simulate":
            self.nd_param.nd_param_dict["Cdl"]=cdl_record
            time_series=self.kalman_dcv_simulate(time_series, self.dim_dict["Q"])
        time_series=np.array(time_series)
        if self.simulation_options["sample_times"] is not False:
            time_series=self.downsample(self.time_vec, time_series, self.simulation_options["sample_times"])
            print(len(time_series))
        elif self.simulation_options["no_transient"]!=False:
            time_series=time_series[self.time_idx]
        
        if self.simulation_options["method"]=="square_wave":
            if self.simulation_options["square_wave_return"]=="net":
                _, _, net, _=self.SW_peak_extractor(time_series)
                time_series=net

        if self.simulation_options["likelihood"]=='fourier':
            filtered=self.top_hat_filter(time_series)
            if (self.simulation_options["test"]==True):
                self.variable_returner()
                plt.plot(self.secret_data_fourier, label="data")
                plt.plot(filtered , alpha=0.7, label="numerical")
                plt.legend()
                plt.show()

            if self.simulation_options["multi_output"]==True:
                return np.column_stack((np.real(filtered), np.imag(filtered)))
            else:
                return filtered
        elif self.simulation_options["likelihood"]=='timeseries':
            if self.simulation_options["test"]==True:
                print(list(normed_params))
                if self.simulation_options["experimental_fitting"]==True:
                    plt.subplot(1,2,1)
                    plt.plot(self.other_values["experiment_voltage"],time_series)
                    plt.subplot(1,2,2)
                    plt.plot(self.other_values["experiment_time"],time_series)
                    plt.plot(self.other_values["experiment_time"],self.secret_data_time_series, alpha=0.7)
                    plt.show()
                else:
                    plt.plot(self.time_vec[self.time_idx], time_series)
                    plt.plot(self.time_vec[self.time_idx], self.secret_data_time_series)
                    plt.show()
            return time_series
    def options_checker(self, simulation_options):
        if "no_transient" not in simulation_options:
            simulation_options["no_transient"]=False
        if "numerical_debugging" not in simulation_options:
            simulation_options["numerical_debugging"]=False
        if "experimental_fitting" not in simulation_options:
            raise KeyError("Experimental fitting option not found - please define")
        if "test" not in simulation_options:
            simulation_options["test"]=False
        if "method" not in simulation_options:
            raise KeyError("Please define a simulation method")
        if "phase_only" not in simulation_options:
            simulation_options["phase_only"]=False
        if "likelihood" not in simulation_options:
            raise KeyError("Please define a likelihood/objective - timeseries or fourier domain")
        if "numerical_method" not in simulation_options:
            simulation_options["numerical_method"]="Brent minimisation"
        if "label" not in simulation_options:
            simulation_options["label"]="MCMC"
        if "adaptive_ru" not in simulation_options:
            simulation_options["adaptive_ru"]=False
        if "optim_list" not in simulation_options:
            simulation_options["optim_list"]=[]
        if "GH_quadrature" not in simulation_options:
            simulation_options["GH_quadrature"]=False
        if "hanning" not in simulation_options:
            simulation_options["hanning"]=True
        if "voltage_only" not in simulation_options:
            simulation_options["voltage_only"]=False
        if "scipy_type" not in simulation_options:
            simulation_options["scipy_type"]=None
        if "square_wave_return" not in simulation_options:
            simulation_options["square_wave_return"]="total"
        if "top_hat_return" not in simulation_options:
            simulation_options["top_hat_return"]="composite"
        if "fourier_scaling" not in simulation_options:
            simulation_options["fourier_scaling"]=None
        if "Kalman_capacitance" not in simulation_options:
            simulation_options["Kalman_capacitance"]=False
        if "multi_output" not in simulation_options:
            simulation_options["multi_output"]=False
        if "dispersion_test" not in simulation_options:
            simulation_options["dispersion_test"]=False
        if "sample_times" not in simulation_options:
            simulation_options["sample_times"]=False
        return simulation_options
    def param_checker(self, params):
        for key in ["freq_array", "amp_array", "phase_array"]:
            if key not in params:
                params[key]=[0]
        for key in ["num_frequencies"]:
            if key not in params:
                params[key]=len(params["freq_array"])
        return params
class paralell_class:
    def __init__(self, params, times, method, bounds, solver):
        self.params=params
        self.times=times
        self.method=method
        self.bounds=bounds
        self.solver=solver
    def paralell_simulate(self, weight_val_entry):
        start=time.time()
        self.sim_params=copy.deepcopy(self.params)
        for i in range(len(weight_val_entry[0])):
            self.sim_params[weight_val_entry[0][i]]=weight_val_entry[1][i]
        time_series=self.solver(self.sim_params, self.times, self.method,-1, self.bounds)
        time_series=np.multiply(time_series, weight_val_entry[2])
        return (time_series)
    def paralell_dispersion(self, weight_list):
        p = mp.Pool(4)
        start1=time.time()
        sc = p.map_async(self,  [weight for weight in weight_list])
        start=time.time()
        results=sc.get()
        p.close()
        disped_time=np.sum(results, axis=0)
        start2=time.time()
        return disped_time
    def __call__(self, x):
        return self.paralell_simulate(x)
