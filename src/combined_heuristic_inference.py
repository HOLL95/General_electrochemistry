import math
import numpy as np
from single_e_class_unified import single_electron
from harmonics_plotter import harmonics
from heuristic_class import DCVTrumpet, PSV_harmonic_minimum, Laviron_EIS
import pints
import copy
import matplotlib.pyplot as plt
class combined_heuristics:
    def __init__(self, common_params, common_options, common_others, common_bounds, global_options,discriminator, *args):
        all_dicts=[discriminator, *args]
        all_names=[x["name"] for x in all_dicts] 
        option_dict=dict(zip(all_names, all_dicts))
        self.discriminator_key=discriminator["name"]
        self.slow_keys=[x["name"] for x in args]
        class_dict={}
        self.x_values={}
        self.y_values={}
        class_switch={"EIS":Laviron_EIS, "Harmonic_min":PSV_harmonic_minimum, "Trumpet":DCVTrumpet}
        input_switch={"params":common_params, "options":common_options, "others":common_others, "bounds":common_bounds}
        self.common_optim_list=list(set().union(*[option_dict[x]["options"]["optim_list"] for x in option_dict.keys()]))
        self.common_bounds={key:common_bounds[key] for key in self.common_optim_list}
        for method_key in option_dict.keys():
            method=option_dict[method_key]
            current_object=copy.deepcopy(input_switch)
            self.x_values[method_key]=method["times"]
            for input_key in input_switch.keys():
                for option_key in method[input_key].keys():
                    current_object[input_key][option_key]=method[input_key][option_key]
            class_dict[method_key]=class_switch[method_key](current_object["params"], current_object["options"], current_object["others"], current_object["bounds"])
            class_dict[method_key].def_optim_list(current_object["options"]["optim_list"])
            if "values" not in method:
                
                self.y_values[method_key]=class_dict[method_key].synthetic_noise([class_dict[method_key].dim_dict[x] for x in class_dict[method_key].optim_list], self.x_values[method_key], method["noise"] )
        for i in range(0, len(all_names)):
            print(global_options["weights"][i])
            self.y_values[all_names[i]]=np.multiply(self.y_values[all_names[i]], global_options["weights"][i])
        self.discriminator_class=class_dict[discriminator["name"]]
        self.class_dict=class_dict
        self.discriminator_idx=[self.common_optim_list.index(x) for x in discriminator["options"]["optim_list"]]
        self.other_idx=[[self.common_optim_list.index(x) for x in self.class_dict[key].optim_list ] for key in self.slow_keys]
        self.discriminator_noise=discriminator["noise_bounds"]
        self.total_likelihood=self.data_combiner([self.y_values[x] for x in all_names])
        self.global_options=global_options
        
        if self.y_values[self.discriminator_key].ndim>1:
            score_func=pints.MultiOutputProblem(self.discriminator_class,self.x_values[self.discriminator_key], self.y_values[self.discriminator_key])
            len_nb=2
        else:
            score_func=pints.SingleOutputProblem(self.discriminator_class,self.x_values[self.discriminator_key], self.y_values[self.discriminator_key])
            len_nb=1
        self.pints_problem=score = pints.SumOfSquaresError(score_func)
        self.current_best=self.pints_problem(np.random.rand(len(self.discriminator_class.optim_list)))
    def simulate(self, parameters, times, lazy_evaluation=False):
        if self.global_options["label"]=="cmaes":
            parameters=self.change_norm_group(parameters, "un_norm")
            #print(list(parameters))
        discriminator_params=[parameters[x] for x in self.discriminator_idx]
        continuation=False
        if lazy_evaluation==True:
            
            if self.pints_problem is None:
                raise ValueError("Need to specify pints.problem")
            else:
                discriminator_value=self.pints_problem(discriminator_params)
                if self.current_best>discriminator_value:
                    continuation=True
                    self.current_best=discriminator_value
                else:
                    ratio= self.current_best/discriminator_value#if the fit is bad->0 if the fit is good ->1
                    
                    u_rand=np.random.rand()
                    print("current best=", self.current_best, "iteration=",discriminator_value, "ratio=",ratio,"urand=",u_rand )
                    if u_rand<ratio:
                        continuation=True
        if lazy_evaluation==False or continuation==True:
            discriminator_data=self.discriminator_class.simulate(discriminator_params, self.x_values[self.discriminator_key])
            data=[discriminator_data]
            for i in range(0, len(self.slow_keys)):
                key=self.slow_keys[i]
                slow_class=self.class_dict[key]
                slow_params=[parameters[self.other_idx[i][j]] for j in range(0, len(self.other_idx[i]))]
                slow_x_val=self.x_values[key]
                data.append(slow_class.simulate(slow_params, slow_x_val))
            for i in range(0, len(data)):
                data[i]=np.multiply(data[i], self.global_options["weights"][i])
            final_data=self.data_combiner(data)
            if self.global_options["test"]==True:
                print(lazy_evaluation, continuation)
                print(self.common_optim_list)
                print(list(parameters))
                plt.plot(self.total_likelihood)
                plt.plot(final_data)
                plt.show()
        else:
            final_data=np.ones(len(times))*1e10
        if self.global_options["return_arg"]=="scalarised":
            return final_data
        else:
            return data
    def data_combiner(self, data):
        final_data=[]
        for i in range(0, len(data)):
            
            if data[i].ndim>1:
                for j in range(0, 2):
                    size=data[i].shape
                    if size[0]>size[1]:
                        final_data=np.append(final_data, data[i][:,j])
                    else:
                        final_data=np.append(final_data, data[i][j,:])
            else:
                final_data=np.append(final_data, data[i])
        return final_data
    def get_MAP(self,noise_bounds, num_runs=1):
        orig_label=self.discriminator_class.simulation_options["label"]
        self.discriminator_class.simulation_options["label"]="cmaes"
        if self.y_values[self.discriminator_key].ndim>1:
            cmaes_problem=pints.MultiOutputProblem(self.discriminator_class,self.x_values[self.discriminator_key], self.y_values[self.discriminator_key])
            len_nb=2
        else:
            cmaes_problem=pints.SingleOutputProblem(self.discriminator_class,self.x_values[self.discriminator_key], self.y_values[self.discriminator_key])
            len_nb=1
        score = pints.GaussianLogLikelihood(cmaes_problem)
        lower_bound=np.append(np.zeros(len(self.discriminator_idx)), [noise_bounds[0]]*len_nb)
        upper_bound=np.append(np.ones(len(self.discriminator_idx)), [noise_bounds[1]]*len_nb)
        CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
        best_score=-1e6
        for i in range(0, num_runs):
            x0=list(np.random.rand(len(self.discriminator_idx)))+[(noise_bounds[1]+noise_bounds[0])/2]*len_nb
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=[0.075 for x in range(0, self.discriminator_class.n_parameters()+len_nb)], boundaries=CMAES_boundaries, method=pints.CMAES)
            cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-4)
            cmaes_fitting.set_parallel(True)
            found_parameters, found_value=cmaes_fitting.run()
            if found_value>best_score:
                dim_params=self.discriminator_class.change_norm_group(found_parameters[:-len_nb], "un_norm")
                best_score=found_value   

        dim_param_dict=dict(zip(self.discriminator_class.optim_list, dim_params))
        self.discriminator_class.simulation_options["label"]=orig_label
        self.pints_problem=score
        return dim_param_dict, found_value, found_parameters[-len_nb:]
    def change_norm_group(self, param_list, method):
        normed_params=copy.deepcopy(param_list)
        if method=="un_norm":
            for i in range(0,len(param_list)):
                boundary=self.common_bounds[self.common_optim_list[i]]
                normed_params[i]=self.discriminator_class.un_normalise(normed_params[i], boundary)
        elif method=="norm":
            for i in range(0,len(param_list)):
                boundary=self.common_bounds[self.common_optim_list[i]]
                normed_params[i]=self.discriminator_class.normalise(normed_params[i],boundary)
        return normed_params
    def n_outputs(self):
        return 1
    def n_parameters(self,):
        return len(self.common_optim_list)
    def test_vals(self, parameters):
        orig_label=self.global_options["label"]
        self.global_options["label"]="MCMC"
        data=self.simulate(parameters, [], lazy_evaluation=False)
        self.global_options["label"]=orig_label
        return data
"""class CombinedLikelihood(pints.ProblemLogLikelihood):
  
    def __init__(self, problem):
        super(CombinedLikelihood, self).__init__(problem)

        # Get number of times, number of outputs
        self._nt = len(self._times)
        self._no = problem.n_outputs()

        # Add parameters to problem
        self._n_parameters = problem.n_parameters() + self._no

        # Pre-calculate parts
        self._logn = 0.5 * self._nt * np.log(2 * np.pi)

    def __call__(self, x):
        sigma = np.asarray(x[-self._no:])
        if any(sigma <= 0):
            return -np.inf
        error = self._values - self._problem.evaluate(x[:-self._no])
        get_values=
        return np.sum(- self._logn - self._nt * np.log(sigma)
                      - np.sum(error**2, axis=0) / (2 * sigma**2))"""


