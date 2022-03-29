import numpy as np
import matplotlib.pyplot as plt
import copy
import pints
import random
import math
import time
from EIS_class import EIS
from circuit_drawer import circuit_artist
class EIS_optimiser(EIS):
    def __init__(self, circuit,  **kwargs):
        self.param_bounds=kwargs["parameter_bounds"]
        self.params=kwargs["param_names"]
        self.optimisation_options={}
        if "test" in kwargs:
            self.test=kwargs["test"]
        else:
            self.test=False
        self.frequency_range=kwargs["frequency_range"]
        circuit_keys=circuit.keys()
        self.optimisation_options["tree"]=False
        for key in circuit.keys():
            if "root" in key:
                self.optimisation_options["tree"]=True
                break
        if self.optimisation_options["tree"]==True:
            dict_constructor=super().__init__()
            circuit_as_dict=dict_constructor.translate_tree(circuit)
        else:
            circuit_as_dict=circuit
        super().__init__(parameter_bounds=self.param_bounds, parameter_names=self.params, circuit=circuit_as_dict, fitting=True, test=self.test)
    def get_std(self, sim, data):
        std_list=[0,0]
        for i in range(0, 2):
            subtract=np.square(np.subtract(sim[:,i], data[:,i]))
            std_list[i]=np.std(subtract)
        return std_list

    def optimise(self, data, sigma_fac=0.001, method="minimisation"):
        cmaes_problem=pints.MultiOutputProblem(self, self.frequency_range, data)
        if method=="likelihood":
            score = pints.GaussianLogLikelihood(cmaes_problem)
            sigma=sigma_fac*np.sum(data)/2*len(data)
            lower_bound=[self.param_bounds[x][0] for x in self.params]+[0.1*sigma]*2
            upper_bound=[self.param_bounds[x][1] for x in self.params]+[10*sigma]*2
            CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
            random_init=abs(np.random.rand(self.n_parameters()))
            x0=self.change_norm_group(random_init, "un_norm", "list")+[sigma]*2
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
        elif method=="minimisation":
            score = pints.SumOfSquaresError(cmaes_problem)
            lower_bound=[self.param_bounds[x][0] for x in self.params]
            upper_bound=[self.param_bounds[x][1] for x in self.params]
            CMAES_boundaries=pints.RectangularBoundaries(lower_bound, upper_bound)
            random_init=abs(np.random.rand(self.n_parameters()))
            x0=self.change_norm_group(random_init, "un_norm", "list")
            cmaes_fitting=pints.OptimisationController(score, x0, sigma0=None, boundaries=CMAES_boundaries, method=pints.CMAES)
        cmaes_fitting.set_max_unchanged_iterations(iterations=200, threshold=1e-7)
        #cmaes_fitting.set_log_to_screen(False)
        cmaes_fitting.set_parallel(True)

        found_parameters, found_value=cmaes_fitting.run()


        if method=="likelihood":
            sim_params=found_parameters[:-2]
            sim_data= self.simulate(sim_params,self.frequency_range)
        else:
            found_value=-found_value
            sim_params=found_parameters
            sim_data= self.simulate(sim_params,self.frequency_range)
            """

            log_score = pints.GaussianLogLikelihood(cmaes_problem)
            stds=self.get_std(data, sim_data)
            sigma=sigma_fac*np.sum(data)/2*len(data)
            score_params=list(found_parameters)+[sigma]*2
            found_value=log_score(score_params)
            print(stds, found_value, "stds")"""

        #DOITDIMENSIONALLY#NORMALISE DEFAULT TO BOUND
        return found_parameters, found_value,cmaes_fitting._optimiser._es.sm.C,sim_data
class EIS_genetics:
    def __init__(self,  **kwargs):
        if "initial_circuit" not in kwargs:
            kwargs["initial_circuit"]=False
        if "initial_tree_size" not in kwargs:
            kwargs["initial_tree_size"]=4
        if "maximum_mutation_size" not in kwargs:
            kwargs["maximum_mutation_size"]=3
        if "minimum_replacement_depth" not in kwargs:
            kwargs["minimum_replacement_depth"]=1
        if "generation_size" not in kwargs:
            kwargs["generation_size"]=4
        if "generation_test" not in kwargs:
            kwargs["generation_test"]=False
        elif "generation_test_save" not in kwargs:
            kwargs["generation_test_save"]=False
        elif kwargs["generation_size"]%2!=0:
            kwargs["generation_size"]+=1
        if "selection" not in kwargs:
            kwargs["selection"]="bayes_factors"
        if "best_record" not in kwargs:
            kwargs["best_record"]=False
        elif kwargs["best_record"]==True:
            self.best_array=[]
            self.best_circuits=[]
        if "num_top_circuits" not in kwargs:
            kwargs["num_top_circuits"]=10
        if kwargs["initial_circuit"]==False:
            initial_generation=[]#
            random_constructor=EIS()
            for i in range(0, kwargs["generation_size"]):
                initial_generation.append(random_constructor.random_circuit_tree(kwargs["initial_tree_size"]))
        self.options=kwargs
        self.initial_generation=initial_generation
        self.best_candidates={"scores":np.array([]), "models":[], "parameters":[], "data":[]}
        self.count=0
    def plot_data(self, data, ax, **kwargs):
        if "label" not in kwargs:
            kwargs["label"]=None
        ax.plot(data[:,0], -data[:,1], label=kwargs["label"])
    def tree_simulation(self, circuit_tree, frequencies, data, param_vals):
        translator=EIS()
        dictionary, params=translator.translate_tree(circuit_tree, get_param_list=True)
        sim_class=self.assess_score(dictionary, params, frequencies, data,get_simulator=True)
        return sim_class.simulate(param_vals, frequencies)
    def dict_simulation(self, circuit_dict, frequencies, data, param_vals, param_names):
        sim_class=self.assess_score(circuit_dict, param_names, frequencies, data,get_simulator=True)
        return sim_class.simulate(param_vals, frequencies)
    def assess_score(self, circuit_dictionary, parameter_list, frequencies, data, **methods):
        if "score_func" not in methods:
            methods["score_func"]="bayes_factors"#
        if "get_simulator" not in methods:
            methods["get_simulator"]=False

        param_bounds={}
        print(parameter_list)
        for param in parameter_list:
            if "R" in param:
                param_bounds[param]=[0, 1000]
            elif "C" in param:
                param_bounds[param]=[0, 1e-2]
            elif "Q" in param:
                param_bounds[param]=[0, 1e-2]
            elif "alpha" in param:
                param_bounds[param]=[0, 1]
            elif "W" in param:
                param_bounds[param]=[0, 200]
            elif "gamma" in param:
                param_bounds[param]=[0, 200]
            elif "delta" in param:
                param_bounds[param]=[0, 200]#in micrometers?
            elif "D" in param:
                param_bounds[param]=[0, 2300]

        optimiser=EIS_optimiser(circuit=circuit_dictionary, parameter_bounds=param_bounds, param_names=parameter_list, frequency_range=frequencies)
        if methods["get_simulator"]==True:
            return optimiser
        method="minimisation"
        found_value=-1e9
        for i in range(0, 2):
            cmaes_params, cmaes_value, cov, simulation=optimiser.optimise(data, method=method)
            print(cmaes_value, found_value)
            if cmaes_value>found_value:

                return_params=cmaes_params
                found_value=cmaes_value
                covariance=cov
                sim_data=simulation

        #fig2, ax2=plt.subplots(1, 2)
        #print(circuit_dictionary)
        #circuit_artist(circuit_dictionary, ax2[1])
        #self.plot_data(sim_data, ax2[0])
        #self.plot_data(data, ax2[0])
        #plt.show()
        if methods["score_func"]=="bayes_factors":
            MLI_term_1=np.power(2*math.pi, len(parameter_list)/2)*found_value
            prior=np.prod([param_bounds[x][1]-param_bounds[x][0] for x in parameter_list])
            MLI_term_2=np.sqrt(np.linalg.det(covariance))/prior
            if self.options["generation_test"]==True:
                return MLI_term_1*MLI_term_2, return_params, sim_data
            else:
                return MLI_term_1*MLI_term_2, return_params
        elif methods["score_func"]=="BIC" or methods["score_func"]=="AIC":
            if methods["score_func"]=="BIC":
                sub=len(parameter_list)*np.log(len(data[:,0])*2)
            else:
                sub=2*len(parameter_list)
            return_val=found_value-sub#*len(parameter_list)#-BIC
            if self.options["generation_test"]==True:
                return return_val, return_params, sim_data
            else:
                return return_val, return_params

    def get_generation_score(self, generation, frequencies, data, selection):
        dictionary_constructor=EIS()
        generation_scores=np.zeros(len(generation))
        returned_params=[]
        if self.options["generation_test"]==True:
            returned_data=[]
        for i in range(0, len(generation)):
            circuit_dict, param_list=dictionary_constructor.translate_tree(generation[i], get_param_list=True)
            if len(param_list)==1:
                meaningful_circuit=False
                while meaningful_circuit==False:
                    new_try=dictionary_constructor.random_circuit_tree(self.options["initial_tree_size"])
                    circuit_dict, param_list=dictionary_constructor.translate_tree(new_try, get_param_list=True)
                    if len(param_list)>1:
                        meaningful_circuit=True
                        generation[i]=new_try
            if self.options["generation_test"]==True:
                generation_scores[i], params, sim_data=self.assess_score(circuit_dict, param_list, frequencies, data, score_func=selection)
            else:
                generation_scores[i], params=self.assess_score(circuit_dict, param_list, frequencies, data, score_func=selection)
            returned_params.append(params)
            if self.options["generation_test"]==True:
                returned_data.append(sim_data)
        if self.options["generation_test"]==True:
            return generation_scores, returned_params, returned_data
        else:
            return generation_scores, returned_params,
    def selection(self, generation, frequencies, data, method):
        if self.options["generation_test"]==True:
            generation_score, param_array, sim_array=self.get_generation_score(generation, frequencies, data, method)
            if self.options["generation_size"]>8:
                end=8
                fig, ax=plt.subplots(2, end)
            else:
                end=self.options["generation_size"]
                fig, ax=plt.subplots(2, end)
            plot_count=0
        else:
            generation_score, param_array=self.get_generation_score(generation, frequencies, data, method)
        if method=="bayes_factors":
            scores=list(enumerate(generation_score))
            winners=[0]*int(len(scores)/2)
            random.shuffle(scores)
            for i in range(0, len(scores), 2):
                bayes_factor=scores[i][1]/scores[i+1][1]
                if bayes_factor>1:
                    windex=i
                    loser=i+1
                else:
                    windex=i+1
                    loser=i
                winners[int(i/2)]=scores[windex][0]
                if self.options["generation_test"]==True:
                    if i<end:
                        self.plot_generation(scores[windex][0], scores[loser][0], generation_score, sim_array, generation,data, ax, plot_count)
                        plot_count+=2

        elif method=="BIC" or method=="AIC":
            rank=np.argsort(generation_score)
            winners=np.flip(rank[int(self.options["generation_size"])//2:])
            losers=np.flip(rank[:int(self.options["generation_size"])//2])
            if self.options["generation_test"]==True:
                for i in range(0, int(end//2)):
                    self.plot_generation(winners[i], losers[i], generation_score,sim_array, generation,data, ax, plot_count)
                    plot_count+=2
        if self.options["best_record"]==True:
            windex=winners[0]
            if len(self.best_array)==0:
                self.best_array.append(generation_score[windex])
                self.best_circuits.append(generation[windex])
            elif generation_score[windex]>self.best_array[-1]:
                self.best_array.append(generation_score[windex])
                self.best_circuits.append(generation[windex])
            else:
                self.best_array.append(self.best_array[-1])
                self.best_circuits.append(self.best_circuits[-1])
        winning_candidates=[generation[x] for x in winners]
        winning_params=[param_array[x] for x in winners]
        winning_scores=[generation_score[x] for x in winners]
        translator=EIS()
        num_entries=len(self.best_candidates["scores"])
        if num_entries==0:
            self.best_candidates["scores"]=winning_scores
            self.best_candidates["models"]=winning_candidates
            self.best_candidates["parameters"]=winning_params
        elif num_entries<self.options["num_top_circuits"]:
            discrepancy=self.options["num_top_circuits"]-num_entries
            for i in range(0, min(len(winners), discrepancy)):
                model=copy.deepcopy(winning_candidates[i])
                param_vals=copy.deepcopy(winning_params[i])
                self.best_candidates["scores"].append(copy.deepcopy(winning_scores[i]))
                self.best_candidates["models"].append(model)
                self.best_candidates["parameters"].append(param_vals)
        else:
            for i in range(0, len(winning_scores)):
                self.best_candidates["scores"]=np.array(self.best_candidates["scores"])
                if winning_candidates[i]==self.best_candidates["models"][i]:
                    if self.best_candidates["scores"][i]<winning_scores[i]:
                        self.best_candidates["scores"][i]=winning_scores[i]
                        self.best_candidates["parameters"][i]=winning_params[i]
                    continue
                better_loc=tuple(np.where(self.best_candidates["scores"]<winning_scores[i]))
                if len(better_loc[0])!=0:
                    better_idx=np.where(self.best_candidates["scores"]==min(self.best_candidates["scores"][better_loc]))[0][0]
                    model=copy.deepcopy(winning_candidates[i])
                    param_vals=copy.deepcopy(winning_params[i])
                    self.best_candidates["scores"][better_idx]=copy.deepcopy(winning_scores[i])
                    self.best_candidates["models"][better_idx]=model
                    self.best_candidates["parameters"][better_idx]=param_vals
        if self.options["generation_test"]==True:
            if self.options["generation_test_save"]==False:
                plt.show()
            else:
                plt.subplots_adjust(top=0.88,
                                    bottom=0.11,
                                    left=0.04,
                                    right=0.99,
                                    hspace=0.0,
                                    wspace=0.2)
                fig.set_size_inches(28, 18)
                fig.savefig(self.options["generation_test_save"]+"Generation{0}.png".format(self.generation_num))
                plt.clf()
                plt.close()
        return copy.deepcopy(winning_candidates), copy.deepcopy(winning_params)

    def reproduce(self, winning_generation, **methods):
        crossover_generation=[]
        next_generation=[]
        mutator=EIS()
        if "keep_winners" not in methods:
            methods["keep_winners"]=False
        if "num_elites" in methods:
            elites=copy.deepcopy(winning_generation[:methods["num_elites"]])
        if methods["keep_winners"]==False:
            num_shuffles=4
        elif  "num_elites" in methods:
            num_shuffles=4
            if methods["num_elites"]>len(winning_generation):
                raise ValueError("{0} is greater than generation size {1}".format(methods["num_elites"], len(winning_generation)))
        else:
            num_shuffles=2


        for i in range(0, num_shuffles):
            random.shuffle(winning_generation)
            for j in range(0, len(winning_generation), 2):
                original=copy.deepcopy(winning_generation[j])
                target=copy.deepcopy(winning_generation[j+1])
                child=mutator.crossover(original, target, self.options["minimum_replacement_depth"])
                crossover_generation.append(child)

        for j in range(0, len(crossover_generation)):
            new_mutator=EIS()
            original=copy.deepcopy(crossover_generation[j])
            next_generation.append(new_mutator.mutation(original, max_mutant_size=self.options["maximum_mutation_size"], min_depth=self.options["minimum_replacement_depth"]))
        if methods["keep_winners"]==True:
            if "num_elites" not in methods:
                next_generation+=winning_generation
            else:
                next_generation[:methods["num_elites"]]=elites
        return next_generation
    def evolve(self, frequencies, data, num_generations=20):
        self.generation_num=0
        current_generation=self.initial_generation
        if len(self.best_candidates["data"])==0:
            self.best_candidates["data"]=data
        for i in range(0, num_generations):
            self.generation_num+=1

            winners, winning_params= self.selection(current_generation, frequencies, data, self.options["selection"])
            current_generation=self.reproduce(winners, keep_winners=True, num_elites=1)
    def plot_generation(self, windex, loser_idx, scores,  sim_array, generation, data,axes, plot_count):
        idxes=[windex, loser_idx]
        titles=["Winner", "Loser"]
        dict_constructor=EIS()
        for i in range(0, len(idxes)):
            circuit_ax=axes[0][plot_count+i]
            idx=idxes[i]
            circuit_dict=dict_constructor.translate_tree(generation[idx])
            circuit_artist(circuit_dict, circuit_ax)
            circuit_ax.set_title(titles[i])
            circuit_ax.set_axis_off()
            data_ax=axes[1][plot_count+i]
            data_ax.plot(data[:,0], -data[:,1])
            sim_data=sim_array[idx]
            data_ax.plot(sim_data[:,0], -sim_data[:,1])
            data_ax.set_title(round(scores[idx], 3))
