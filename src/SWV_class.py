from numpy import np
from matplotlib.pyplot import plt
from single_e_class_unified import single_electron
class SWV_PS(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):

        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "synthetic_noise" not in self.simulation_options:
            self.simulation_options["synthetic_noise"]=0
        if "record_exps" not in self.simulation_options:
            self.simulation_options["record_exps"]=False
        if "SWVtest" not in self.simulation_options:
            self.simulation_options["SWVtest"]=False
    def simulate(self, parameters, frequency_range):
        maxes=np.zeros(len(frequency_range))

        for j in range(0, len(frequency_range)):

            self.dim_dict["omega"]=frequency_range[j]
            self.nd_param=params(self.dim_dict)
            #self.calculate_times()
            #start=time.time()
            #volts=self.define_voltages()
            #start=time.time()

            current=super().simulate(parameters, [])
            f, b, net, potential=super().SW_peak_extractor(current)

            if self.simulation_options["synthetic_noise"]!=0:
                current=self.add_noise(net, self.simulation_options["synthetic_noise"]*max(net))
                #current=self.rolling_window(current, 8)
            else:
                current=net
            #plt.plot(potential, current)
                #current=self.rolling_window(current, 8)
            if self.simulation_options["record_exps"]==True:
                self.saved_sims["current"].append(current)
                self.saved_sims["voltage"].append(volts)
            first_half=tuple(np.where(potential<self.dim_dict["E_0"]))
            second_half=tuple(np.where(potential>self.dim_dict["E_0"]))
            data=[first_half, second_half]
            peak_pos=[0 ,0]
            for i in range(0, 2):
                maximum=max(current[data[i]])
                max_idx=potential[data[i]][np.where(current[data[i]]==maximum)]
                peak_pos[i]=max_idx
            maxes[j]=(peak_pos[1]-peak_pos[0])*1000
        if self.simulation_options["SWVtest"]==True:
            plt.scatter(1/frequency_range, maxes)
            plt.scatter(1/frequency_range, self.test)
            plt.show()
            #print(time.time()-start)
        return maxes
class SWV_QRM(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):

        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "synthetic_noise" not in self.simulation_options:
            self.simulation_options["synthetic_noise"]=0
        if "record_exps" not in self.simulation_options:
            self.simulation_options["record_exps"]=False
        if "SWVtest" not in self.simulation_options:
            self.simulation_options["SWVtest"]=False
        if "amplitudes_set" not in self.simulation_options:
            self.simulation_options["amplitudes_set"]=False
    def set_amplitudes(self, amplitudes):
        self.Esw_range=amplitudes
        self.simulation_options["amplitudes_set"]=True
    def simulate(self, parameters, frequency_range):
        start=time.time()
        maxes=np.zeros(len(frequency_range))
        if self.simulation_options["amplitudes_set"]!=True:
            raise ValueError("Need to define SW ampltidues")
        for j in range(0, len(frequency_range)):
            self.dim_dict["omega"]=frequency_range[j]
            delta_p=np.zeros(len(self.Esw_range))
            for q in range(0, len(self.Esw_range)):
                self.dim_dict["SW_amplitude"]=self.Esw_range[q]
                self.nd_param=params(self.dim_dict)
                #self.calculate_times()
                #start=time.time()
                #volts=self.define_voltages()
                current=super().simulate(parameters, [])
                f, b, net, potential=super().SW_peak_extractor(current)

                if self.simulation_options["synthetic_noise"]!=0:

                    current=self.add_noise(net, self.simulation_options["synthetic_noise"]*max(net))
                    #current=self.rolling_window(current, 8)
                else:
                    current=net

                #plt.plot(potential, current)
                    #current=self.rolling_window(current, 8)
                if self.simulation_options["record_exps"]==True:
                    self.saved_sims["current"].append(current)
                    self.saved_sims["voltage"].append(volts)
                delta_p[q]=(max(current))/self.Esw_range[q]
            #plt.show()
            maxes[j]=(self.Esw_range[np.where(delta_p==max(delta_p))])

        if self.simulation_options["SWVtest"]==True:
            plt.scatter(1/frequency_range, maxes)
            plt.scatter(1/frequency_range, self.test)
            plt.show()

        return maxes
class SWV_alpha(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):

        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "synthetic_noise" not in self.simulation_options:
            self.simulation_options["synthetic_noise"]=0
        if "record_exps" not in self.simulation_options:
            self.simulation_options["record_exps"]=False
        if "SWVtest" not in self.simulation_options:
            self.simulation_options["SWVtest"]=False
        if "amplitudes_set" not in self.simulation_options:
            self.simulation_options["amplitudes_set"]=False
    def simulate(self, parameters, frequency_range):
            self.nd_param=params(self.dim_dict)

            current=super().simulate(parameters, [])
            f, b, net, potential=super().SW_peak_extractor(current)
            if self.simulation_options["synthetic_noise"]!=0:
                print("adding noise")
                f=self.add_noise(f, self.simulation_options["synthetic_noise"]*max(f))
                b=self.add_noise(b, self.simulation_options["synthetic_noise"]*max(b))
            data=[max(abs(b)), max(abs(f))]
            return data
