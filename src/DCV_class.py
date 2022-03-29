from numpy import np
from matplotlib.pyplot import plt
from single_e_class_unified import single_electron
class DCVTrumpet(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):

        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "trumpet_method" not in self.simulation_options:
            self.simulation_options["trumpet_method"]="both"
        if "trumpet_test" not in self.simulation_options:
            self.simulation_options["trumpet_test"]=False
        if "synthetic_noise" not in self.simulation_options:
            self.simulation_options["synthetic_noise"]=0
        if "record_exps" not in self.simulation_options:
            self.simulation_options["record_exps"]=False
        if "omega" in dim_parameter_dictionary:
            if dim_parameter_dictionary["omega"]!=0:
                dim_parameter_dictionary["omega"]==0
                warnings.warn("Setting DCV frequency to 0, you idiot")
        if "find_in_range" not in self.simulation_options:
            self.simulation_options["find_in_range"]=False

        self.saved_sims={"current":[], "voltage":[]}

    def trumpet_positions(self, current, voltage):
        volts=np.array(voltage)
        amps=np.array(current)

        if self.simulation_options["find_in_range"]!=False:
            dim_volts=self.e_nondim(volts)
            search_idx=tuple(np.where((dim_volts>self.simulation_options["find_in_range"][0])&(dim_volts<self.simulation_options["find_in_range"][1])))
            #fig, ax=plt.subplots(1,1)
            #ax.plot(dim_volts, amps)
            volts=volts[search_idx]
            amps=amps[search_idx]

            #print(self.simulation_options["find_in_range"])
            #ax.plot(volts, amps)
            #plt.show()
        max_pos=volts[np.where(amps==max(amps))]
        min_pos=volts[np.where(amps==min(amps))]
        return max_pos, min_pos
    def n_outputs(self):
        if self.simulation_options["trumpet_method"]=="both":
            return 2
        else:
            return 1
    def trumpet_plot(self, scan_rates, trumpets, **kwargs):
        if "ax" not in kwargs:
            ax=None
        else:
            ax=kwargs["ax"]
        if "label" not in kwargs:
            label=None
        else:
            label=kwargs["label"]
        if "description" not in kwargs:
            kwargs["description"]=False
        else:
            label=None
        if "log" not in kwargs:
            kwargs["log"]=np.log10
        if len(trumpets.shape)!=2:
            raise ValueError("For plotting reductive and oxidative sweeps together")
        else:
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
            if ax==None:
                plt.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label="Forward sim", color=colors[0])
                plt.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], label="Reverse sim", color=colors[0], facecolors='none')
                plt.legend()
                plt.show()
            else:
                if kwargs["description"]==False:
                    if ax.collections:
                        pass
                    else:
                        self.color_counter=0
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label=label, color=colors[self.color_counter])
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], color=colors[self.color_counter], facecolors='none')
                    self.color_counter+=1
                else:
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label="$E_{p(ox)}-E^0$", color=colors[0])
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], label="$E_{p(red)}-E^0$", color=colors[0], facecolors='none')
    def poly_2(self, x, a, b, c):
        return (a*x**2)+b*x+c
    def poly_1(self, x, m,c):
        return m*x+c
    def k0_interpoltation(self, scans, trumpet, **units):
        scan_rates=np.log(scans)
        anodic_sweep=trumpet[:, 0]
        cathodic_sweep=trumpet[:, 1]
        data=[anodic_sweep, cathodic_sweep]
        if "T" not in units:
            units["T"]=278
        if "n" not in units:
            units["n"]=1
        if "E_0" not in units:

            units["E_0"]=(anodic_sweep[-1]-cathodic_sweep[-1])/2
            print(units["E_0"])
        if "deviation" not in units:
            deviation=100
        else:
            deviation=units["deviation"]
        if "plot" not in units:
            units["plot"]=False
        else:
            if "ax" not in units:
                units["ax"]=None
                fig, ax=plt.subplots(1,1)
            else:
                ax=units["ax"]
        units["F"]=96485.3329
        units["R"]=8.31446261815324
        nF=units["n"]*units["F"]
        RT=units["R"]*units["T"]
        difference=self.square_error(anodic_sweep, cathodic_sweep)
        mean_low_scans=np.mean(difference[:5])
        low_scan_deviation=np.divide(difference, mean_low_scans)
        start_scan=tuple(np.where(low_scan_deviation>deviation))
        inferred_k0=[0 for x in range(0, len(data))]
        for i in range(0, len(data)):
            fitting_scan_rates=scan_rates[start_scan]
            fitting_data=np.subtract(data[i][start_scan], units["E_0"])
            popt, _ = curve_fit(self.poly_1, fitting_scan_rates, fitting_data)
            m=popt[0]

            c=popt[1]
            #print(m,c)
            if i ==0:
                alpha=1-((RT)/(m*nF))
                exp=np.exp(c*nF*(1-alpha)/(RT))
                inferred_k0[i]=(nF*(1-alpha))/(RT*exp)
            else:
                alpha=-((RT)/(m*nF))
                exp=np.exp(c*nF*(-alpha)/(RT))
                inferred_k0[i]=(nF*(alpha))/(RT*exp)
            fitted_curve=[self.poly_1(t, *popt) for t in fitting_scan_rates]
            if units["plot"]==True:
                if i==0:
                    ax.plot(fitting_scan_rates, fitted_curve, color="red", label="Linear region")
                else:
                    ax.plot(fitting_scan_rates, fitted_curve, color="red")
        if units["plot"]==True:
            self.trumpet_plot(scans, np.column_stack((data[0]-units["E_0"], data[1]-units["E_0"])), ax=ax, log=np.log, description=True)
            plt.legend()
            fig=plt.gcf()
            fig.set_size_inches(7, 4.5)
            plt.show()
            fig.savefig("DCV_trumpet_demonstration.png", dpi=500)
        return inferred_k0
    def simulate(self, parameters, scan_rates):

        forward_sweep_pos=np.zeros(len(scan_rates))
        reverse_sweep_pos=np.zeros(len(scan_rates))
        for i in range(0, len(scan_rates)):

            self.dim_dict["v"]=scan_rates[i]
            self.nd_param=params(self.dim_dict)
            self.calculate_times()
            volts=self.define_voltages()
            current=super().simulate(parameters, [])
            if self.simulation_options["synthetic_noise"]!=0:
                current=self.add_noise(current, self.simulation_options["synthetic_noise"]*max(current))

                #current=self.rolling_window(current, 8)
            if self.simulation_options["record_exps"]==True:
                self.saved_sims["current"].append(current)
                self.saved_sims["voltage"].append(volts)
            forward_sweep_pos[i], reverse_sweep_pos[i]=self.trumpet_positions(current, volts)
        if "dcv_sep" in self.optim_list:
            forward_sweep_pos+=self.nd_param.nd_param_dict["dcv_sep"]
            reverse_sweep_pos-=self.nd_param.nd_param_dict["dcv_sep"]
        if self.simulation_options["trumpet_method"]=="both":
            if self.simulation_options["trumpet_test"]==True:
                print(parameters)
                log10_scans=np.log10(scan_rates)
                fig, ax=plt.sublots(1,1)
                ax.scatter(log10_scans, forward_sweep_pos)
                ax.scatter(log10_scans, reverse_sweep_pos)
                ax.scatter(log10_scans, self.secret_data_trumpet[:,0])
                ax.scatter(log10_scans, self.secret_data_trumpet[:,1])
                plt.show()
            return np.column_stack((forward_sweep_pos, reverse_sweep_pos))
        elif self.simulation_options["trumpet_method"]=="forward":
            return forward_sweep_pos
        else:
            return reverse_sweep_pos
