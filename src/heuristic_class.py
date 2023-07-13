import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
import warnings
import time
from params_class import params
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from decimal import Decimal
from matplotlib.widgets import Slider, Button, RadioButtons
from numpy.lib.stride_tricks import sliding_window_view
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
        if "potentials" not in simulation_options:
            self.simulation_options["potentials"]={}

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
        if "colour_counter" not in kwargs:
            kwargs["cc"]=None
        else:
            kwargs["cc"]=kwargs["colour_counter"]
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
                    if kwargs["cc"] is None:
                        if ax.collections:
                            pass
                        else:
                            self.color_counter=0
                    else:
                        self.color_counter=kwargs["cc"]
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label=label, color=colors[self.color_counter])
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], color=colors[self.color_counter], facecolors='none')
                    if kwargs["cc"] is None:
                        self.color_counter+=1
                    
                else:
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 0], label="$E_{p(ox)}-E^0$", color=colors[0])
                    ax.scatter(kwargs["log"](scan_rates),  trumpets[:, 1], label="$E_{p(red)}-E^0$", color=colors[0], facecolors='none')
   
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
    def synthetic_noise(self, parameters, scan_rates, noise):
        self.simulation_options["synthetic_noise"]=noise
        data=self.simulate(parameters, scan_rates)
        self.simulation_options["synthetic_noise"]=0
        return data
    def simulate(self, parameters, scan_rates, optimise_flag=True):
        forward_sweep_pos=np.zeros(len(scan_rates))
        reverse_sweep_pos=np.zeros(len(scan_rates))
        sim_time=0
        
        for i in range(0, len(scan_rates)):
            
            self.dim_dict["v"]=scan_rates[i]
            self.nd_param=params(self.dim_dict)
            self.calculate_times()
            if optimise_flag==True:
                if scan_rates[i] not in self.simulation_options["potentials"]:
                    self.simulation_options["potentials"][scan_rates[i]]=self.define_voltages()
                volts=self.simulation_options["potentials"][scan_rates[i]]
            else:
                volts=self.define_voltages()
            start=time.time()
            current=super().simulate(parameters, [])

            #sim_time+=(time.time()-start)
            #print(time.time()-start, "sim_time", len(current))
            if self.simulation_options["synthetic_noise"]!=0:
                current=self.add_noise(current, self.simulation_options["synthetic_noise"]*max(current))

                #current=self.rolling_window(current, 8)
            if self.simulation_options["record_exps"]==True:
                self.saved_sims["current"].append(current)
                self.saved_sims["voltage"].append(volts)
            forward_sweep_pos[i], reverse_sweep_pos[i]=self.trumpet_positions(current, volts)
        #print("loop_time", time.time()-start1)
        #print("sim_time", sim_time)
        if "dcv_sep" in self.optim_list:
            forward_sweep_pos+=self.nd_param.nd_param_dict["dcv_sep"]
            reverse_sweep_pos-=self.nd_param.nd_param_dict["dcv_sep"]
        #print("func_time", time.time()-start1)
        if self.simulation_options["trumpet_method"]=="both":
            if self.simulation_options["trumpet_test"]==True:
                print(parameters)
                log10_scans=np.log10(scan_rates)
                fig, ax=plt.subplots(1,1)
                ax.scatter(log10_scans, forward_sweep_pos)
                ax.scatter(log10_scans, reverse_sweep_pos)
                ax.scatter(log10_scans, self.secret_data_trumpet[:,0])
                ax.scatter(log10_scans, self.secret_data_trumpet[:,1])
                plt.show()
            return np.column_stack((forward_sweep_pos, reverse_sweep_pos))
        elif self.simulation_options["trumpet_method"]=="concatenated":
            return np.append(forward_sweep_pos, reverse_sweep_pos)
        elif self.simulation_options["trumpet_method"]=="forward":
            return forward_sweep_pos
        else:
            return reverse_sweep_pos
class DCV_peak_area():
    
    def __init__(self, times, potential, current, area):
        
       self.times=times
       self.area=area
       self.current=current
       self.potential=potential
       self.func_switch={"2":self.poly_2, "3":self.poly_3, "4":self.poly_4}
       self.middle_idx=list(potential).index(max(potential))
    def poly_2(self, x, a, b, c):
        return (a*x**2)+b*x+c
    def poly_3(self, x, a, b, c, d):
        return (a*x**3)+(b*x**2)+c*x+d
    def poly_4(self, x, a, b, c, d, e):
        return (a*x**4)+(b*x**3)+(c*x**2)+d*x+e
    def background_subtract(self, params):
        func=self.func_switch[self.func_order]
        half_len=len(self.potential[self.middle_idx:])
        first_idx=np.where(self.potential>params[4])[0][0]
        second_idx=np.where(self.potential[:self.middle_idx]>params[5])[0][0]
        #fig, ax=plt.subplots()
        #ax.plot(self.potential, self.current)
        #ax.plot(self.potential[np.where(self.potential[:self.middle_idx]>params[5])], self.current[np.where(self.potential[:self.middle_idx]>params[5])])
        #plt.show()
        fourth_idx=half_len+np.where(self.potential[self.middle_idx:]<params[6])[0][0]
        third_idx=half_len+np.where(self.potential[self.middle_idx:]<params[7])[0][0]
        #print(len(self.potential), params[6], params[7])
        #print(first_idx, second_idx, third_idx, fourth_idx)
        idx_1=[first_idx, third_idx]
        idx_2=[second_idx, fourth_idx]
        interesting_section=[[params[0], params[1]], [params[2], params[3]]]
        current_results=self.current
        subtract_current=np.zeros(len(current_results))
        fitted_curves=np.zeros(len(current_results))
        nondim_v=self.potential
        time_results=self.times
        #plt.plot(self.potential, self.current)
        return_arg={}
        for i in range(0, 2):
            current_half=current_results[idx_1[i]:idx_2[i]]
            time_half=time_results[idx_1[i]:idx_2[i]]
            volt_half=self.potential[idx_1[i]:idx_2[i]]
            noise_idx=np.where((volt_half<interesting_section[i][0]) | (volt_half>interesting_section[i][1]))
            signal_idx=np.where((volt_half>interesting_section[i][0]) & (volt_half<interesting_section[i][1]))
            noise_voltages=volt_half[noise_idx]
            noise_current=current_half[noise_idx]
            noise_times=time_half[noise_idx]

            popt, pcov = curve_fit(func, noise_times, noise_current)
            fitted_curve=[func(t, *popt) for t in time_half]
            return_arg["poly_{0}".format(i)]=[volt_half, fitted_curve]
            
            #plt.plot(volt_half, fitted_curve, color="red")
            sub_c=np.subtract(current_half, fitted_curve)
            subtract_current[idx_1[i]:idx_2[i]]=sub_c
            fitted_curves[idx_1[i]:idx_2[i]]=fitted_curve
            #plt.plot(volt_half[signal_idx], sub_c[signal_idx])
            area=simpson(sub_c[signal_idx], time_half[signal_idx])
            return_arg["bg_{0}".format(i)]=[noise_voltages, noise_current]
            return_arg["subtract_{0}".format(i)]=[volt_half[signal_idx], sub_c[signal_idx]]
            gamma=abs(area/(self.area*96485.3321))
            return_arg["gamma_{0}".format(i)]="{:.3E}".format(Decimal(gamma))
        return return_arg
            #print(gamma)
    #def update_lines(self, poly_lines, subtracted_lines, params):
    #    get_vals=self.background_subtract(params)
    #    for i in range(0,2):
    #        print(poly_lines)
    #        print(subtracted_lines)
    #        poly_lines[i].set_data(get_vals["poly_{0}".format(i)][0], get_vals["poly_{0}".format(i)][1])
    #        subtracted_lines[i].set_data(get_vals["subtract_{0}".format(i)][0], get_vals["subtract_{0}".format(i)][1])
    def update(self, value):
        params=[self.slider_array[key].val for key in self.slider_array.keys() ] 
        get_vals=self.background_subtract(params)
        txt=["f", "b"]
        for i in range(0,2):
            self.red_lines[i].set_data(get_vals["poly_{0}".format(i)][0], get_vals["poly_{0}".format(i)][1])
            self.subtracted_lines[i].set_data(get_vals["subtract_{0}".format(i)][0], get_vals["subtract_{0}".format(i)][1])
            self.gamma_text[i].set_text("$\\Gamma_"+txt[i]+"="+get_vals["gamma_{0}".format(i)]+"$ mol cm$^{-2}$")
            if self.show_bg==True:

                self.bg_lines[i].set_data(get_vals["bg_{0}".format(i)][0], get_vals["bg_{0}".format(i)][1])
            else:
                self.bg_lines[i].set_data(0,0)
    def draw_background_subtract(self,):
        fig, ax=plt.subplots()
        
        line, = ax.plot(self.potential, self.current, lw=2)
        self.red_lines=[ax.plot(0,0, color="red")[0] for x in range(0, 2)]
        self.subtracted_lines=[ax.plot(0,0, color="black", linestyle="--")[0] for x in range(0, 2)]
        self.bg_lines=[ax.plot(0,0, color="orange")[0] for x in range(0, 2)]
        fig.subplots_adjust(left=0.1, bottom=0.35, right=0.75)
        params=["Ox start", "Ox end", "Red start", "Red end", "Forward start", "Forward end", "Reverse start", "Reverse end"]
        init_e0=(min(self.potential)+max(self.potential))/2
        init_e0_pc=0.1
        init_start=init_e0-init_e0_pc
        init_end=init_e0+init_e0_pc
        val=0.05
        init_param_values=[init_start, init_end, init_start, init_end, min(self.potential)+val, max(self.potential)-val, min(self.potential)+val, max(self.potential)-val]
        

        param_dict=dict(zip(params, init_param_values))
        interval=0.25/8
        resetax = plt.axes([0.8, 0.8, 0.1, 0.04])
        text_ax= plt.axes([0.76, 0.4, 0.1, 0.04])
        self.button = Button(resetax, 'Reset',  hovercolor='0.975')
        self.button.on_clicked(self.reset)
        polyax = plt.axes([0.8, 0.625, 0.1, 0.15])
        self.radio2 = RadioButtons(
        polyax, ('2', '3', '4'),
        )
        polyax = plt.axes([0.8, 0.475, 0.1, 0.15])
        self.radio1 = RadioButtons(
        polyax, ("Show BG", "Hide BG"),
        )
        self.show_bg=True
        self.func_order="3"
        txt=["f", "b"]
        get_vals=self.background_subtract(init_param_values)
        self.gamma_text=[text_ax.text(0.0, 1-(i*1), "$\\Gamma_"+txt[i]+"="+get_vals["gamma_{0}".format(i)]+"$ mol cm$^{-2}$") for i in range(0, 2)]
        text_ax.set_axis_off()
        
        self.radio2.on_clicked(self.radio_button)
        self.radio1.on_clicked(self.show_bg_func)
        for radios in [self.radio2, self.radio1]:
            for circle in radios.circles:
                circle.set_radius(0.09)
        self.slider_array={}
        class_array={}
        for i in range(0, len(params)):
            axfreq = fig.add_axes([0.2, 0.25-(i*interval), 0.65, 0.03])
            self.slider_array[params[i]] = Slider(
                ax=axfreq,
                label=params[i],
                valmin=min(self.potential),
                valmax=max(self.potential),
                valinit=init_param_values[i],
            )
            self.slider_array[params[i]].on_changed(self.update)
        self.update(init_param_values)
        fig.set_size_inches(9, 6)
    def reset(self, event):
        [self.slider_array[key].reset() for key in self.slider_array.keys() ] 
    def radio_button(self, value):
        self.func_order=value
    def show_bg_func(self, value):
        if value=="Show BG":
            self.show_bg=True
        else:
            self.show_bg=False
        
class Laviron_EIS(single_electron):
    def __init__(self, dim_parameter_dictionary={}, simulation_options={}, other_values={}, param_bounds={}):
        self.F=96485.3321
        self.R=8.3145
        self.T=298
        self.FRT=self.F/(self.R*self.T)
        
        if len(dim_parameter_dictionary)==0 and len(simulation_options)==0:
            self.simulation_options={"data_representation":"nyquist"}
            self.dim_dict={}
            return
        from EIS_class import EIS
        EIS_Cs=["EIS_Cdl", "EIS_Cf"]
        for i in range(0, len(EIS_Cs)):
            if EIS_Cs[i] not in simulation_options:
                simulation_options[EIS_Cs[i]]="C{0}".format(i+1)
            elif simulation_options[EIS_Cs[i]]=="C":
                simulation_options[EIS_Cs[i]]="C{0}".format(i+1)
            elif simulation_options[EIS_Cs[i]]=="CPE":
                simulation_options[EIS_Cs[i]]=("Q{0}".format(i+1), "alpha{0}".format(i+1))
            else:
                raise ValueError("{0} needs to be either C (capacitor) or CPE (constant phase element)".format(EIS_Cs[i]))
        if "data_representation" not in simulation_options:
            simulation_options["data_representation"]="nyquist"  
        elif simulation_options["data_representation"]=="bode":
            if "bode_split" not in simulation_options:
                simulation_options["bode_split"]=None
            
        if "DC_pot" not in simulation_options:
            raise ValueError("Please define a DC EIS potential")
     
        self.Laviron_circuit={"z1":"R0", "z2":{"p1":simulation_options["EIS_Cdl"], "p2":["R1", simulation_options["EIS_Cf"]]}}
        self.simulator=EIS(circuit=self.Laviron_circuit, invert_imaginary=simulation_options["invert_imaginary"])
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
    def n_outputs(self):
        if self.simulation_options["bode_split"]==None:
            return 2     
        else:
            return 1
    def synthetic_noise(self, parameters, frequencies, noise, flag="proportional"):

        sim=self.simulate(parameters, frequencies)
        if flag=="proportional":
            return_arg=np.column_stack((self.add_noise(sim[:,0], noise, method="proportional"), self.add_noise(sim[:,1], noise, method="proportional")))
        else:
            return_arg=np.column_stack((self.add_noise(sim[:,0], noise*np.mean(sim[:,0])), self.add_noise(sim[:,1], noise*np.mean(-sim[:,1]))))   
        
        return return_arg
    def clean_simulate(self,params,frequencies, **kwargs):
        from EIS_class import EIS
        self.optim_list=[]
        self.simulation_options["label"]="MCMC"
        self.simulation_options["test"]=False
        if "data_representation" in kwargs:
            self.simulation_options["data_representation"]=kwargs["data_representation"]
       
        EIS_Cs=["EIS_Cdl", "EIS_Cf"]
        for i in range(0, len(EIS_Cs)):
            if EIS_Cs[i] not in kwargs:
                kwargs[EIS_Cs[i]]="C{0}".format(i+1)
            elif kwargs[EIS_Cs[i]]=="C":
                kwargs[EIS_Cs[i]]="C{0}".format(i+1)
            elif kwargs[EIS_Cs[i]]=="CPE":
                kwargs[EIS_Cs[i]]=("Q{0}".format(i+1), "alpha{0}".format(i+1))
                if i==1 and "cpe_alpha_faradaic" not in params:
                    raise ValueError("Need a cpe_alpha_faradaic param")
                else:
                    self.dim_dict["cpe_alpha_faradaic"]=params["cpe_alpha_faradaic"]
                if i==0 and "cpe_alpha_cdl" not in params:
                    raise ValueError("Need a cpe_alpha_cdl param")
                else:
                    self.dim_dict["cpe_alpha_cdl"]=params["cpe_alpha_cdl"]
               
            else:
                raise ValueError("{0} needs to be either C (capacitor) or CPE (constant phase element)".format(EIS_Cs[i]))
            self.simulation_options[EIS_Cs[i]]= kwargs[EIS_Cs[i]]
        self.Laviron_circuit={"z1":"R0", "z2":{"p1":kwargs["EIS_Cdl"], "p2":["R1", kwargs["EIS_Cf"]]}}
        self.simulator=EIS(circuit=self.Laviron_circuit, invert_imaginary=False)
        self.simulation_options["DC_pot"]=params["DC_pot"]
        for key in ["k_0", "E_0", "alpha", "gamma","area", "Ru", "Cdl"]:
            self.dim_dict[key]=params[key]
        return self.simulate([], frequencies)
    def simulate(self, parameters, frequencies, print_circuit_params=False):
        if self.simulation_options["label"]=="cmaes":
            params=self.change_norm_group(parameters, "un_norm")
        else:
            params=parameters
        for i in range(0, len(self.optim_list)):

            self.dim_dict[self.optim_list[i]]=params[i]
        
        k0=self.dim_dict["k_0"]
        e0=self.dim_dict["E_0"]
        alpha=self.dim_dict["alpha"]
       
        gamma=self.dim_dict["gamma"]
        area=self.dim_dict["area"]
        dc_pot=self.simulation_options["DC_pot"]
        ratio=np.exp(self.FRT*(e0-dc_pot))
        ox=gamma/(ratio+1)
        red=gamma-ox
        Ra_coeff=(self.R*self.T)/((self.F**2)*area*k0)
        nu_1_alpha=np.exp((1-alpha)*self.FRT*(dc_pot-e0))
        nu_alpha=np.exp((-alpha)*self.FRT*(dc_pot-e0))
        Ra=Ra_coeff*((alpha*ox*nu_alpha)+((1-alpha)*red*nu_1_alpha))**-1
        sigma=k0*Ra*(nu_alpha+nu_1_alpha)
        Cf=1/sigma

        EIS_params={}
        EIS_params["R0"]=self.dim_dict["Ru"]
        EIS_params["R1"]=Ra#self.dim_dict["R2"]
        
        if self.simulation_options["EIS_Cdl"]=="C1":
            EIS_params["C1"]=self.dim_dict["Cdl"]
        elif self.simulation_options["EIS_Cdl"]==("Q1", "alpha1"):
            EIS_params["Q1"]=self.dim_dict["Cdl"]
            EIS_params["alpha1"]=self.dim_dict["cpe_alpha_cdl"]
        if self.simulation_options["EIS_Cf"]=="C2":
            EIS_params["C2"]=Cf
        elif self.simulation_options["EIS_Cf"]==("Q2", "alpha2"):
            cpe_alpha=abs(1-self.dim_dict["cpe_alpha_faradaic"])
            EIS_params["Q2"]=1/(Cf**(cpe_alpha-1)*Ra**(-cpe_alpha))
            EIS_params["alpha2"]=self.dim_dict["cpe_alpha_faradaic"]
        if print_circuit_params==True:
            print(EIS_params)
        #print(EIS_params)
        #.print(Ra_coeff)
        #EIS_params={'R0': 5, 'C1': 1e-06, 'R1': 59.27316911806477, 'C2': 2.4156737037803954e-05}
        Z_vals=self.simulator.test_vals(EIS_params, frequencies)
        
        if self.simulation_options["test"]==True:
            fig, ax=plt.subplots()
            self.simulator.nyquist(Z_vals, ax=ax, orthonormal=False, s=1)
            self.simulator.nyquist(self.secret_data_EIS, ax=ax, orthonormal=False, s=1)
            plt.show()
        if self.simulation_options["data_representation"]=="nyquist": 
            return Z_vals
        elif self.simulation_options["data_representation"]=="bode":
            return_arg=self.simulator.convert_to_bode(Z_vals)
            if "phase" in self.optim_list:
                return_arg[:,0]+=self.dim_dict["phase"]
            if self.simulation_options["bode_split"]!=None:
                if self.simulation_options["bode_split"]=="phase":
                    return return_arg[:,0]
                elif self.simulation_options["bode_split"]=="magnitude":
                    return return_arg[:,1]
            else:
               
                return return_arg
class PSV_harmonic_minimum(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        simulation_options["method"]="sinusoidal"
        simulation_options["psv_copying"]=True
        dim_parameter_dictionary["num_peaks"]=20
        simulation_options["no_transient"]=1/dim_parameter_dictionary["omega"]
        if "synthetic_noise" not in simulation_options:
            simulation_options["synthetic_noise"]=0
        if "num_steps" not in simulation_options:
            simulation_options["num_steps"]=100
        if "E_step_range" not in simulation_options:
            simulation_options["E_step_range"]=100e-3
        if "E_step_start" not in simulation_options:
            raise ValueError("Please define the initial step")
        simulation_options["max_even_harm"]=other_values["harmonic_range"][-1]//2
        simulation_options["even_harms"]=np.multiply(range(1, simulation_options["max_even_harm"]+1), 2)
        simulation_options["num_even_harms"]=len(simulation_options["even_harms"])
        simulation_options["E_step"]=simulation_options["E_step_range"]/simulation_options["num_steps"]
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        self.dim_dict["Cdl"]=0
        for i in range(1, 4):
            self.dim_dict["CdlE{0}".format(i)]=0
    def get_even_amplitudes(self, time, current, rolling_window=False):
        f=np.fft.fftfreq(len(time), time[1]-time[0])
        Y=np.fft.fft(current)
        abs_Y=abs(Y)
        #plt.plot(f, abs_Y)
        #plt.show()
        get_primary_harm=abs(max(f[np.where(abs_Y==max(abs_Y))]))
        box_width=0.05
        even_harms=self.simulation_options["even_harms"]
        return_arg=np.zeros(self.simulation_options["num_even_harms"])
        for i in range(0, self.simulation_options["num_even_harms"]):
            return_arg[i]=max(abs_Y[np.where((f>even_harms[i]*(1-box_width)) &(f<even_harms[i]*(1+box_width)))])
        return return_arg
    def synthetic_noise(self, parameters, scan_rates, noise):
        self.simulation_options["synthetic_noise"]=noise
        data=self.simulate(parameters, scan_rates, rolling_window=True)
        self.simulation_options["synthetic_noise"]=0
        return data
    def rolling_mean(self, x, window):
        
        return np.convolve(x, np.ones(window)/window, mode='same')


    def simulate(self, parameters, frequencies, rolling_window=False):
        magnitudes=np.zeros((self.simulation_options["num_even_harms"], self.simulation_options["num_steps"]))
        E_start_vals=self.simulation_options["E_step_start"]+ np.multiply(self.simulation_options["E_step"],np.arange(0, self.simulation_options["num_steps"]))
        for i in range(0, self.simulation_options["num_steps"]):
            self.dim_dict["E_start"]=E_start_vals[i]#(i*self.simulation_options["E_step"])

            current=super().simulate(parameters, frequencies)
            if self.simulation_options["synthetic_noise"]!=0:
                current=self.add_noise(current, self.simulation_options["synthetic_noise"]*max(current))
            #plt.plot(self.simulation_times, current)
            #plt.show()
            magnitudes[:, i]=self.get_even_amplitudes(self.simulation_times, current)
        minima=np.zeros(self.simulation_options["num_even_harms"])
        
        for i in range(0, self.simulation_options["num_even_harms"]):
            if rolling_window==True:
                magnitudes[i,:]=self.rolling_mean(magnitudes[i,:], window=4)
            
            minima[i]=E_start_vals[np.where(magnitudes[i,:]==min(magnitudes[i,:]))]/self.nd_param.c_E0
        return minima
        