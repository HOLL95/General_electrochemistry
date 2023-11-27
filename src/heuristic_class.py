import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
import warnings
import time
import os
import re
from params_class import params
from scipy.optimize import curve_fit
from scipy.integrate import simpson
from dispersion_class import dispersion
from decimal import Decimal
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons
from numpy.lib.stride_tricks import sliding_window_view
from EIS_class import EIS
class DCVTrumpet(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        if simulation_options["method"]!="dcv":
            raise ValueError("Trumpet plots are a DCV experiment")
        if simulation_options["likelihood"]!="timeseries":
            raise ValueError("Trumpet plots take place in the time domain")
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
        if "trumpet_method" not in self.simulation_options:
            self.simulation_options["trumpet_method"]="both"
        if "trumpet_test" not in self.simulation_options:
            self.simulation_options["trumpet_test"]=False
        if "synthetic_noise" not in self.simulation_options:
            self.simulation_options["synthetic_noise"]=0
        if "record_exps" not in self.simulation_options:
            self.simulation_options["record_exps"]=False
        if "invert_positions" not in self.simulation_options:
            self.simulation_options["invert_positions"]=False
        if "omega" in dim_parameter_dictionary:
            if dim_parameter_dictionary["omega"]!=0:
                dim_parameter_dictionary["omega"]==0
                warnings.warn("Setting DCV frequency to 0, you idiot")
        if "find_in_range" not in self.simulation_options:
            self.simulation_options["find_in_range"]=False
        if "potentials" not in simulation_options:
            self.simulation_options["potentials"]={}

        self.saved_sims={"current":[], "voltage":[]}
    def trumpet_positions(self, current, voltage, dim_flag=True):
        volts=np.array(voltage)
        amps=np.array(current)

        if self.simulation_options["find_in_range"]!=False:
            if dim_flag==True:
                dim_volts=self.e_nondim(volts)
            else:
                dim_volts=voltage
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
                plt.xlabel("Log scan rate")
                plt.ylabel("Peak position (V)")
                plt.legend()
                plt.show()
            else:
                ax.set_xlabel("Log scan rate")
                ax.set_ylabel("Peak position (V)")
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
    def get_all_voltages(self, scan_rates):
        potentials=[]
        times=[]
        for i in range(0, len(scan_rates)):
            
            self.dim_dict["v"]=scan_rates[i]
            self.nd_param=params(self.dim_dict)
            self.calculate_times()
            
            potentials.append(self.define_voltages())
            times.append(self.t_nondim(self.time_vec))
        return np.array(times),  np.array(potentials)
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
                print(self.change_norm_group(parameters, "un_norm"))
                log10_scans=np.log10(scan_rates)
                fig, ax=plt.subplots(1,1)
                ax.scatter(log10_scans, forward_sweep_pos)
                ax.scatter(log10_scans, reverse_sweep_pos)
                ax.scatter(log10_scans, self.secret_data_trumpet[:,0])
                ax.scatter(log10_scans, self.secret_data_trumpet[:,1])
                plt.show()
            if self.simulation_options["invert_positions"]==True:
                return np.column_stack((reverse_sweep_pos, forward_sweep_pos))
            else:
                return np.column_stack((forward_sweep_pos, reverse_sweep_pos))
        elif self.simulation_options["trumpet_method"]=="concatenated":
            return np.append(forward_sweep_pos, reverse_sweep_pos)
        elif self.simulation_options["trumpet_method"]=="forward":
            return forward_sweep_pos
        else:
            return reverse_sweep_pos

        
class DCV_peak_area():
    
    def __init__(self, times, potential, current, area, **kwargs):
        
        self.times=times
        self.area=area
        self.current=current
        self.potential=potential
        self.func_switch={"1":self.poly_1, "2":self.poly_2, "3":self.poly_3, "4":self.poly_4}
        self.all_plots=["total", "subtract", "bg", "poly"]
        self.check_status=dict(zip(self.all_plots, [False]*len(self.all_plots)))
        self.vlines=[]
        self.scatter_points=[]
        self.peak_positions=[None, None]
        self.show_peaks="Lines"
        if "func_order" not in kwargs:
            func_order=3
        else:
            func_order=kwargs["func_order"]
        if "position_list" not in kwargs:
            warnings.warn("No peak position list found")
            self.position_list={}
        elif isinstance(kwargs["position_list"], dict) is False:
            raise ValueError("Position list needs to be a dictionary (defined by {})")
        else:
            self.position_list=kwargs["position_list"]
        if "data_filename" not in kwargs:
            self.data_filename=None
        else:
            self.data_filename=kwargs["data_filename"]
        
        if "scan_rate" not in kwargs:
            self.scan_rate=None
        else:
            self.scan_rate=kwargs["scan_rate"]
        self.func_order=func_order
        self.middle_idx=list(potential).index(max(potential))
    def poly_1(self, x, a, b):
        return a*x+b
    def poly_2(self, x, a, b, c):
        return (a*x**2)+b*x+c
    def poly_3(self, x, a, b, c, d):
        return (a*x**3)+(b*x**2)+c*x+d
    def poly_4(self, x, a, b, c, d, e):
        return (a*x**4)+(b*x**3)+(c*x**2)+d*x+e
    def background_subtract(self, params):
        #1Lower intersting section sweep1
        #2Upper interesting section sweep1
        #3Lower intersting section sweep2
        #4Upper interesting section sweep2
        #5 start of first sweep
        #6end of first sweep
        #7 start of second sweep
        #8end of second sweep
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
            #plt.plot(volt_half, current_half)
            #print(volt_half)
            #print(interesting_section[i][0], interesting_section[i][1])
            noise_idx=np.where((volt_half<interesting_section[i][0]) | (volt_half>interesting_section[i][1]))
            signal_idx=np.where((volt_half>interesting_section[i][0]) & (volt_half<interesting_section[i][1]))
            noise_voltages=volt_half[noise_idx]
            noise_current=current_half[noise_idx]
            noise_times=time_half[noise_idx]
            #plt.plot(noise_voltages, noise_current)
            #plt.show()
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
    def get_slider_vals(self,):
        params=[self.slider_array[key].val for key in self.slider_array.keys()] 
        if params[0]<params[4]:
            params[0]=params[4]
        if params[1]>params[5]:
            params[1]=params[5]
        if params[2]<params[6]:
            params[2]=params[6]
        if params[3]>params[7]:
            params[3]=params[7]
        return params
    def update(self, value):
        #1Lower intersting section sweep1
        #2Upper interesting section sweep1
        #3Lower intersting section sweep2
        #4Upper interesting section sweep2
        #5 start of first sweep
        #6end of first sweep
        #7 start of second sweep
        #8end of second sweep

        params=self.get_slider_vals()
        get_vals=self.background_subtract(params)
        txt=["f", "b"]
        plot_dict=dict(zip(self.all_plots[1:], [self.subtracted_lines, self.bg_lines,self.red_lines,]))
        
        if self.check_status["total"]==True:
            self.total_line.set_data(0,0)
        else:
            self.total_line.set_data(self.potential, self.current)
        for elem in self.scatter_points:
            elem.remove()
        for elem in self.vlines:
            elem.remove()
        self.vlines=[]
        self.scatter_points=[]
        for i in range(0,2):
            
            for key in plot_dict.keys():
                if self.check_status[key]==True:
                     plot_dict[key][i].set_data(0,0)
                else:
                    plot_dict[key][i].set_data(get_vals["{0}_{1}".format(key, i)][0],get_vals["{0}_{1}".format(key, i)][1])
                if key=="subtract":
                    current=get_vals["{0}_{1}".format(key, i)][1]
                    potential=get_vals["{0}_{1}".format(key, i)][0]
                    abs_current=abs(current)
                    current_max=max(abs_current)
                    loc=np.where(abs_current==current_max)
                    potential_max=potential[loc][0]
                    actual_current_max=current[loc]
                    self.peak_positions[i]=potential_max
                    if self.show_peak_position!="Hide":
                       
                        
                        if self.show_peaks=="Points":

                            self.scatter_points.append(self.all_Ax.scatter(potential_max, actual_current_max, s=100,color="purple", marker="x"))
                        elif self.show_peaks=="Lines":
                            self.vlines.append(self.all_Ax.axvline(potential_max, color="purple", linestyle="--"))
            self.gamma_text[i].set_text("$\\Gamma_"+txt[i]+"="+get_vals["gamma_{0}".format(i)]+"$ mol cm$^{-2}$")
            
            self.all_Ax.relim()
            self.all_Ax.autoscale_view()
    def draw_background_subtract(self,**kwargs):
        
        fig, ax=plt.subplots()
        self.all_Ax=ax
        self.total_line, = ax.plot(self.potential, self.current, lw=2)

        self.red_lines=[ax.plot(0,0, color="red")[0] for x in range(0, 2)]
        self.subtracted_lines=[ax.plot(0,0, color="black", linestyle="--")[0] for x in range(0, 2)]
        self.bg_lines=[ax.plot(0,0, color="orange")[0] for x in range(0, 2)]
        fig.subplots_adjust(left=0.1, bottom=0.35, right=0.605)
        params=["Ox start", "Ox end", "Red start", "Red end", "Forward start", "Forward end", "Reverse start", "Reverse end"]
        init_e0=(min(self.potential)+max(self.potential))/2
        init_e0_pc=0.1
        init_start=init_e0-init_e0_pc
        init_end=init_e0+init_e0_pc
        val=0.05
        if "init_vals" not in kwargs:
            kwargs["init_vals"]=None
        if kwargs["init_vals"]==None:
            init_param_values=[init_start, init_end, init_start, init_end, min(self.potential)+val, max(self.potential)-val, min(self.potential)+val, max(self.potential)-val]
        else:
            init_param_values=kwargs["init_vals"]
      
        param_dict=dict(zip(params, init_param_values))
        interval=0.25/8
        titleax = plt.axes([0.65, 0.84, 0.1, 0.04])
        titleax.set_title("BG subtraction")
        titleax.set_axis_off()
        resetax = plt.axes([0.65, 0.82, 0.1, 0.04])
        text_ax= plt.axes([0.61, 0.4, 0.1, 0.04])
        self.button = Button(resetax, 'Reset',  hovercolor='0.975')
        self.button.on_clicked(self.reset)
        polyax = plt.axes([0.65, 0.625, 0.1, 0.15])
        self.radio2 = RadioButtons(
        polyax, ('2', '3', '4'),
        )
        #polyax = plt.axes([0.65, 0.475, 0.1, 0.15])
        #self.radio1 = RadioButtons(
        #polyax, ("Show BG", "Hide BG"),
        #)
        self.show_bg=True
        self.func_order="2"
        txt=["f", "b"]
        get_vals=self.background_subtract(init_param_values)
        self.gamma_text=[text_ax.text(0.0, 1-(i*1), "$\\Gamma_"+txt[i]+"="+get_vals["gamma_{0}".format(i)]+"$ mol cm$^{-2}$") for i in range(0, 2)]
        text_ax.set_axis_off()
        

        kinetic_title_ax= plt.axes([0.85, 0.84, 0.1, 0.04])
        kinetic_title_ax.set_title("Peak position")
        kinetic_title_ax.set_axis_off()
        #save_Ax = plt.axes([0.84, 0.82, 0.12, 0.04])
        #self.button = Button(save_Ax, 'Save position',  hovercolor='0.975')

        axbox = plt.axes([0.84, 0.82, 0.12, 0.04])
        axbox.text(0.22, 1.2,"Scan rate", transform=axbox.transAxes)
        self.text_box = TextBox(axbox, "", textalignment="center")
        if self.data_filename is not None:
            self.all_Ax.set_title(self.data_filename)
        if self.scan_rate is not None:
            self.text_box.set_val(self.scan_rate)
        elif self.data_filename is not None:
            
            lower_name=self.data_filename.lower()
            
            if "mv" in lower_name:
               
                match=re.findall("\d+(?:\.\d+)?(?=mv)", lower_name)
                if len(match)==1:
                    self.text_box.set_val(match[0])
        hideax= plt.axes([0.85, 0.625, 0.12, 0.15])
        self.check = CheckButtons(
            ax=hideax,
            labels=("Hide total", "Hide sub", "Hide BG", "Hide poly"),
            

        )
        polyax = plt.axes([0.85, 0.475, 0.12, 0.15])
        self.radio3 = RadioButtons(
        polyax, ("Lines", "Points","Hide"),
        )
        self.save_text=axbox.text(0, -0.6, "Not saved", transform=axbox.transAxes)

        self.text_box.on_submit(self.submit_scanrate)
        self.radio2.on_clicked(self.radio_button)
        self.check.on_clicked(self.hiding)
        #self.radio1.on_clicked(self.show_bg_func)
        self.radio3.on_clicked(self.show_peak_position)
        for element in [self.radio2, self.check, self.radio3]:
            element.on_clicked(self.update)
        for radios in [self.radio2]:
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
        fig.set_size_inches(9, 8)
    def reset(self, event):
        [self.slider_array[key].reset() for key in self.slider_array.keys()] 
    def radio_button(self, value):
        self.func_order=value
    def hiding(self, label):
        self.check_status=dict(zip(self.all_plots, self.check.get_status()))
    def show_peak_position(self, value):
        self.show_peaks=value
    def submit_scanrate(self,expression):
        
        try:
            key=float(expression)
            self.position_list[key]=self.peak_positions
            self.save_text.set_text("{0}mV added".format(key))
        except:
            self.save_text.set_text("Not a scan rate")
    def get_scale_dict(self):
        return self.position_list

class Automated_trumpet(DCV_peak_area):
    def __init__(self, file_list, trumpet_file_name, **kwargs):
        if "filetype" not in kwargs:
            kwargs["filetype"]="Ivium"
        if kwargs["filetype"]!="Ivium":
            raise ValueError("Other files not supported")
        if "skiprows" not in kwargs:
            kwargs["skiprows"]=0
        if "data_loc" not in kwargs:
            kwargs["data_loc"]=""
        if "area" not in kwargs:
            raise ValueError("Need to define an area")
        update_file_list, scan_rates=self.sort_file_list(file_list)
        if isinstance(update_file_list, bool) is False:
            file_list=update_file_list
        scale_dict={}
        for i in range(0,len(file_list)):
            DCV_data=np.loadtxt(kwargs["data_loc"]+file_list[i], skiprows=kwargs["skiprows"])
            time=DCV_data[:,0]
            current=DCV_data[:,2]
            potential=DCV_data[:,1]
            if isinstance(scan_rates, bool) is False:
                scan_arg=scan_rates[i]
            else:
                scan_arg=None
            if i==0:
                init_vals=None
            super().__init__(time,potential, current, kwargs["area"], data_filename=file_list[i], position_list=scale_dict, scan_rate=scan_arg)
            self.draw_background_subtract(init_vals=init_vals)
            plt.show()
            
            scale_dict=self.get_scale_dict()
            init_vals=self.get_slider_vals()
        key_list=sorted([int(x) for x in scale_dict.keys()])
        trumpet_file=open(trumpet_file_name, "w")
        for key in key_list:
            
            line=(",").join([str(key), str(scale_dict[key][0]), str(scale_dict[key][1])])+"\n"
            trumpet_file.write(line)
        trumpet_file.close()
    def sort_file_list(self, file_list):
        element_list=[]
        scan_list=[]
        split_list_list=[]
        for i in range(0, len(file_list)):
            filename=file_list[i].lower()
            mv_has_scan=True
            if "mv" in filename:
                try:
                    match=re.findall("\d+(?:\.\d+)?(?=mv)", filename)[0]
                    scan_list.append(float(match))
                except:
                    mv_has_scan=False
            elif "mv" not in filename or mv_has_scan==False: 
                split_list=re.split(r"[\s.;_-]+", filename)
                split_list_list.append(split_list)
                new_list=[]
                for element in split_list:
                    try:
                        new_list.append(float(element))
                    except:
                        continue
                element_list.append(new_list)
        if len(scan_list)!=len(file_list):
            for i in range(0, len(element_list[0])):
                column=[element_list[x][i] for x in range(0, len(element_list))]
                if len(column)!=len(set(column)):
                    continue
                else:
                    maximum=max(column)
                    minimum=min(column)
                    if np.log10(maximum-minimum)<2:
                        continue
                    else:
                        scan_list+=column
        if len(scan_list)!=len(file_list):
            
            print("Have not been able to automatically sort the files. If you want this to work, either add `mv` after the scan rate in the filename, or have a consistent naming scheme")
            return False,False
        else:
            sorted_idx=np.argsort(scan_list)
            return_list=[file_list[x] for x in sorted_idx]
            return return_list, sorted(scan_list)

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
            simulation_options["bode_split"]=None
        elif simulation_options["data_representation"]=="bode":
            if "bode_split" not in simulation_options:
                simulation_options["bode_split"]=None
        elif simulation_options["data_representation"]=="nyquist":
            simulation_options["bode_split"]=None
        if "DC_pot" not in simulation_options:
            raise ValueError("Please define a DC EIS potential")
        if "Rct_only" not in simulation_options:
            simulation_options["Rct_only"]=False
        if "invert_imaginary" not in simulation_options:
            simulation_options["invert_imaginary"]=False
        self.Laviron_circuit={"z1":"R0", "z2":{"p1":simulation_options["EIS_Cdl"], "p2":["R1", simulation_options["EIS_Cf"]]}}
        self.simulator=EIS(circuit=self.Laviron_circuit, invert_imaginary=simulation_options["invert_imaginary"])
        if "v" not in dim_parameter_dictionary or "original_omega" not in dim_parameter_dictionary:
            dim_parameter_dictionary["v"]=1
            dim_parameter_dictionary["omega"]=1
        if "E_start" not in dim_parameter_dictionary:
            dim_parameter_dictionary["E_start"]=-10e-3
            dim_parameter_dictionary["E_reverse"]=10e-3
            
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
    def n_outputs(self):
        if self.simulation_options["bode_split"]==None:
            return 2     
        else:
            return 1
    def def_optim_list(self, parameters):
        super().def_optim_list(parameters)
        if self.simulation_options["dispersion"]==True:
            total_elements=np.prod(self.simulation_options["dispersion_bins"])
            electrochem_dict={"p1":self.simulation_options["EIS_Cdl"]}
            for i in range(0,int(total_elements)):
                electrochem_dict["p{0}".format(i+2)]=["R{0}".format(i+1), "C{0}".format(i+2)]
            
            self.Laviron_circuit={"z1":"R0", "z2":electrochem_dict}
            self.simulator=EIS(circuit=self.Laviron_circuit, invert_imaginary=self.simulation_options["invert_imaginary"])
            self.all_names=self.simulator.param_names
        else:
            self.Laviron_circuit={"z1":"R0", "z2":{"p1":self.simulation_options["EIS_Cdl"], "p2":["R1", self.simulation_options["EIS_Cf"]]}}
            self.simulator=EIS(circuit=self.Laviron_circuit, invert_imaginary=self.simulation_options["invert_imaginary"])
    def synthetic_noise(self, parameters, frequencies, noise, flag="proportional"):

        sim=self.simulate(parameters, frequencies)
        if flag=="proportional":
            return_arg=np.column_stack((self.add_noise(sim[:,0], noise, method="proportional"), self.add_noise(sim[:,1], noise, method="proportional")))
        else:
            return_arg=np.column_stack((self.add_noise(sim[:,0], noise*np.mean(sim[:,0])), self.add_noise(sim[:,1], noise*np.mean(-sim[:,1]))))   
        
        return return_ar
    def get_all_voltages(self, frequencies, oscillations=2):
        potentials=[]
        times=[]
        freqs=np.multiply(frequencies, 2*np.pi)
        for i in range(0, len(frequencies)):
            
            times.append(np.linspace(0, oscillations/frequencies[i]))
            potentials.append(5e-3*np.sin(times[i]*freqs[i]))
        return np.array(times), np.array(potentials)
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
    def calculate_circuit_parameters(self,print_circuit_params):
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
        #print(Cf)

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
            if self.simulation_options["Rct_only"]==True:
                EIS_params["C2"]=self.dim_dict["Cfarad"]
        elif self.simulation_options["EIS_Cf"]==("Q2", "alpha2"):
            cpe_alpha=abs(1-self.dim_dict["cpe_alpha_faradaic"])
            EIS_params["Q2"]=1/(Cf**(cpe_alpha-1)*Ra**(-cpe_alpha))
            if self.simulation_options["Rct_only"]==True:
                #print("*"*40)
                EIS_params["Q2"]=self.dim_dict["Cfarad"]
            EIS_params["alpha2"]=self.dim_dict["cpe_alpha_faradaic"]
        if print_circuit_params==True:
            print(EIS_params)
        return EIS_params
    def simulate(self, parameters, frequencies, print_circuit_params=False):
        if self.simulation_options["label"]=="cmaes":
            params=self.change_norm_group(parameters, "un_norm")
        else:
            params=parameters
        for i in range(0, len(self.optim_list)):

            self.dim_dict[self.optim_list[i]]=params[i]

        
        
        #print(EIS_params)
        #.print(Ra_coeff)
        #EIS_params={'R0': 5, 'C1': 1e-06, 'R1': 59.27316911806477, 'C2': 2.4156737037803954e-05}
        if self.simulation_options["dispersion"]==True:
            
            self.disp_test=[]
            Z_vals=np.zeros((len(frequencies),self.n_outputs()))
            if self.simulation_options["GH_quadrature"]==True:
                sim_params, self.values, self.weights=self.disp_class.generic_dispersion(self.dim_dict, self.other_values["GH_dict"])
            else:
                sim_params, self.values, self.weights=self.disp_class.generic_dispersion(self.dim_dict)
                
            #print(self.values, "LAV")

            for i in range(0, len(self.weights)):
                for j in range(0, len(sim_params)):   
                    self.dim_dict[sim_params[j]]=self.values[i][j]
                    #self.dim_dict[sim_params[j]]=
                    #print(self.values[i][j])
                simulation_params=self.calculate_circuit_parameters(print_circuit_params)
                if self.simulation_options["EIS_Cf"]=="CPE":
                    raise ValueError("Can't have dispersion and a CPE for the faradaic process")
                current_weight=np.prod(self.weights[i])
                if i==0:
                   EIS_params=simulation_params
                else:
                    EIS_params["C{0}".format(i+2)]=simulation_params["C2"]
                    EIS_params["R{0}".format(i+1)]=simulation_params["R1"]
                #print(self.Laviron_circuit)
                #print(EIS_params)
                EIS_params["C{0}".format(i+2)]*=current_weight
                EIS_params["R{0}".format(i+1)]/=current_weight
                """if i==0:
                    simulation_params={key:EIS_params[key]*np.prod(self.weights[i]) for key in EIS_params}
                else:
                    current_weight=np.prod(self.weights[i])
                    for key in simulation_params.keys():
                        simulation_params[key]+=EIS_params[key]*current_weight"""
                #sim_vals=self.simulator.test_vals(EIS_params, frequencies)
                if self.simulation_options["dispersion_test"]==True:
                    self.disp_test.append(sim_vals)
                #current_z_val=np.multiply(sim_vals, np.prod(self.weights[i]))
                #Z_vals=np.add(Z_vals, current_z_val)
            Z_vals=self.simulator.test_vals(EIS_params, frequencies)
        else:       
            EIS_params=self.calculate_circuit_parameters(print_circuit_params)
            #print(EIS_params, "Normal")
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
        if "return_magnitudes" not in simulation_options:
            simulation_options["return_magnitudes"]=False
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
    def define_potentials(self,):
        vals= self.simulation_options["E_step_start"]+ np.multiply(self.simulation_options["E_step"],np.arange(0, self.simulation_options["num_steps"]))
       
        return vals
    def get_all_voltages(self, ):
        E_start_vals=self.define_potentials()
        potentials=[]
        times=[]
        for i in range(0, self.simulation_options["num_steps"]):
            self.dim_dict["E_start"]=E_start_vals[i]
            self.update_params([self.dim_dict[x] for x in self.optim_list])
            potentials.append(self.define_voltages())
            times.append(self.t_nondim(self.time_vec))
        return np.array(times), np.array(potentials)
    def simulate(self, parameters, frequencies, rolling_window=False):
        magnitudes=np.zeros((self.simulation_options["num_even_harms"], self.simulation_options["num_steps"]))
        E_start_vals=self.define_potentials()
        for i in range(0, self.simulation_options["num_steps"]):
            self.dim_dict["E_start"]=E_start_vals[i]#(i*self.simulation_options["E_step"])
            
            current=super().simulate(parameters, frequencies)
            if self.simulation_options["synthetic_noise"]!=0:
                current=self.add_noise(current, self.simulation_options["synthetic_noise"]*max(current))
            #plt.plot(self.simulation_times, current)
            #plt.show()
            magnitudes[:, i]=self.get_even_amplitudes(self.simulation_times, current)
        minima=np.zeros(self.simulation_options["num_even_harms"])
        if self.simulation_options["return_magnitudes"]==True:
            return magnitudes
        for i in range(0, self.simulation_options["num_even_harms"]):
            if rolling_window==True:
                magnitudes[i,:]=self.rolling_mean(magnitudes[i,:], window=4)
            
            minima[i]=E_start_vals[np.where(magnitudes[i,:]==min(magnitudes[i,:]))]/self.nd_param.c_E0
        
        return minima
        
