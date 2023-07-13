import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
from EIS_class import EIS
import warnings
import time
import re
import copy
from scipy.special import iv
from params_class import params
class EIS_TD(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        if "data_representation" not in simulation_options:
            simulation_options["data_representation"]="bode"
        simulation_options["likelihood"]="timeseries"
        if "secret_data" not in other_values:
            other_values["secret_data"]=None
        if "eis_test" not in simulation_options:
            simulation_options["eis_test"]=False
        if "cdl_dispersion" not in simulation_options:
            simulation_options["frequency_dispersion"]=False
       
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
    def define_frequencies(self,min_f, max_f, points_per_decade=10):
        frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
        freqs=[10**x for x in frequency_powers]
        return np.array(freqs)
    def n_outputs(self):
        if self.simulation_options["data_representation"]=="bode":
            return 2
        elif self.simulation_options["data_representation"]=="nyquist":
            return 2
        elif self.simulation_options["data_representation"]=="all":
            return 4
    def n_parameters(self,):
        return len(self.optim_list)
    def def_optim_list(self, parameters):
        if self.simulation_options["frequency_dispersion"] is False:
            super().def_optim_list(parameters)
        else:
            matches=[re.compile("{0}_[0-9]+".format(x)) for x in self.simulation_options["frequency_dispersion"]]

            for i in range(0, len(parameters)):
                for j in range(0, len(matches)):
                    matched_number=matches[j].match(parameters[i])
                    if matched_number is not None:
                        self.param_bounds[parameters[i]]=self.param_bounds[self.simulation_options["frequency_dispersion"][j]]
                        self.dim_dict[parameters[i]]=None
            super().def_optim_list(parameters)
    def simulate(self,parameters, freqs):
        sf=1/self.dim_dict["sampling_freq"]
        if (np.log2(sf)%2)!=0:
            sf=2**np.ceil(np.log2(sf))
        
        self.dim_dict["sampling_freq"]=1/sf
        num_points=int((self.dim_dict["num_peaks"])*sf)
        int_sf=int(sf)
        Z=np.zeros((len(freqs), num_points), dtype="complex")
        threshold=0.05
        impedances=np.zeros(len(freqs), dtype="complex")
        magnitudes=np.zeros(len(freqs))
        phases=np.zeros(len(freqs))
        nu=self.dim_dict["E_0"]
        orig_time_vec=self.time_vec
        copy_list=copy.deepcopy(self.optim_list)
        save_list=copy.deepcopy(self.optim_list)
        p_dict=dict(zip(copy_list, parameters))
        if self.simulation_options["frequency_dispersion"] is not False:
            freq_disp_params=self.simulation_options["frequency_dispersion"]
            dispersion_array=np.zeros((len(freq_disp_params), len(freqs)))
            for j in range(0, len(freq_disp_params)):
                for i in range(0, len(freqs)):
                    param_name="{0}_{1}".format(freq_disp_params[j],i)
                    dispersion_array[j, i]=p_dict[param_name]
                    idx=copy_list.index(param_name)
                    copy_list.pop(idx)
            sim_params=copy_list+freq_disp_params
            super().def_optim_list(sim_params)

        for i in range(0, len(freqs)):
            problem_child=False
            self.time_vec=orig_time_vec
            for j in range(0,2):
                if j==1:

                    if problem_child==False:
                        continue
                    else:
                        try:
                            self.time_vec=np.arange(0, new_end, self.nd_param.nd_param_dict["sampling_freq"])
                        except:
                            #print(new_end, freqs[i])
                           
                            self.time_vec=np.arange(0, 300, self.nd_param.nd_param_dict["sampling_freq"])
                        
                        


                #time_end=num_osc/freqs[i]
                #times1=np.linspace(0, time_end, num_points, endpoint=False)

                self.dim_dict["omega"]=freqs[i]
                self.dim_dict["original_omega"]=freqs[i]
                if self.simulation_options["frequency_dispersion"] is not False:
                    parameters=[p_dict[x] for x in copy_list]+[dispersion_array[x, i] for x in range(0, len(freq_disp_params))]
                    #print(parameters)
                nd_current=super().simulate(parameters, [])#current(cdl, freqs[i], times,phase)
                
                I=self.i_nondim(nd_current)[int_sf:]
                V=self.e_nondim(self.define_voltages())[int_sf:]#
                times=self.t_nondim(self.time_vec)[int_sf:]
                #if j==1:
                #    plt.title(new_end)
                #    plt.plot(times, I)
                #    plt.show()
                ffts=[]
                   
                for dataset in [V, I]:
                    
                    fft=(1/num_points)*np.fft.fftshift(np.fft.fft(dataset))
                    abs_fft=abs(fft)
                    fft[abs_fft<threshold*max(abs_fft[1:])]=1e-10
                    ffts.append(fft)
            
                Z_f=np.divide(ffts[0], ffts[1])
                abs_fft=np.abs(Z_f)
                fft_freq=np.fft.fftshift(np.fft.fftfreq(len(times), times[1]-times[0]))
                plt_idx=np.where((fft_freq>(freqs[i]-(0.5*freqs[i]))) & (fft_freq<(freqs[i]+(0.5*freqs[i]))))
                pos_idx=np.where(fft_freq>0)
                subbed_f=abs(np.subtract(fft_freq, freqs[i]))
                freq_idx=np.where(subbed_f==min(subbed_f))            
                impedances[i]=Z_f[freq_idx][0]
                phases[i]=abs(np.angle(fft, deg=True))[freq_idx][0]
                if phases[i]<1e-10:
                    low_grad=3e-6
                    #print(parameters)
                    #values, weights=self.return_distributions(16)
                    #print(values)
                    #rolled=self.rolling_window(I, int_sf)
                    #plt.subplot(1,4,1)
                    #plt.plot( times, I)
                    #plt.plot(times, rolled)
                    
                    #logged=np.log(abs(rolled[3*int_sf:-int_sf]))
                    #logged_t=times[3*int_sf:-int_sf]
                    #interped= np.polyfit(logged_t, logged ,1) #np.interp(times, logged_t, logged)

                    #b=np.exp(interped[1])
                    #gradient=-interped[0]
                    #print(intercept)

                    
                    #x_val=np.log((b*gradient)/low_grad)
                    #new_end=np.ceil(x_val/self.nd_param.c_T0)
                    #if new_end>300 or new_end<0:
                    #    new_end=300
                    
                    #plt.subplot(1,4,2)
                    #ifft=abs_fft=abs(1/num_points*np.fft.fftshift(np.fft.fft(I)))
                    #plt.axhline(threshold*max(ifft))
                    #plt.plot(np.fft.fftfreq(len(times), times[1]-times[0]),ifft)
                    #plt.plot(logged_t, logged)
                    #plt.plot(times, np.log([b*np.exp(-gradient*x) for x in times]))
                    #plt.subplot(1,4,3)
                    #plt.plot(times, [abs(-gradient*b*np.exp(-gradient*x)) for x in times])
                    #plt.axvline(x_val)
                    #plt.subplot(1,4,4)
                    #plt_idx=pos_idx
                    #plt.semilogy(fft_freq, abs(ffts[0][plt_idx]))
                    #plt.title(freqs[i])
                    #plt.semilogy(fft_freq, abs(ffts[1]))
                    
                    #plt.plot(np.fft.fftfreq(len(times), times[1]-times[0]),ffts[1])
                    #plt.show()
                    #problem_child=True
                
                magnitudes[i]=abs_fft[freq_idx][0]
                if problem_child==False or j==1:
                    if self.simulation_options["dispersion_test"]==True and len(self.simulation_options["dispersion_parameters"])>0:
                        if i%10==0:
                            fig,ax=plt.subplots(1,2)
                            for q in range(0, len(self.disp_test)):
                                ax[0].plot(times, self.i_nondim(self.disp_test[q])[int_sf:])
                            ax[0].plot(times, I, lw=2, color="black")
                            orig_optim_list=self.optim_list
                            #print(self.simulation_options["dispersion_parameters"])
                            self.def_optim_list(self.simulation_options["dispersion_parameters"])

                            sim_params=[self.dim_dict["{0}_mean".format(x)] for x in self.simulation_options["dispersion_parameters"]]
                            undisped_I=self.i_nondim(super().simulate(sim_params, []))[int_sf:]#current(cdl, freqs[i], times,phase)
                            self.def_optim_list(orig_optim_list)
                            #I=self.i_nondim(nd_current)[int_sf:]
                            ax[1].plot(times, I)
                            ax[1].plot(times, undisped_I)
                            plt.show()

                        
            index=np.where(np.isnan(phases)==False)
            real=impedances.real
            imag=impedances.imag
        #real[index]=0
        #imag[index]=0
        if self.simulation_options["frequency_dispersion"] is not False:
            self.optim_list=save_list
        if self.simulation_options["eis_test"]==True:
            fig, ax=plt.subplots(1,2)
            twinx=ax[0].twinx()
            if self.other_values["secret_data"] is not None:
                EIS().bode(self.other_values["secret_data"], freqs, ax=ax[0], twinx=twinx, label="Data")
                EIS().nyquist(self.other_values["secret_data"], ax=ax[1], label="Data", orthonormal=False)
            EIS().bode(np.column_stack((real, imag)), freqs, ax=ax[0], twinx=twinx, label="Sim")
            EIS().nyquist(np.column_stack((real, imag)), ax=ax[1], label="Sim", orthonormal=False)
            EIS().bode( EIS().convert_to_bode(np.column_stack((real, imag))), freqs, data_type="phase_mag")
            plt.show()
            plt.show()
        if self.simulation_options["data_representation"]=="bode":
            
            return EIS().convert_to_bode(np.column_stack((real, imag)))
        elif self.simulation_options["data_representation"]=="nyquist":
            return np.column_stack((real, imag))
        elif self.simulation_options["data_representation"]=="all":
            phase_mag=EIS().convert_to_bode(np.column_stack((real, imag)))
            return np.column_stack((real, imag, phase_mag[:,0], phase_mag[:,1]))

class EIS_TD_sequential(EIS_TD):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        super().__init__(dim_parameter_dictionary, simulation_options, other_values, param_bounds)
    def simulate(self,parameters, single_frequency):   
        actual_freq=[single_frequency[1]]
        simulation=super().simulate(parameters, actual_freq)
        return simulation[0]
    def n_outputs(self):   
        return 1