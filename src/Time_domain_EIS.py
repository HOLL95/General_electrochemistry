import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
import warnings
class Time_domain_EIS(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        if "threshold" not in simulation_options:
            simulation_options["threshold"]=0.5
        
        sf=1/dim_parameter_dictionary["sampling_freq"]
        if (np.log2(sf)%2)!=0:
            sf=2**np.ceil(np.log2(sf))
            warnings.warn("Changing sampling rate to {0}".format(sf))
            dim_parameter_dictionary["sampling_freq"]=1/sf 
            
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
    
    def td_simulate(self, parameters, freqs):

        num_points=int(self.dim_dict["num_peaks"]/self.dim_dict["sampling_freq"])

        Z=np.zeros((len(freqs), num_points), dtype="complex")
        threshold=self.simulation_options["threshold"]
        impedances=np.zeros(len(freqs), dtype="complex")
        magnitudes=np.zeros(len(freqs))
        phases=np.zeros(len(freqs))
        print(self.dim_dict["Cdl"])
        for i in range(0, len(freqs)):
            #print(self.nd_param.nd_param_dict["nd_omega"])
            self.dim_dict["original_omega"]=freqs[i]
            self.dim_dict["omega"]=freqs[i]
            
            super().__init__("", self.dim_dict, self.simulation_options, self.other_values, self.param_bounds)
            #print(self.nd_param.nd_param_dict["nd_omega"])
            time_end=self.dim_dict["num_peaks"]/freqs[i]
            times=np.linspace(0, time_end, num_points, endpoint=False)
            times1=self.time_vec
            volt_q=np.array([self.voltage_query(x) for x in times1])
            
            #V=self.e_nondim(volt_q[:,0])
            V=self.potential(self.dim_dict["d_E"], freqs[i], times, self.dim_dict["phase"])
            #plt.plot(np.subtract(V1, V))
            #plt.show()
            #plt.plot(V)
            #plt.plot(V1)
            #plt.show()
            #print("I", self.dim_dict["d_E"], freqs[i], self.dim_dict["phase"])
            I=self.pure_capacitor(self.dim_dict["Cdl"], self.dim_dict["d_E"], freqs[i], times,self.dim_dict["phase"])
            #self.nd_param.nd_param_dict["d_E"]=self.dim_dict["d_E"]
            #print("I1",self.nd_param.nd_param_dict["nd_omega"], self.nd_param.nd_param_dict["phase"], self.nd_param.nd_param_dict["d_E"])
          #  I1=volt_q[:,1]
          #  plt.plot(I)
            #plt.plot(I1)
            #plt.show()
            ffts=[]
            for dataset in [V, I]:
                fft=1/num_points*np.fft.fftshift(np.fft.fft(dataset))
                abs_fft=abs(fft)
                fft[abs_fft<threshold*max(abs_fft)]=1
                ffts.append(fft)
            Z_f=np.divide(ffts[0], ffts[1])
            abs_fft=np.abs(Z_f)
            fft_freq=np.fft.fftshift(np.fft.fftfreq(num_points, times[1]-times[0]))
            plt_idx=np.where((fft_freq>(freqs[i]-(0.5*freqs[i]))) & (fft_freq<(freqs[i]+(0.5*freqs[i]))))
            subbed_f=abs(np.subtract(fft_freq, freqs[i]))
            freq_idx=np.where(subbed_f==min(subbed_f))
            impedances[i]=Z_f[freq_idx][0]
            phases[i]=abs(np.angle(fft, deg=True))[freq_idx][0]
            magnitudes[i]=abs_fft[freq_idx][0]
        return phases, magnitudes,impedances, freqs
    def potential(self, amp,frequency, time, phase):
        return amp*np.sin(2*np.pi*frequency*time+phase)
    def pure_capacitor(self, cdl, amp, frequency, time, phase):
        return cdl*frequency*amp*np.cos(2*np.pi*frequency*time+phase)
        

