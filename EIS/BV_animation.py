import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
import matplotlib.animation as animation
from EIS_class import EIS
import numpy as np
def potential(amp,frequency, time, phase):
    return amp*np.sin(2*np.pi*frequency*time+phase)
def current(cdl,frequency, time, phase):
    return (cdl)*frequency*np.cos(2*np.pi*frequency*time+phase)

class eis_sim:
    def __init__(self, param_list, simulation_options, other_values, param_bounds, optim_list):
        
        self.param_list=param_list
        self.sim_opts=simulation_options
        self.ov=other_values
        self.pb=param_bounds
        self.ol=optim_list
    def simulate(self, parameters):
        self.param_list["omega"]=parameters["omega"]
        self.param_list["original_omega"]=parameters["omega"]
        self.sim_opts["no_transient"]=1/self.param_list["omega"]
        sim_class=single_electron("", self.param_list, self.sim_opts, self.ov, self.pb)
       
        sim_class.def_optim_list(self.ol)
        params=[parameters[x] for x in sim_class.optim_list]
        self.sim_class=sim_class
        return sim_class.test_vals(params, "timeseries")
param_list={
        "E_0":0.001,
        'E_start':  -5e-3, #(starting dc voltage - V)
        'E_reverse':5e-3,
        'omega':10,  #    (frequency Hz)
        "original_omega":10,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl':1e-5, #(capacitance parameters)
        'CdlE1': 0.000653657774506*0,
        'CdlE2': 0.000245772700637*0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 75.03999999999999, #(reaction rate s-1)
        'alpha': 0.55,
        "E0_mean":0.2,
        "E0_std": 0,
    
        "alpha_mean":0.45,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/2**8),
        'phase' :0,
        "cap_phase":0,
        "time_end": None,
        'num_peaks': 30,
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
#time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test":False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "top_hat_return":"composite",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(2, 7)),
    "bounds_val":20000,
    
}
param_bounds={
    'E_0':[-10e-3, 10e-3],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 2e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [1e-3, 2e3], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
    "cap_phase":[math.pi/2, 2*math.pi],
    "E0_mean":[param_list['E_start'],param_list['E_reverse']],
    "E0_std": [1e-5,  0.1],
    "alpha_mean":[0.4, 0.65],
    "alpha_std":[1e-3, 0.3],
    "k0_shape":[0,1],
    "k0_scale":[0,1e4],
    "k0_range":[1e2, 1e4],
    'phase' : [math.pi, 2*math.pi],
}


import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def impedance_response(min_f, max_f, points_per_decade, num_osc, parameters,sim_class, sf=200, amplitude=5e-3):
    if (np.log2(sf)%2)!=0:
        sf=2**np.ceil(np.log2(sf))
    num_points=int((num_osc)*sf)
    frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
    freqs=[10**x for x in frequency_powers]
    Z=np.zeros((len(freqs), num_points), dtype="complex")
    threshold=0.05
    phase=0
    impedances=np.zeros(len(freqs), dtype="complex")
    magnitudes=np.zeros(len(freqs))
    phases=np.zeros(len(freqs))
    parameters=sim_class.param_list
    problem_child=False
    potentials=[]
    currents=[]
    time_arrays=[]
    frequencies=[]
    iffts=[]
    vffts=[]
    for i in range(0, len(frequency_powers)):
        time_end=num_osc/freqs[i]
        times1=np.linspace(0, time_end, num_points, endpoint=False)

        parameters["omega"]=freqs[i]

        nd_current=sim_class.simulate(parameters)#current(cdl, freqs[i], times,phase)
        conv_class=sim_class.sim_class
        I=conv_class.i_nondim(nd_current)
        I=conv_class.add_noise(I, 0.01*max(I))
        if i==0:
            print(conv_class.dim_dict["k_0"])
        V=conv_class.e_nondim(conv_class.define_voltages())[conv_class.time_idx]#


        #V1=potential(amplitude, freqs[i], times1, phase)
        times=conv_class.t_nondim(conv_class.time_vec)[conv_class.time_idx]
        
        #plt.plot(abs(np.fft.fft(I)))
        #I+=+0.5*max(I)*np.random.rand(len(I))
        #I=np.add(I, max(I)*np.random.rand(len(I)))
        #plt.plot(abs(np.fft.fft(I)))
        #plt.plot(I)
        #plt.show()
        ffts=[]
        #plt.plot(times1, V1)

      
        #plt.plot(times, V)
        #plt.show()
        for dataset in [V, I]:
            
            fft=1/num_points*np.fft.fftshift(np.fft.fft(dataset))
            abs_fft=abs(fft)
            fft[abs_fft<threshold*max(abs_fft[1:])]=1e-10
            ffts.append(fft)

     
        potentials.append(V)

        currents.append(I)
        time_arrays.append(times)
        Z_f=np.divide(ffts[0], ffts[1])
        #plt.plot(times,V)
        #plt.show()

        abs_fft=np.abs(Z_f)
        #abs_V=np.abs(fft_V)
        #Z[i,:][abs_fft<threshold*max(abs_fft)]=0
        fft_freq=np.fft.fftshift(np.fft.fftfreq(len(times), times[1]-times[0]))
        plt_idx=np.where((fft_freq>(freqs[i]-(0.5*freqs[i]))) & (fft_freq<(freqs[i]+(0.5*freqs[i]))))
        #plt.loglog(fft_freq[plt_idx], abs_fft[plt_idx])
        subbed_f=abs(np.subtract(fft_freq, freqs[i]))
        freq_idx=np.where(subbed_f==min(subbed_f))
        iffts.append(abs(ffts[1]))
        vffts.append(abs(ffts[0]))
        frequencies.append(fft_freq)
        #plt.axvline(fft_freq[freq_idx], linestyle="--")
        
        impedances[i]=Z_f[freq_idx][0]
        #print(impedances)
        phases[i]=abs(np.angle(fft, deg=True))[freq_idx][0]
        if phases[i]<1e-10:
                #print(freqs[i], phases[i])
                #print(conv_class.dim_dict["E_0"])
                """fig, ax=plt.subplots(1,2)
                ax[0].plot(times, V)
                ax[1].plot(times, I)
                plt.show()
                plt.plot(fft_freq, ffts[0])
                plt.show()
                plt.plot(fft_freq, ffts[1])
                plt.show()"""
                print("!!!")
                phases[i]=None
                
        
   
        #plt.show()
        #plt.plot(np.angle(fft, deg=True))
        #plt.show()
        magnitudes[i]=abs_fft[freq_idx][0]
    return magnitudes, phases,impedances, potentials, currents, frequencies, iffts, vffts, time_arrays
#plt.show()

min_f=-2
max_f=6
points_per_decade=10


import copy

frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
freqs=[10**x for x in frequency_powers]
param_dict={'k_0': 75.03999999999999, 'Ru': 100, 'Cdl': 1e-5, 'gamma': 1e-10, 'E_0': 0.001, 'alpha': 0.55, 'area': 0.07, 'DC_pot': 0}
param_names=["gamma", "k_0", "Cdl", "alpha", "Ru"]
sim_class=eis_sim(copy.deepcopy(param_list), simulation_options,other_values, param_bounds, param_names+["omega"] )

param_dict["E_0"]=0.001
m, p, z, potentials, currents, frequencies, iffts, vffts, time_arrays=impedance_response(min_f=min_f, max_f=max_f, points_per_decade=points_per_decade, num_osc=param_list["num_peaks"], 
                    parameters=param_dict,sim_class=sim_class, 
                    sf=1/param_list["sampling_freq"],amplitude=param_list["d_E"] )

index=np.where(np.isnan(p)==False)

real=z.real[index]


spectra=np.column_stack((real, z.imag[index]))
lav_circuit={"z1":"R0", "z2":{"p1":"C1", "p2":["R1", "C2"]}}
lav_ec=EIS(circuit=lav_circuit)
save_data=EIS().convert_to_bode(np.column_stack((real, z.imag[index])))
ec_data=lav_ec.test_vals({'R0': 100.89130827528389, 'R1': 985.4168864168606, 'C1': 7.246044133637491e-07, 'C2': 6.522158184501145e-06}, np.multiply(freqs,2*math.pi))
#fig, ax=plt.subplots()
#twinx=ax.twinx()
#EIS().bode(save_data, freqs, data_type="phase_mag", ax=ax, twinx=twinx)
#EIS().bode(ec_data, freqs, twinx=twinx,ax=ax)

class plot_class:
    def __init__(self, potentials, currents,times,frequencies, fft_V, fft_I, circuit_impedance, td_impedance):
        self.times=times
        self.frequencies=frequencies
        self.potential=potentials
        self.current=currents
        self.fft_V=fft_V
        self.fft_I=fft_I
        self.circuit_impedance=circuit_impedance
        self.td_impedance=td_impedance
        self.fig,self.ax=plt.subplots(2,2)
        
        self.twinx3=self.ax[1,1].twinx()
        self.twinx2=self.ax[1,0].twinx()
        #self.twinx1=self.ax[0].twinx()
        #plt.plot(self.potential[0])
        self.p_line,=self.ax[0,0].plot(self.times[0], self.potential[0])
        self.ax[0,0].set_xlabel("Time (s)")
        self.ax[0,0].set_ylabel("Potential (V)")
        self.ax[0,1].set_xlabel("Time (s)")
        self.ax[0,1].set_ylabel("Current (A)")
        self.ax[1,0].set_xlabel("Frequency (Hz)")
        self.ax[1,0].set_ylabel("Magnitude")
        self.twinx2.set_ylabel("Magnitude")
        self.c_line,=self.ax[0,1].plot(self.times[0], self.current[0], color="Red")
        
        self.vfft_line,=self.ax[1,0].loglog(self.frequencies[0], self.fft_V[0], label="FFT(Potential)")
        self.ax[1,0].plot(0,0, label="FFT(Current)", linestyle="--", color="red")
        self.ifft_line,=self.twinx2.loglog(self.frequencies[0], self.fft_I[0], color="Red", linestyle="--")
        self.ax[1,0].legend(loc="upper left")
        self.current_t_end=self.times[0][-1]
        self.ax[1,1].plot(0,0, color="orange",  label="Time domain")
        EIS().bode(circuit_impedance, freqs, ax=self.ax[1,1], twinx=self.twinx3, compact_labels=True, label="Equivalent circuit")
        self.ax[1,1].legend(loc="upper right")
    def animate(self,i):
        if self.current_t_end/2>(self.times[i][-1]):
            self.ax[0,0].set_xlim([0, self.times[i][-1]])
            self.ax[0,1].set_xlim([0, self.times[i][-1]])
            self.current_t_end=self.times[i][-1]
        self.ax[0,0].set_ylim([1.05*min(self.potential[i]), 1.05*max(self.potential[i])])
        self.ax[0,1].set_ylim([1.05*min(self.current[i]), 1.05*max(self.current[i])])
        self.p_line.set_xdata(self.times[i])
        self.c_line.set_xdata(self.times[i])
        self.p_line.set_ydata(self.potential[i])
        self.c_line.set_ydata(self.current[i])
        self.ax[1,0].set_xlim([0, self.frequencies[i][-1]])
        self.ifft_line.set_xdata(self.frequencies[i])
        self.vfft_line.set_xdata(self.frequencies[i])
        self.ifft_line.set_ydata(self.fft_I[i])
        self.vfft_line.set_ydata(self.fft_V[i])
        print(1.15*max(self.fft_I[i]), 1.15*max(self.fft_V[i]))
        self.twinx2.set_ylim([0, 1.35*max(self.fft_I[i])])
        self.ax[1,0].set_ylim([0, 1.35*max(self.fft_V[i])])
        self.ax[1,1].scatter(np.log10(freqs[i]), -self.td_impedance[i,0], color="orange", marker="v")
        self.twinx3.scatter(np.log10(freqs[i]), self.td_impedance[i,1], color="orange", marker="o")
td=plot_class( potentials, currents, time_arrays, frequencies, iffts, vffts,ec_data, save_data)
td.fig.set_size_inches(8, 8)
plt.subplots_adjust(left=0.108,
                    bottom=0.11, 
                    right=0.91,
                    top=0.88,
                    wspace=0.527,
                    hspace=0.305)

ani = animation.FuncAnimation(
    td.fig, td.animate, interval=300, blit=True)



plt.show()

