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
from EIS_class import EIS
import numpy as np
def potential(amp,frequency, time, phase):
    return amp*np.sin(2*np.pi*frequency*time+phase)
def current(cdl,frequency, time, phase):
    return (cdl)*frequency*np.cos(2*np.pi*frequency*time+phase)
param_list={
        "E_0":0.001,
        'E_start':  -5e-3, #(starting dc voltage - V)
        'E_reverse':5e-3,
        'omega':10,  #    (frequency Hz)
        "original_omega":10,
        'd_E': 5e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl':1e-4, #(capacitance parameters)
        'CdlE1': 0.000653657774506*0,
        'CdlE2': 0.000245772700637*0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 0.1, #(reaction rate s-1)
        'alpha': 0.55,
        "E0_mean":0.2,
        "E0_std": 0,
    
        "alpha_mean":0.45,
        "alpha_std":1e-3,
        'sampling_freq' : (1.0/2**8),
        'phase' :0,
        "cap_phase":0,
        "time_end": None,
        'num_peaks': 5,
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

def impedance_response(min_f, max_f, points_per_decade, num_osc, parameters,sim_class, sf=200, amplitude=5e-3, noise=None):
    if (np.log2(sf)%2)!=0:
        sf=2**np.ceil(np.log2(sf))
    num_points=int((num_osc)*sf)
    frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
    freqs=[10**x for x in frequency_powers]
    Z=np.zeros((len(freqs), num_points), dtype="complex")
    threshold=0.5
    phase=0
    impedances=np.zeros(len(freqs), dtype="complex")
    magnitudes=np.zeros(len(freqs))
    phases=np.zeros(len(freqs))
    parameters=sim_class.param_list
    problem_child=False
    for i in range(0, len(frequency_powers)):
        time_end=num_osc/freqs[i]
        times1=np.linspace(0, time_end, num_points, endpoint=False)

        parameters["omega"]=freqs[i]

        nd_current=sim_class.simulate(parameters)#current(cdl, freqs[i], times,phase)
        conv_class=sim_class.sim_class
        I=conv_class.i_nondim(nd_current)
        if noise is not None:
            I=conv_class.add_noise(I, noise*max(I))
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
            fft[abs_fft<threshold*max(abs_fft)]=1e-10
            ffts.append(fft)

     

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
        #plt.axvline(fft_freq[freq_idx], linestyle="--")
        
        impedances[i]=Z_f[freq_idx][0]
        #print(impedances)
        phases[i]=abs(np.angle(fft, deg=True))[freq_idx][0]
        """ if phases[i]<1e-10 or (problem_child==True):
                print(freqs[i], phases[i])
                print(conv_class.dim_dict["E_0"])
                fig, ax=plt.subplots(1,2)
                ax[0].plot(times, V)
                ax[1].plot(times, I)
                plt.show()
                plt.plot(fft_freq, ffts[0])
                plt.show()
                plt.plot(fft_freq, ffts[1])
                plt.show()
                problem_child=True"""
        
   
        #plt.show()
        #plt.plot(np.angle(fft, deg=True))
        #plt.show()
        magnitudes[i]=abs_fft[freq_idx][0]
    return magnitudes, phases,impedances
#plt.show()
cdl=1e-3
min_f=-3
max_f=7
points_per_decade=10
fig, ax=plt.subplots()
twinx=ax.twinx()
param_val_scans={"k_0":[0.1, 10, 100, 125], 
            "gamma":[7.5e-11, 9e-11, 1.1e-10, 1.25e-10],
            "Cdl":[2e-5, 5e-5, 1e-4, 5e-4],
            "Ru":[0.1, 1, 10, 1000], 
            "alpha":[0.4, 0.5, 0.6, 0.7],
            "phase":[0, np.pi/4, np.pi/2, 0.75*np.pi],
            "cap_phase":[0, np.pi/4, np.pi/2], }
for noise in [0.005, 0.01,0.03, 0.05,0.5]:
    param_names=list(param_val_scans.keys())
    frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
    freqs=[10**x for x in frequency_powers]
    sim_class=eis_sim(param_list, simulation_options,other_values, param_bounds, param_names+["omega"] )
    param_dict={key:param_list[key] for key in param_names}
    param_dict["E_0"]=0.001
    m, p, z=impedance_response(min_f=min_f, max_f=max_f, points_per_decade=points_per_decade, num_osc=param_list["num_peaks"], 
                        parameters=param_dict,sim_class=sim_class, 
                        sf=1/param_list["sampling_freq"],amplitude=param_list["d_E"], noise=noise )



    EIS().bode(np.column_stack((z.real, z.imag)), freqs, twinx=twinx, data_type="complex", compact_labels=True, ax=ax, label="{0} % * max(I)".format(noise*100))
plt.show()