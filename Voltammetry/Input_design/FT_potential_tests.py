import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from pints import plot
from harmonics_plotter import harmonics
import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
import matplotlib.pyplot as plt
import sys
harm_range=list(range(4, 6))
from scipy import interpolate
class FT_potential_osc:
    def __init__(self):
            param_list={
                "E_0":0.2,
                'E_start':  -500e-3, #(starting dc voltage - V)
                'E_reverse':200e-3,
                'omega':8.88480830076,  #    (frequency Hz)
                "v": 200e-3,
                'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
                'area': 0.07, #(electrode surface area cm^2)
                'Ru': 100.0,  #     (uncompensated resistance ohms)
                'Cdl': 1e-4, #(capacitance parameters)
                'CdlE1': 0.000653657774506,
                'CdlE2': 0.000245772700637,
                "CdlE3":-1e-6,
                'gamma': 2e-11,
                "original_gamma":2e-11,        # (surface coverage per unit area)
                'k_0': 1000, #(reaction rate s-1)
                'alpha': 0.5,
                "E0_mean":0.2,
                "E0_std": 0.09,
                "cap_phase":0,
                "alpha_mean":0.5,
                "alpha_std":1e-3,
                'sampling_freq' : (1.0/200),
                'phase' :0,
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
                "test": False,
                "method": "ramped",
                "phase_only":False,
                "likelihood":likelihood_options[0],
                "numerical_method": solver_list[1],
                "label": "MCMC",
                "optim_list":[]
            }
            other_values={
                "filter_val": 0.5,
                "harmonic_range":harm_range,
                "bounds_val":20000,
            }
            param_bounds={
                'E_0':[param_list['E_start'],param_list['E_reverse']],
                'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
                'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
                'Cdl': [0,1e-3], #(capacitance parameters)
                'CdlE1': [-0.05,0.15],#0.000653657774506,
                'CdlE2': [-0.01,0.01],#0.000245772700637,
                'CdlE3': [-0.01,0.01],#1.10053945995e-06,
                'gamma': [0.1*param_list["original_gamma"],10*param_list["original_gamma"]],
                'k_0': [0.1, 1e4], #(reaction rate s-1)
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

            sim=single_electron(None, param_list, simulation_options, other_values, param_bounds)
            #current=sim.test_vals([], "timeseries")
            potential=sim.e_nondim(sim.define_voltages(transient=True))
            Y=np.fft.fft(potential)
            time=sim.t_nondim(sim.time_vec[sim.time_idx])

            freq=np.fft.fftfreq(len(potential), time[1]-time[0])
            one_side_Y=Y[:len(Y)//2]
            one_side_f=freq[:len(Y)//2]
            total_signal=np.zeros(len(time))
            plot_order=10
            abs_one_side=abs(one_side_Y)
            threshold_pc=0.1
            threshold_idx=np.where(abs_one_side>(threshold_pc*max(abs_one_side)))
            thresh_freqs=one_side_f[threshold_idx]
            thresh_Y=one_side_Y[threshold_idx]
            length=len(potential)
            self.sinusoid_dict={"freq_{0}".format(x):0 for x in range(1, 2*len(thresh_freqs)+1)}
            for key in ["amp_", "phase_"]:
                for j in range(1, 2*len(thresh_freqs)+1):
                    self.sinusoid_dict[key+str(j)]=0
            len_thresh=len(thresh_freqs)
            for i in range(0, len(thresh_Y)):
                freq=thresh_freqs[i]*2*math.pi
                cosine=thresh_Y[i].real*np.cos(freq*time)/length
                sine=thresh_Y[i].imag*np.sin(freq*time)/length
                for m in [i, i+len_thresh]:
                    self.sinusoid_dict["freq_{0}".format(m)]=thresh_freqs[i]
                    if m==i:
                        self.sinusoid_dict["amp_{0}".format(m)]=2*thresh_Y[i].real/length
                        self.sinusoid_dict["phase_{0}".format(m)]=math.pi/2
                    else:
                        self.sinusoid_dict["amp_{0}".format(m)]=2*thresh_Y[i].imag/length
                        self.sinusoid_dict["phase_{0}".format(m)]=0
                
                total_signal=np.add(total_signal, cosine)
                total_signal=np.add(total_signal, sine)
            print(self.sinusoid_dict)
            plt.plot(time, potential)    
            plt.plot(time, total_signal, alpha=0.8)
            plt.show()
FT_potential_osc()