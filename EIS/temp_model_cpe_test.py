import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scikits.odes import dae
from scipy.optimize import fsolve
from collections import deque
import math
class solve_functions:
    def __init__(self, params, derivative_vars, dt, end, current_idx, source_func, cpe_locs=None):
        self.params=params
        self.t_array=np.arange(0, end, dt)
        self.num_cpe=2
        self.cpe_potential_arrays={"cpe_"+str(x+1):deque([0,0,0,0]) for x in range(0, self.num_cpe)}
        self.cpe_arrays={"cpe_"+str(x+1):np.zeros(len(self.t_array)) for x in range(0, self.num_cpe)}
        self.cpe_keys=["cpe_{0}".format(x) for x in range(1, self.num_cpe+1)]
        self.cpe_coeffs=np.multiply(1/6, [11, -18,9, -2])
        self.current_idx=current_idx
        self.current_history=np.zeros(len(self.t_array))
        self.derivative_vars=derivative_vars
        self.deriv_history=[0 for x in self.derivative_vars]
        self.t_counter=0
        self.dt=dt
        if cpe_locs!=None:
            self.cpe_locs=cpe_locs
        self.source_func=source_func
    def residual(self, x):
        times=self.t_array[:self.t_counter]
        t=times[-1]
        result=[0,0,0,0,0,0,0]
        xdot=np.zeros(len(result))
        for i in range(0, len(self.derivative_vars)):
            xdot[self.derivative_vars[i]]=(x[self.derivative_vars[i]]-self.deriv_history[i])/self.dt
        #print(xdot)
        #parameter_area
        R0=self.params["R0"]
        R1=self.params["R1"]
        C1=self.params["C1"]
        self.cpe_dict={}
        result[0]=-x[4] + x[0]/R1 - x[1]/R1+(C1)*xdot[0]+(-C1)*xdot[2]
        result[1]=x[5] - x[0]/R1 + x[1]/R1
        result[2]=x[6]+(-C1)*xdot[0]+(C1)*xdot[2]
        result[3]=-x[5] - x[6] + x[3]/R0
        result[4]=self.source_func(t) - x[0]
        for i in range(0, self.num_cpe):
            str_idx=str(i+1)
            self.cpe_dict[str(self.cpe_locs[i])]=self.cpe(self.t_array[:self.t_counter], self.cpe_arrays["cpe_"+str_idx][:self.t_counter], self.params["Q"+str_idx], self.params["alpha"+str_idx], "cpe_"+str_idx)

        result[5]=-self.cpe_dict["5"] + x[1] - x[3]
        result[6]=-self.cpe_dict["6"] + x[2] - x[3]
        return result

    def cpe(self, linear_t, linear_i, Q, alpha, key):

        #print(t, i, "1")
        if linear_t[-1]==0:
            return 0
        powered_t=np.zeros(len(linear_t))
        powered_t[1:]= np.power(linear_t[1:], alpha-1)
        flipped_times=np.flip(powered_t)
        #print(flipped_times)
        convolv=np.sum(np.multiply(flipped_times, linear_i))
        v_1_dt=self.dt*convolv/(Q*gamma(alpha))
        if self.t_counter<5:
            highest_idx=self.t_counter-1
            coeffs=self.cpe_coeffs[:highest_idx]
            vals=list(self.cpe_potential_arrays[key])[:highest_idx]
            scaled_vals=np.sum(np.multiply(vals, coeffs))
        else:
            scaled_vals=np.sum(np.multiply(self.cpe_potential_arrays[key], self.cpe_coeffs))

        return_val=v_1_dt+scaled_vals

        return return_val
    def simulate(self):
        var_array=[0,0,0,0,0,0,0]
        current_array=np.zeros(len(self.t_array))
        for i in range(1, len(self.t_array)):
            self.t_counter=i
            var_array=fsolve(self.residual, var_array)
            for j in range(0, len(self.derivative_vars)):
                self.deriv_history[j]=var_array[self.derivative_vars[j]]
            for j in range(0, self.num_cpe):
                key=self.cpe_keys[j]
                self.cpe_arrays[key][self.t_counter]=var_array[self.cpe_locs[j]]#
                self.cpe_potential_arrays[key].rotate(1)
                self.cpe_potential_arrays[key][0]=self.cpe_dict[str(self.cpe_locs[j])]
            self.current_history[i]=var_array[self.current_idx]

        return self.current_history
def impedance_simulate(sim_class, frequencies, amplitude=1e-3, num_osc=3.5):
    nyquist=np.zeros((2, len(frequencies)))
    for i in range(0, len(frequencies)):
        freq_func=lambda t: amplitude*np.sin(2*math.pi*frequencies[i]*t)
        end=num_osc*2*math.pi/frequencies[i]
        dt=1/(frequencies[i]*600)
        new_class=solve_functions(sim_class.params, sim_class.derivative_vars, dt=dt, end=end, current_idx=sim_class.current_idx,
                        source_func=freq_func, cpe_locs=sim_class.cpe_locs)
        current=new_class.simulate()
        potential=np.array([freq_func(t) for t in new_class.t_array])
        ss_idx=np.where((new_class.t_array>(1.9/frequencies[i])) & (new_class.t_array<(3.1/frequencies[i])))
        ss_potential=potential[ss_idx]
        ss_time=new_class.t_array[ss_idx]
        half_window=1/(2*frequencies[i])
        max_potential_time=ss_time[np.where(ss_potential==max(ss_potential))][0]
        current_peak_idx=np.where((new_class.t_array>(max_potential_time-half_window))&(new_class.t_array<(max_potential_time+half_window)))
        ss_current=current[current_peak_idx]
        ss_current_time=new_class.t_array[current_peak_idx]
        max_current_time=ss_current_time[np.where(ss_current==max(ss_current))][0]
        time_diff=abs(max_potential_time-max_current_time)
        phase=2*math.pi*(time_diff*frequencies[i])
        print(phase*180/(2*math.pi))
        magnitude=max(ss_potential)/max(ss_current)
        z_freq=magnitude*np.exp(1j*phase)
        nyquist[:, i]=[np.real(z_freq), np.imag(z_freq)]

    return nyquist
dt=1e-05

def external_simulate():
    sol_fs=solve_functions(params={"R0":8.090592910399652,"R1":19.875424111891668,"Q1":0.01,"alpha1":0.615753920655586,"C1":8.977927925301545e-24,"Q2":0.0070130021403192735,"alpha2":0.281135665228085,},
                                    derivative_vars=[0,2],
                                    dt=dt,
                                    end=dt*10,
                                    current_idx=4,
                                    source_func=lambda t:0.3*np.sin(math.pi*2*8*t),
                                    cpe_locs=[5,6])
    return sol_fs.simulate(), sol_fs.t_array, [sol_fs.source_func(t) for t in sol_fs.t_array]
external_simulate()
