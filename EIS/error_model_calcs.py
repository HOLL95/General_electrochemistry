import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scikits.odes import dae
from scipy.optimize import fsolve
import math
class solve_functions:
    def __init__(self, params, derivative_vars, dt, end, current_idx, source_func, cpe_locs=None):
        self.params=params
        self.t_array=np.arange(0, end, dt)
        self.num_cpe=0
        self.cpe_arrays={}
        self.cpe_keys=["cpe_{0}".format(x) for x in range(1, self.num_cpe+1)]
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
        result=[0,0,0]
        xdot=np.zeros(len(result))
        for i in range(0, len(self.derivative_vars)):
            xdot[self.derivative_vars[i]]=(x[self.derivative_vars[i]]-self.deriv_history[i])/self.dt
        #parameter_area
        R1=self.params["R1"]
        C1=self.params["C1"]
        result[0]=-x[2]+(C1)*xdot[0]+(-C1)*xdot[1]
        result[1]=x[1]/R1+(-C1)*xdot[0]+(C1)*xdot[1]
        result[2]=-self.source_func(t) - x[0]

        return result

    def cpe(self, linear_t, linear_i, Q, alpha):

        #print(t, i, "1")
        if linear_t[-1]==0:
            return 0
        powered_t=np.zeros(len(linear_t))
        powered_t[1:]= np.power(linear_t[1:], alpha-1)
        flipped_times=np.flip(powered_t)
        #print(flipped_times)
        convolv=np.sum(np.multiply(flipped_times, linear_i))
        return self.dt*convolv/(Q*gamma(alpha))
    def simulate(self):
        var_array=[0,0,0]
        current_array=np.zeros(len(self.t_array))
        for i in range(1, len(self.t_array)):
            self.t_counter=i
            var_array=fsolve(self.residual, var_array)
            for j in range(0, len(self.derivative_vars)):
                self.deriv_history[j]=var_array[self.derivative_vars[j]]
            for j in range(0, self.num_cpe):
                self.cpe_arrays[self.cpe_keys[j]][self.t_counter]=var_array[self.cpe_locs[j]]
            self.current_history[i]=var_array[self.current_idx]
        return self.current_history
def RMSE(y, y1):
    return np.sqrt(np.sum(np.square(np.subtract(y, y1))))
def impedance_simulate(sim_class, frequencies, amplitude=1e-3, num_osc=2):
    nyquist=np.zeros((2, len(frequencies)))

    for i in range(0, len(frequencies)):
        freq_func=lambda t: amplitude*np.sin(2*math.pi*frequencies[i]*t)
        end=num_osc*2*math.pi/frequencies[i]
        old_current=sim_class.simulate()
        sfs=[100, 200, 400, 600, 800, 1000,1500, 2000, 4000, 5000, 6000, 7000, 8000]
        error=np.zeros(len(sfs))

        for j in range(0, len(sfs)):
            dt=1/(frequencies[i]*sfs[j])
            new_class=solve_functions(sim_class.params, sim_class.derivative_vars, dt=dt, end=end, current_idx=sim_class.current_idx,
                            source_func=freq_func, cpe_locs=sim_class.cpe_locs)
            current=new_class.simulate()
            potential=[freq_func(t) for t in new_class.t_array]
            if j==0:
                old_current=np.zeros(len(current))
                old_time=new_class.t_array
            interp_current=np.interp(new_class.t_array, old_time, old_current)
            error[j]=RMSE(current, interp_current)
            old_current=current
            old_time=new_class.t_array

            #plt.plot(potential, current)
        plt.loglog(sfs, error)
        plt.show()

        i_fft=np.fft.fft(current)
        v_fft=np.fft.fft(potential)


        abs_i=abs(i_fft)
        abs_v=abs(v_fft)
        ft_freq=np.fft.fftfreq(len(current), dt)
        i_freq=np.where(abs_i==max(abs_i))
        v_freq=np.where(abs_v==max(abs_v))
        i_peak=i_fft[i_freq]
        v_peak=v_fft[v_freq]
        z_freq=v_peak[0]/i_peak[0]
        nyquist[:, i]=[np.real(z_freq), -np.imag(z_freq)]

    return nyquist
dt=0.001


sol_fs=solve_functions(params={"R1":1, "C1":1, "Q1":1, "alpha1":0.5},
                        derivative_vars=[0,1],
                        dt=dt,
                        end=dt*1000,
                        current_idx=2,
                        source_func=lambda t:0.03*np.sin(math.pi*2*8*t),
                        cpe_locs=[])


imp=impedance_simulate(sol_fs, [10**x for x in np.linspace(-2, 5, 10)])
plt.scatter(imp[0,:], imp[1, :])
plt.show()
