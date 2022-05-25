import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scikits.odes import dae
from scipy.optimize import fsolve
class solve_functions:
    def __init__(self, params, derivative_vars):
        self.params=params
        self.t_array=[0]
        self.cpe_arrays={"cpe_1":[0]}
        self.derivative_vars=derivative_vars
        self.deriv_history=[0 for x in self.derivative_vars]
        self.t_var=0
        self.dt=0


    def source(self, t):
        return np.sin(5*t)
    def residual(self, x):
        t_gap=self.t-self.t_var
        if t_gap==0:
            self.cpe_arrays["cpe_1"][-1]=x[3]
            self.t_array[-1]=self.t
            increased_time=False
        else:
            self.t_array.append(self.t)
            self.cpe_arrays["cpe_1"].append(x[3])
            increased_time=True
        self.t_var=self.t
        result=[0,0,0,0]
        xdot=np.zeros(len(result))
        for i in range(0, len(self.derivative_vars)):
            xdot[self.derivative_vars[i]]=(x[self.derivative_vars[i]]-self.deriv_history[i])/self.dt

        result[0]=x[3] - x[2]
        result[1]=self.params["C1"]*xdot[1] - x[3]
        result[2]=self.source(self.t) - x[0]
        result[3]=-self.cpe(self.t_array, self.cpe_arrays["cpe_1"], self.params["Q1"], self.params["alpha1"]) + x[0] - x[1]
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
for q in [1e-5, 1e-3, 1, 100]:
    sol_fs=solve_functions(params={"R1":1, "C1":1e-7, "Q1":q, "alpha1":0.5}, derivative_vars=[1])
    dt=0.001
    e1=0
    e2=0
    i_cpe=0
    sol_fs.dt=dt
    times=np.arange(0, dt*1000, dt)
    charge_array=np.zeros((len(times)))
    interval_potential=[sol_fs.source(t) for t in times]
    for i in range(1, len(times)):
        #vtot=interval_potential[i]
        #curr_time_vec=times[:i]
        #charge_array[0][i]=vtot-sol_fs.cpe(curr_time_vec, charge_array[:i], sol_fs.params["Q1"], sol_fs.params["alpha1"])
        sol_fs.t=times[i]
        e1,e2,charge_array[i],i_cpe=fsolve(sol_fs.residual, [e1, e2, charge_array[-1], i_cpe])
        for i in range(0, len(sol_fs.derivative_vars)):
            sol_fs.deriv_history[i]=e2
    plt.plot(interval_potential, charge_array)
plt.show()
