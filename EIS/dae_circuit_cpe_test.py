import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scikits.odes import dae
from scipy.optimize import fsolve
class solve_functions:
    def __init__(self, params):
        self.params=params
        self.t_array=[0]
        self.cpe_arrays={"cpe_1":[0]}
        self.t_var=0
        self.dt=0
    def source(self, t):
        return np.sin(5*t)
    def residual(self, t, x, xdot, result):
        t_gap=t-self.t_var
        increased_time=False

        if t_gap<0:
            self.dt=self.dt+t_gap
        elif t_gap==0:
            pass
        else:
            self.dt=t_gap
            increased_time=True
        #print(t, self.t_var, dt, increased_time)
        if increased_time==False:
            self.t_array[-1]=t
            self.cpe_arrays["cpe_1"][-1]=x[3]
        else:
            self.t_array.append(t)
            self.cpe_arrays["cpe_1"].append(x[3])
        #print(self.t_array)
        #print("t=", t, increased_time, "t_gap=", t_gap, "t_var=", self.t_var, "dt=", self.dt, "2")
        print(t)
        self.t_var=t
        result[0]=x[3] - x[2]
        result[1]=self.params["C1"]*xdot[1] - x[3]
        result[2]=self.source(t) - x[0]
        result[3]=-self.cpe(self.t_array, self.cpe_arrays["cpe_1"], self.params["Q1"], self.params["alpha1"]) + x[0] - x[1]
        print(result)
        #print(-self.cpe(self.t_array, self.cpe_arrays["cpe_1"], self.params["Q1"], self.params["alpha1"]) , x[0] , x[1], result[3])
        #print(result)

    def cpe(self, t, i, Q, alpha):

        #print(t, i, "1")
        if t[-1]==0:
            return 0

        linear_t=np.arange(0, t[-1]+self.dt, self.dt)
        #poly_vals=Polynomial.fit(t, i, deg=2)
        #linear_t, linear_i=poly_vals.linspace(np.ceil(t[-1]/self.dt), [0, t[-1]])
        linear_i=np.interp(linear_t, t, i)
        #dt=linear_t[1]-linear_t[0]
        #print(linear_t, linear_i, "2")
        powered_t=np.zeros(len(linear_t))
        powered_t[1:]= np.power(linear_t[1:], alpha-1)
        #print(powered_t)
        flipped_times=np.flip(powered_t)
        #print(flipped_times)
        convolv=np.sum(np.multiply(flipped_times, linear_i))
        print("voltage=", self.dt*convolv/(Q*gamma(alpha)))
        return self.dt*convolv/(Q*gamma(alpha))


z0=[0,0,0, 0]
zp0=[0,0,0, 0]
sol_fs=solve_functions(params={"R1":10, "C1":1e-3, "Q1":1, "alpha1":0.5})
dt=1e-9
sol_fs.dt=dt
times=np.arange(0, dt*1000, dt)
charge_array=np.zeros(len(times))
interval_potential=[sol_fs.source(t) for t in times]

solver = dae('ida', sol_fs.residual,
             first_step_size=dt,
             atol=1e-6,
             rtol=1e-6,
             algebraic_vars_idx=[0, 2, 3],
             one_step_compute=True,
             old_api=False)
solver.init_step(t0=0, y0=z0, yp0=zp0)
times=np.arange(0, dt*2, dt)
y_retn=np.zeros(len(times))
for i in range(1, len(times)):
    solver.step(times[i])
"""for i in [1e-3]:#, 1e-4, 1e-5]:
    sol_fs=solve_functions(params={"R1":10, "C1":i, "Q1":1, "alpha1":0.5})
    solver = dae('ida', sol_fs.residual,
                 compute_initcond='yp0',
                 first_step_size=1e-18,
                 atol=1e-6,
                 rtol=1e-6,
                 algebraic_vars_idx=[0, 2, 3],
                 compute_initcond_t0 = 60,
                 old_api=False)
    solution = solver.solve(np.linspace(0, 1e-19, 100), z0, zp0)
    plt.plot(solution.values.t, solution.values.y[:, 2], label=i)

plt.legend()
plt.show()"""
