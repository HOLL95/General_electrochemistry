import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scikits.odes import dae
class solve_functions:
    def __init__(self, params):
        self.params=params
        self.t_array=[]
        self.cpe_arrays={"cpe_1":[]}
        self.t_var=0
        self.dt=0
    def source(self, t):

        return np.sin(5*t)
    def residual(self, t, x, xdot, result):

        R1=self.params["R1"]
        C1=self.params["C1"]
        result[0]=-x[2] + x[0]/R1 - x[1]/R1
        result[1]=-x[0]/R1 + x[1]/R1+C1*xdot[1]
        result[2]=self.source(t) - x[0]





z0=[0,0,0]
zp0=[0,0,0]
sol_fs=solve_functions(params={"R1":10, "C1":1e-3, "Q1":1, "alpha1":0.5})
dt=0.1
solver = dae('ida', sol_fs.residual,
             compute_initcond='yp0',
             first_step_size=dt,
             atol=1e-6,
             rtol=1e-6,
             algebraic_vars_idx=[0, 2],
             compute_initcond_t0 = 60,
             one_step_compute=True,
             old_api=False)
solver.init_step(t0=0, y0=z0, yp0=zp0)
times=np.arange(0, 1, dt)
y_retn=np.zeros(len(times))

for i in range(1, len(times)):
    print(times[i])
    yn=solver.step(times[i])
    print(yn)

"""for i in [1e-3]:#, 1e-4, 1e-5]:
    sol_fs=solve_functions(params={"R1":10, "C1":i, "Q1":1, "alpha1":0.5})
    solver = dae('ida', sol_fs.residual,
                 compute_initcond='yp0',
                 first_step_size=1e-18,
                 atol=1e-6,
                 rtol=1e-6,
                 algebraic_vars_idx=[0, 2],
                 compute_initcond_t0 = 60,
                 old_api=False)
    solution = solver.solve(np.linspace(0, 2, 100), z0, zp0)
    plt.plot(solution.values.t, solution.values.y[:, 2], label=i)

plt.legend()
plt.show()"""
