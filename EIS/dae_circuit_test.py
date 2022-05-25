import matplotlib.pyplot as plt
import numpy as np
from scikits.odes import dae
class solve_functions:
    def __init__(self, params):
        self.params=params
        self.t=[0]
        self.t_var=0
        self.record_array=[]
        self.dt=0
    def source(self, t):
        return np.sin(5*t)
    def residual(self, t, x, xdot, result):
        t_gap=t-self.t_var
        self.record_array.append(t)
        increased_time=False
        if t_gap<0:
            self.dt=self.dt+t_gap
        elif t_gap==0:
            pass
        else:
            self.dt=t_gap
            increased_time=True
        if increased_time==False:
            self.t[-1]=t
        else:
            self.t.append(t)


        result[0]=x[0]/self.params["R1"]-x[1]/self.params["R1"]-x[2]
        result[1]=-x[0]/self.params["R1"]+x[1]/self.params["R1"]+self.params["C1"]*xdot[1]
        result[2]=-x[0]-self.source(t)
        self.t_var=t

z0=[0,0,0]
zp0=[0,0,0]
for i in [1e-3]:#, 1e-4, 1e-5]:
    sol_fs=solve_functions(params={"R1":10, "C1":i})
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
def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))
def non_decreasing(L):
    return all(x<=y for x, y in zip(L, L[1:]))
plt.legend()
plt.show()
plt.plot(np.log10(sol_fs.t))
plt.plot(np.log10(sol_fs.record_array))
print(non_decreasing(sol_fs.t), non_decreasing(sol_fs.record_array))
plt.show()
