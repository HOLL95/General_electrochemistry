import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import gamma
from scikits.odes import dae
from scipy.optimize import fsolve
import math
class solve_functions:
    def __init__(self, params, high_point_flag=False):
        self.params=params
        self.t_array=[0]
        cpe_str="cpe_"
        self.cpe_arrays={"cpe_1":[0]}
        self.t_var=0
        self.dt=0
        self.hpf=high_point_flag
        self.cpe_pot=[]
        self.cpe_coeffs=np.multiply(1/6, np.flip([11, -18,9, -2]))
        self.counter=1
    def source(self, t):
        return 0.3*np.sin(2*math.pi*8*t)
    def residual(self,x):

        #print(self.cpe_arrays["cpe_1"], "~"*30)
        t_gap=self.t-self.t_var
        increased_time=False
        if t_gap==0:
            self.cpe_arrays["cpe_1"][-1]=x[3]
            self.t_array[-1]=self.t
        else:
            self.t_array.append(self.t)
            self.cpe_arrays["cpe_1"].append(x[3])
        self.t_var=self.t
        #print(self.counter)
        #print(x)
        result=[0,0,0,0]
        result[0]=x[3] - x[2]
        result[1]=-x[3] + x[1]/self.params["R0"]
        result[2]=  self.source(self.t) - x[0]
        cpe_pot=self.cpe(self.t_array, self.cpe_arrays["cpe_1"], self.params["Q1"], self.params["alpha1"])
        result[3]=-cpe_pot + x[0] - x[1]#
        #if self.counter<6:
        #    cpe_pot=0

        if t_gap==0:
            self.cpe_pot[-1]=cpe_pot
        else:
            self.cpe_pot.append(cpe_pot)

            #if self.hpf==True:
            #    print(self.cpe_pot, "potential")
        #print(self.t)
        #print(x, "x")
        #print(result, "result")
        return result

    def cpe(self, linear_t, linear_i, Q, alpha):
        #print(linear_t, linear_i, "2")
        #print(linear_i)
        #print(t, i, "1")
        if linear_t[-1]==0:
            return 0

        #linear_t=np.arange(0, t[-1]+self.dt, self.dt)
        #print(linear_t, t, i)
        #print(len(t), len(i))
        #poly_vals=Polynomial.fit(t, i, deg=2)
        #linear_t, linear_i=poly_vals.linspace(np.ceil(t[-1]/self.dt), [0, t[-1]])
        #linear_i=np.interp(linear_t, t, i)
        #print(linear_i)
        #dt=linear_t[1]-linear_t[0]

        powered_t=np.zeros(len(linear_t))
        powered_t[1:]= np.power(linear_t[1:], alpha-1)
        #print(powered_t)
        flipped_times=np.flip(powered_t)
        #print(flipped_times)
        convolv=np.sum(np.multiply(flipped_times, linear_i))
        #print("voltage=", self.dt*convolv/(Q*gamma(alpha)))

        if self.counter<6:
            return_val=self.dt*convolv/(Q*gamma(alpha))
        else:
            if self.counter==6:
                self.cpe_pot=np.multiply(self.cpe_pot, 0)
                self.cpe_pot=list(self.cpe_pot)
            #print(self.cpe_pot)
            #print(self.cpe_pot[self.counter-5:self.counter])
            if self.hpf==False:
                return_val=self.dt*convolv/(Q*gamma(alpha))#+(np.sum(np.multiply(self.cpe_coeffs, self.cpe_pot[-5:-1])))
            else:

                return_val=self.dt*convolv/(Q*gamma(alpha))+(np.sum(np.multiply(self.cpe_coeffs, self.cpe_pot[self.counter-5:self.counter-1])))
                print(self.cpe_pot[self.counter-5:self.counter-1], "new_potential")
                print(return_val, "RV", (np.sum(np.multiply(self.cpe_coeffs, self.cpe_pot[self.counter-5:self.counter-1]))), self.counter, self.dt*convolv/(Q*gamma(alpha))+(np.sum(np.multiply(self.cpe_coeffs, self.cpe_pot[self.counter-5:self.counter-1]))))
                print(self.dt*convolv/(Q*gamma(alpha)))
        return return_val
from time_domain_simulator_class import time_domain
import copy
translated_circuit={"z1":("Q1", "alpha1"), "z2":"R0"}
sim_dict={ "R0":1, "Q1":1e-2, "alpha1":0.5}
td=time_domain(translated_circuit, params=sim_dict)
c, t, p=td.simulate()
z0=[0,0,0, 0]
e1=0
e2=0
i_cpe=0
zp0=[0,0,0, 0]
sol_fs=solve_functions(params={"R0":1,  "Q1":1e-2, "alpha1":0.5})
dt=0.0001
sol_fs.dt=dt
times=np.arange(0, dt*1000, dt)
charge_array=np.zeros((2,len(times)))
interval_potential=[sol_fs.source(t) for t in times]
for i in range(1, len(times)):
    #print("~"*30)
    vtot=interval_potential[i]
    curr_time_vec=times[:i]
    #charge_array[0][i]=vtot-sol_fs.cpe(curr_time_vec, charge_array[1][:i], sol_fs.params["Q1"], sol_fs.params["alpha1"])
    sol_fs.t=times[i]
    e1,e2,charge_array[1][i],i_cpe=fsolve(sol_fs.residual, [e1, e2, charge_array[1][i-1], i_cpe])
    #print(e1,e2,charge_array[1][i],i_cpe)

for i in range(1, len(times)):
    vtot=interval_potential[i]
    curr_time_vec=times[:i]
    charge_array[0][i]=vtot-sol_fs.cpe(curr_time_vec, charge_array[0][:i], sol_fs.params["Q1"], sol_fs.params["alpha1"])



#plt.plot(p, c)
#plt.plot(t, p)
#plt.plot(times, interval_potential)
plt.plot(interval_potential[1:], sol_fs.cpe_pot)
z=copy.deepcopy(sol_fs.cpe_pot)
for i in range(5, len(sol_fs.cpe_pot)):
    z[i]=(z[i-1])+(np.sum(np.multiply(sol_fs.cpe_coeffs, z[i-5:i-1])))
plt.plot(interval_potential[1+len(interval_potential)//2:], z[len(interval_potential)//2:])
plt.show()
old_cpe_pot=copy.deepcopy(sol_fs.cpe_pot)
z0=[0,0,0, 0]
e1=0
e2=0
i_cpe=0
zp0=[0,0,0, 0]
sol_fs=solve_functions(params={"R0":1,  "Q1":1e-2, "alpha1":0.5}, high_point_flag=True)
print(sol_fs.cpe_arrays["cpe_1"], "hey")
dt=0.0001
sol_fs.dt=dt
times=np.arange(0, dt*10, dt)
charge_array=np.zeros((2,len(times)))
interval_potential=[sol_fs.source(t) for t in times]
for i in range(1, len(times)):
    #print("~"*30)
    vtot=interval_potential[i]
    curr_time_vec=times[:i]
    #charge_array[0][i]=vtot-sol_fs.cpe(curr_time_vec, charge_array[1][:i], sol_fs.params["Q1"], sol_fs.params["alpha1"])
    sol_fs.t=times[i]
    sol_fs.counter=i
    e1,e2,charge_array[1][i],i_cpe=fsolve(sol_fs.residual, [e1, e2, charge_array[1][i-1], i_cpe])
    if i>5:
        print(z[i], "Calc_pot",  (np.sum(np.multiply(sol_fs.cpe_coeffs, old_cpe_pot[i-5:i-1]))), old_cpe_pot[i-1],  (np.sum(np.multiply(sol_fs.cpe_coeffs, old_cpe_pot[i-5:i-1]))) +  old_cpe_pot[i-1], i)
        #print(z[i-1])
    print(z[i-5:i-1], "old_potential")
    #print(e1,e2,charge_array[1][i],i_cpe)




plt.plot(interval_potential, charge_array[1,:])
plt.show()
"""solver = dae('ida', sol_fs.residual,
             compute_initcond='yp0',
             first_step_size=dt,
             atol=1e-6,
             rtol=1e-6,
             algebraic_vars_idx=[0, 2, 3],
             compute_initcond_t0 = 60,
             one_step_compute=True,
             old_api=False)
solver.init_step(t0=0, y0=z0, yp0=zp0)
times=np.arange(0, dt*2, dt)
y_retn=np.zeros(len(times))
for i in range(1, len(times)):
    solver.step(times[i])"""
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
