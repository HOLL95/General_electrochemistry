import numpy as np
import sympy
import matplotlib.pyplot as plt
import copy
param_names=["K", "r"]
params={x:sympy.symbols(x) for x in param_names}
t=sympy.symbols("t")
p0=sympy.symbols("P0")
P_t=params["K"]/(1+(sympy.exp(-params["r"]*t)*(params["K"]-p0)/p0))

P_t=(params["K"]*p0*sympy.exp(params["r"]*t))/(params["K"]+p0*(sympy.exp(params["r"]*t)-1))
print(P_t)
sympy.pprint(P_t)
partial_derivs=[sympy.diff(P_t, x) for x in [params[x] for x in param_names]]
for i in range(0, len(partial_derivs)):
    print(partial_derivs[i])
def logistic_sensitivity(t, params, P0):
    K,r=params
    ert=np.exp(r*t)
    sens_1=-K*P0*ert/np.square(K + P0*(ert - 1)) + P0*ert/(K + P0*(ert - 1))
    sens_2=(-K*(P0**2)*t*np.exp(2*r*t))/np.square(K + P0*(ert - 1)) + (K*P0*t*ert)/(K + P0*(ert - 1))
    
    timeseries=K*P0*ert/(K + P0*(ert - 1))
    return np.vstack((timeseries, sens_1, sens_2))
def logistic_ts(t, params, P0):
    K,r=params
    ert=np.exp(r*t)
    return (K*P0*ert)/(K + P0*(ert - 1))
growth_rate=0.5
carrying_capacity=200
orig_params=[carrying_capacity, growth_rate]
delta=1e-3
init_pop=2
num_points=50
time=np.linspace(0,25, num_points)
logistic_sens=logistic_sensitivity(time, orig_params, init_pop)
timeseries=np.zeros((2, num_points))
numeric_sens=np.zeros((2, num_points))
intervals=[-1, 1]
for i in range(0, len(param_names)):
   
    for j in range(0, len(intervals)):
        sim_params=copy.deepcopy(orig_params)
        sim_params[i]=orig_params[i]+(delta*orig_params[i]*intervals[j])
        print(sim_params)
        timeseries[j, :]=logistic_ts(time, sim_params, init_pop)
        
    
    numeric_sens[i,:]=np.divide(np.subtract(timeseries[1],timeseries[0]), 2*delta)

plt.plot(time, logistic_sens[0,:])
plt.show()
gif, ax=plt.subplots(1,1)
ax.plot(time, carrying_capacity*logistic_sens[1, :], label=param_names[0])
ax.scatter(time, numeric_sens[0, :])
ax.plot(0, 0, color="red", label=param_names[1])
ax.legend()

ax.plot(time, growth_rate*logistic_sens[2,:], color="red")
ax.scatter(time, numeric_sens[1, :], color="red")
plt.show()