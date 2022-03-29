import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import matplotlib.pyplot as plt
import numpy as np
from EIS_class import EIS
from EIS_optimiser import EIS_genetics, EIS_optimiser
from circuit_drawer import circuit_artist
import math
from scipy import special
ZARC={"z2":{"p1":("Q1", "alpha1"), "p2":"R1"}}
"""translator=EIS()
circuit_artist(ZARC)
ax=plt.gca()
ax.set_axis_off()
plt.show()"""
frequency_powers=np.arange(0, 4, 0.1)
frequencies=[10**x for x in frequency_powers]
omegas=np.multiply(frequencies, 2*math.pi)
bounds={
"R1":[0, 1000],
"Q1":[0, 1e-2],
"alpha1":[0.1, 0.9],
"R2":[0, 1000],
"W1":[1, 200]
}
true_params={"R1":10, "Q1":0.001, "alpha1":0.7}
param_names=["R1", "Q1", "alpha1"]
sim=EIS_genetics()
fig, ax=plt.subplots()
#sim.plot_data(sim.dict_simulation(ZARC, frequencies, [], [true_params[x] for x in param_names],  param_names),ax)
#plt.show()
num_oscillations=5
time_start=0
sampling_rate=200


qmag=true_params["Q1"]
alpha=true_params["alpha1"]
r=true_params["R1"]
step=0.001
ramp =0.01
t_end=0.5
gamma=special.gamma(alpha)
print(gamma, "gamma")
length=int(t_end/step)
qstepu=[]#np.zeros(length)
convolv=[]#np.zeros(length)
q=0
results=[]
def cpe(Q, gamma, i, t, charge_array, time_array):
    total=0
    #print("cpe", time_array)
    #print(time_array)
    #plt.plot(time_array)
    #plt.show()
    for j in range(0, i):
        total+=time_array[i-j]*charge_array[j]

    return total/(Q*gamma)
def cpe1(Q, gamma, i, t, charge_array, time_array, dt):
    total=0
    #print("cpe1", time_array)
    #plt.plot(time_array)
    #plt.show()
    if i!=0:
        time_array=np.flip(time_array)

        total=np.sum(np.multiply(time_array, charge_array))



    return dt*total/(Q*gamma)
def naive_vcpe(times, current,  denom, dt, i):

    #time_multiples=np.power(np.flip(times), alpha-1)
    #print(times)
    total=0
    #print(time_array)
    for j in range(0, i):
        total+=times[i-j]*current[j]
    return (dt/(denom))*total
t_plot=[]
vpe_recorder=[]
for i in range(0, length):
    t=i*step
    t_plot.append(t)
    if t==0:
        convolv.append(0)
    else:
        convolv.append(np.power(t, alpha-1))
        #print(convolv[-1])
    vtotal=t*ramp
    vcpe=cpe(qmag, gamma, i, t, qstepu, convolv)
    vpe_recorder.append(vcpe)
    #print(vcpe)
    current=(vtotal-vcpe)/r
    qstepu.append(current*step)
    results.append(current)
plt.plot(t_plot, vpe_recorder)
t_plot=[]
vpe_recorder=[]
qstepu=[]#np.zeros(length)
print(convolv)

q=0
results=[]
times_2=np.arange(0, t_end, step)
convolv=np.zeros(len(times_2))
dt=step
current_2=np.zeros(length)
convolv[1:]=np.power(times_2[1:], true_params["alpha1"]-1)
print(convolv)
print(len(convolv), len(current_2))
for i in range(0, length):
    t=i*step
    t_plot.append(t)
    vtotal=t*ramp
    vcpe=cpe1(qmag, gamma, i, t, current_2[:i], convolv[:i], dt)
    vpe_recorder.append(vcpe)
    #print(vcpe)
    current_2[i]=(vtotal-vcpe)/r
plt.plot(times_2, vpe_recorder)
plt.show()
sampling_frequency=0.001
sampling_rate=int(1/sampling_frequency)
interval_points=sampling_rate*num_oscillations
times=np.zeros(interval_points*len(frequencies))
potential=np.zeros(interval_points*len(frequencies))
current=np.zeros(interval_points*len(frequencies))
denom=special.gamma(true_params["alpha1"])*true_params["Q1"]
vpe_recorder=[]
for i in range(0, 1):

    period=t_end
    interval_times=np.linspace(0, period, interval_points)
    dt=interval_times[1]-interval_times[0]
    print(dt)
    interval_potential=np.multiply(interval_times, ramp)#np.sin(omegas[i]*interval_times)
    potential[i*interval_points:(i+1)*interval_points]=interval_potential
    times[i*interval_points:(i+1)*interval_points]=np.add(interval_times, times[int((i*interval_points)-1)])
    interval_current=np.zeros(interval_points)
    #plt.plot(interval_times, interval_potential)
    #plt.show()
    counter=0
    #interval_current[1]=interval_potential[0]/true_params["R1"]
    times=[]
    current=[0]
    dt=step
    for z in range(0, len(t_plot)):
        if z==0:
            times=[0]
            vpe=0
        else:
            times.append(np.power(z*dt,alpha-1))
            #print(times, current)
            vpe=naive_vcpe(times,current , denom, dt, z)
        current.append(((z*dt*ramp)-vpe)/true_params["R1"])
        vpe_recorder.append(vpe)
        #interval_current[z]=(interval_potential[z-1]-vpe)/true_params["R1"]
    plt.plot(t_plot, vpe_recorder)
    plt.show()



"""sampling_frequency=0.001
sampling_rate=int(1/sampling_frequency)
interval_points=sampling_rate*num_oscillations
times=np.zeros(interval_points*len(frequencies))
potential=np.zeros(interval_points*len(frequencies))
current=np.zeros(interval_points*len(frequencies))
denom=special.gamma(true_params["alpha1"])*true_params["Q1"]
for i in range(0, 1):

    period=(num_oscillations/frequencies[i])
    interval_times=np.linspace(0, period, interval_points)
    dt=interval_times[1]-interval_times[0]
    print(dt)
    interval_potential=np.multiply(interval_times, step)#np.sin(omegas[i]*interval_times)
    potential[i*interval_points:(i+1)*interval_points]=interval_potential
    times[i*interval_points:(i+1)*interval_points]=np.add(interval_times, times[int((i*interval_points)-1)])
    interval_current=np.zeros(interval_points)
    #plt.plot(interval_times, interval_potential)
    #plt.show()
    counter=0
    #interval_current[1]=interval_potential[0]/true_params["R1"]
    for z in range(0, interval_points):
        vpe=naive_vcpe(interval_times[:z], dt*interval_current[:z], denom, dt)
        interval_current[z]=(interval_potential[z-1]-vpe)/true_params["R1"]
    plt.plot(interval_times, interval_current)
    plt.show()
"""
