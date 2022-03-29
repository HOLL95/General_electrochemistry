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
from scipy import fft as syft
ZARC={"z2":{"p1":("Q1", "alpha1"), "p2":"R1"}}
"""translator=EIS()
circuit_artist(ZARC)
ax=plt.gca()
ax.set_axis_off()
plt.show()"""
frequency_powers=np.arange(1, 2, 0.2)
frequencies=[10**x for x in frequency_powers]
omegas=np.multiply(frequencies, 2*math.pi)
bounds={
"R1":[0, 1000],
"Q1":[0, 1e-2],
"alpha1":[0.1, 0.9],
"R2":[0, 1000],
"W1":[1, 200]
}
true_params={"R1":10, "Q1":0.001, "alpha1":0.5}
param_names=["R1", "Q1", "alpha1"]
sim=EIS_genetics()
fig, ax=plt.subplots()
#sim.plot_data(sim.dict_simulation(ZARC, frequencies, [], [true_params[x] for x in param_names],  param_names),ax)

num_oscillations=5
time_start=0
sampling_rate=200


def cpe1(denom, i, charge_array, time_array, dt):
    total=0
    if i!=0:
        time_array=np.flip(time_array)
        total=np.sum(np.multiply(time_array, charge_array))
    return dt*total/denom


alphas=[0.7, 0.8, 0.9]
fig, ax=plt.subplots(1,len(alphas))
for z in range(0, len(alphas)):
    for m in range(0, len(frequencies)):

        true_params["alpha1"]=alphas[z]
        q=0
        ramp=0.01
        step=0.0001
        sampling_rate=4000
        freq_percentage=0.05
        sampling_frequency=1/sampling_rate
        interval_points=sampling_rate*num_oscillations
        total_i_fft=np.zeros(interval_points, dtype="complex")
        total_v_fft=np.zeros(interval_points, dtype="complex")
        impede=np.zeros((len(frequencies), 2))
        for j in range(0, 1):
            period=(num_oscillations/frequencies[m])
            interval_times=np.linspace(0, period, interval_points)
            dt=interval_times[1]-interval_times[0]
            interval_potential=0.1*np.sin(omegas[m]*interval_times)#interval_times*ramp
            convolv=np.zeros(interval_points)
            current_2=np.zeros(interval_points)
            convolv[1:]=np.power(interval_times[1:], true_params["alpha1"]-1)
            denom=true_params["Q1"]*special.gamma(true_params["alpha1"])
            for i in range(0, interval_points):
                vtotal=interval_potential[i]
                vcpe=cpe1(denom, i, current_2[:i], convolv[:i], dt)
                #print(vcpe)
                current_2[i]=(vtotal-vcpe)/true_params["R1"]
            ax[z].plot(interval_potential, current_2, label="f={0}Hz".format(round(frequencies[m], 1)))
        ax[z].set_xlabel("Potential(V)")
        ax[z].set_ylabel("Current(A)")
        ax[z].set_title("$\\alpha={0}$".format(alphas[z]))

plt.show()
