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
ZARC={"z1":("Q1", "alpha1"), "z2":"R1"}
def zarc(frequency, q, alpha, r):
    return r+(1/(q*(1j*frequency)**alpha))
"""translator=EIS()
circuit_artist(ZARC)
ax=plt.gca()
ax.set_axis_off()
plt.show()"""
frequency_powers=np.arange(0, 6, 0.2)
frequencies=[10**x for x in frequency_powers]
omegas=np.multiply(frequencies, 2*math.pi)
bounds={
"R1":[0, 1000],
"Q1":[0, 1e-2],
"alpha1":[0.1, 0.9],
"R2":[0, 1000],
"W1":[1, 200]
}
true_params={"R1":10, "Q1":0.001, "alpha1":0.8}
param_names=["R1", "Q1", "alpha1"]
sim=EIS_genetics()
normaliser=EIS(circuit=ZARC, parameter_bounds=bounds, parameter_names=param_names, test=True, fitting=True)
sim1=normaliser.simulate([true_params[x] for x in param_names], omegas)
fig, ax=plt.subplots(1, 2)
sim.plot_data(sim.dict_simulation(ZARC, omegas, [], [true_params[x] for x in param_names],  param_names),ax[1], label="Analytic frequency simulation")
num_oscillations=3
time_start=0
sampling_rate=200


def cpe1(denom, i, charge_array, time_array, dt):
    total=0
    if i!=0:
        time_array=np.flip(time_array)
        total=np.sum(np.multiply(time_array, charge_array))
    return dt*total/denom
q=0
ramp=0.01
step=0.0001
sampling_rate=3000
freq_percentage=0.05
sampling_frequency=1/sampling_rate
interval_points=sampling_rate*num_oscillations
total_i_fft=np.zeros(interval_points, dtype="complex")
total_v_fft=np.zeros(interval_points, dtype="complex")
impede=np.zeros((len(frequencies), 4))
for j in range(0, len(frequencies)):
    period=(num_oscillations/frequencies[j])
    interval_times=np.linspace(0, period, interval_points)
    dt=interval_times[1]-interval_times[0]
    interval_potential=0.01*np.sin(omegas[j]*interval_times)
    convolv=np.zeros(interval_points)
    current_2=np.zeros(interval_points)
    convolv[1:]=np.power(interval_times[1:], true_params["alpha1"]-1)
    denom=true_params["Q1"]*special.gamma(true_params["alpha1"])
    for i in range(0, interval_points):
        vtotal=interval_potential[i]
        vcpe=cpe1(denom, i, current_2[:i], convolv[:i], dt)
        #print(vcpe)
        current_2[i]=(vtotal-vcpe)/true_params["R1"]
    zarc_pred=zarc(omegas[j], true_params["Q1"], true_params["alpha1"], true_params["R1"])
    #ax[1].scatter(np.real(zarc_pred), -np.imag(zarc_pred))
    i_fft=syft.fft(current_2)
    v_fft=syft.fft(interval_potential)
    fft_freq=syft.fftfreq(interval_points, dt)
    total_i_fft=np.add(total_i_fft, i_fft)
    total_v_fft=np.add(total_v_fft, v_fft)
    peak_loc=np.where((fft_freq>(frequencies[j]*(1-freq_percentage))) & (fft_freq<(frequencies[j]*(1+freq_percentage))))
    abs_i=abs(i_fft)
    abs_v=abs(v_fft)
    i_peak=i_fft[np.where(abs_i==max(abs_i))]
    v_peak=v_fft[np.where(abs_v==max(abs_v))]


    z_freq=v_peak[0]/i_peak[0]
    potential_peak_idx=np.where((interval_times>(1.9/frequencies[j])) & (interval_times<(3.1/frequencies[j])) )
    ss_potential=interval_potential[potential_peak_idx]
    ss_time=interval_times[potential_peak_idx]
    max_potential_time=ss_time[np.where(ss_potential==max(ss_potential))][0]
    half_window=1/(2*frequencies[j])
    current_peak_idx=np.where((interval_times>(max_potential_time-half_window))&(interval_times<(max_potential_time+half_window)))
    ss_current=current_2[current_peak_idx]
    ss_current_time=interval_times[current_peak_idx]
    max_current_time=ss_current_time[np.where(ss_current==max(ss_current))][0]
    time_diff=abs(max_potential_time-max_current_time)
    phase=2*math.pi*(time_diff*frequencies[j])

    """plt.plot(ss_time, ss_potential)
    plt.axvline(max_potential_time)
    plt.axvline(max_current_time)
    plt.title(phase)
    ax=plt.gca()
    ax2=ax.twinx()
    ax2.plot(ss_current_time, ss_current, color="red")
    plt.show()"""
    magnitude=max(ss_potential)/max(ss_current)
    calculated_impede=magnitude*np.exp(1j*phase)
    impede[j, 0]=np.real(calculated_impede)
    impede[j, 1]=-np.imag(calculated_impede)
    impede[j, 2]=np.real(z_freq)
    impede[j, 3]=np.imag(z_freq)
ax[0].set_xlabel("$Z_r$")
ax[0].set_ylabel("$-Z_i$")
ax[1].set_xlabel("$Z_r$")
ax[1].set_ylabel("$-Z_i$")
sim.plot_data(impede[:, 2:], ax[1], label="FT method")
sim.plot_data(impede[:, :2], ax[1], label="Amp+phase method")
ax[1].legend()
ax[0].legend()
plt.show()
