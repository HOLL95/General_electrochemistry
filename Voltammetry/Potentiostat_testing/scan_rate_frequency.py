import numpy as np
import matplotlib.pyplot as plt
import math



scan_rate_range=np.linspace(1e-3, 350e-3, 1000)
desired_harmonic=30
desired_harmonic+=0.5
actual_minimum=4e-6

E_reverse=0.5
E_start=-0.5
distance=(E_reverse-E_start)
DC_range=distance/0.5

nyquist=(0.5/actual_minimum)/desired_harmonic
frequencies=[1, 10, 50, 100, 200, 400, nyquist]
fig, ax=plt.subplots(1,2)
for j in range(0, len(frequencies)):
    points_list=np.zeros(len(scan_rate_range))
    desired_Hz=frequencies[j]
    min_interval=0.5/(desired_Hz*desired_harmonic)
    print(min_interval)
    for i in range(0, len(scan_rate_range)):
        
        
        
        #min_interval=1/5000
        #desired_Hz=10
        desired_scan_rate=scan_rate_range[i]

        end_time=(2*distance)/desired_scan_rate
        num_points=int(end_time//min_interval)+1
        points_list[i]=num_points
    if j==len(frequencies)-1:
        label="%.2f Hz (Gamry max)" % desired_Hz
    else:
        label="%d Hz" % desired_Hz
    if min_interval<actual_minimum:
        ax[0].loglog(scan_rate_range*1000, points_list, label=label, linestyle="--")
    else:
        ax[0].loglog(scan_rate_range*1000, points_list, label=label)
ax[0].axhline(250000, linestyle="--", color="black", label="1e6 points")
ax[0].axhline(1e6, linestyle="--", color="black", label="2.5e5 points")
ax[0].set_xlabel("Scan rate (mV)")
ax[0].set_ylabel("Number of points required")
ax[0].set_title("*Accessing %d th harmonic, %d mV potential window" % (desired_harmonic-0.5, distance*1000)) 
ax[0].legend()


MAH_frequencies=np.linspace(1, 1000, 1000)
MAH_values=np.zeros(len(MAH_frequencies))
invert_min=0.5/actual_minimum
for i in range(0, len(MAH_frequencies)):
    desired_Hz=MAH_frequencies[i]
    MAH=invert_min/desired_Hz
    if (MAH-0.5)>np.floor(MAH):
        MAH_values[i]=np.floor(MAH)
    else:
        MAH_values[i]=np.floor(MAH)-1
ax[1].loglog(MAH_frequencies, MAH_values)
ax[1].set_xlabel("Input frequency (Hz)")
ax[1].set_ylabel("Maximum accessible harmonic at %d samples $s^{-1}$" % (1/actual_minimum))

plt.show()


"""if num_points>65000:
    splitting=True
else:
    splitting=False


total_sine_waves=end_time*desired_Hz
Ac_amplitude=0.15
t=np.linspace(0, 1, int(num_points))
#total_sine_waves=10
y=E_reverse-(np.abs(t-0.5)*DC_range)+Ac_amplitude*np.sin(total_sine_waves*2*math.pi*t)
f=open("-0.2 to 0.5_100mVs-1_5Hz.txt", "w")
for i in range(0, len(t)):
    f.write("{0} \n".format(y[i]))
f.close()
print(end_time)
print("Set MultiCycle points to {0}".format(num_points))
max_accessible_freq=abs(fft_freq[num_points//2])
print("Maximum accessible harmonic frequency is {0}".format(max_accessible_freq))
print("Maximum accessible harmonic number is {0}".format(int(max_accessible_freq//desired_Hz)))
plt.plot(np.arange(0, end_time, min_interval), y)
plt.show()  """