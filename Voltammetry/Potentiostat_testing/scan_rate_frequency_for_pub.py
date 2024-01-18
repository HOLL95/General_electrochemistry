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
frequencies=[1, 10, 50, 100, 200, 400, 4000]
fig, ax=plt.subplots()
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

    label="%d Hz" % desired_Hz
    if min_interval<actual_minimum:
        ax.loglog(scan_rate_range*1000, points_list, label=label, linestyle="--")
    else:
        ax.loglog(scan_rate_range*1000, points_list, label=label)
#ax.axhline(250000, linestyle="--", color="black", label="1e6 points")
ax.axhline(1e6, linestyle="--", color="black", label="1e6 points")
ax.set_xlabel("Scan rate (mV)")
ax.set_ylabel("Number of points required")
ax.set_title("*Accessing %d th harmonic, %d mV potential window" % (desired_harmonic-0.5, distance*1000)) 
ax.legend()




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