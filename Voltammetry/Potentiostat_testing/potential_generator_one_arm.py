import numpy as np
import matplotlib.pyplot as plt
import math

min_interval=0.34*1e-3
desired_Hz=72
desired_scan_rate=30e-3
E_reverse=0.5
E_start=-0.2
distance=(E_reverse-E_start)
DC_range=distance
end_time=distance/desired_scan_rate
num_points=int(end_time//min_interval)+1
if num_points>60000:
    splitting=True
else:
    splitting=False
fft_freq=np.fft.fftfreq(num_points, min_interval)
print(max(fft_freq), (0.5/min_interval), )
total_sine_waves=end_time*desired_Hz
Ac_amplitude=0.15
t=np.linspace(0, 1, int(num_points))
#total_sine_waves=10
y=E_start+(t*DC_range)+Ac_amplitude*np.sin(total_sine_waves*2*math.pi*t)
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
plt.show()  
