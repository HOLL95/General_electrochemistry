import numpy as np
import matplotlib.pyplot as plt
import math
freq=5
pot_range=100e-3
step=1e-3
num_oscillations=10
start_potential=-50e-3
num_steps=pot_range/step
num_cycles=(num_steps*num_oscillations)
end_time=num_cycles/freq
interval=0.01
len_t=end_time/interval
p_per_step=len_t/num_steps
stepped=start_potential+(step*np.floor_divide(np.arange(0, len_t, 1), p_per_step))
amp=0.3
t=np.arange(0, end_time, interval)

sinusoid=amp*np.sin(freq*2*math.pi*t)
stepped_potential=np.add(stepped, sinusoid)
f=open("stepped_potential.txt", "w")
if len_t>62000:
    raise ValueError("Too many points for Ivium")

for i in range(0, len(t)):
    f.write("{0} \n".format(stepped_potential[i]))
f.close()


print("Set MultiCycle points to {0}".format(int(len_t)))
plt.plot(t, stepped_potential)
plt.xlabel("Time (s)")
plt.ylabel("Potential (V)")
plt.show()