import numpy as np
import matplotlib.pyplot as plt
import math
freq=5
range=100e-3
step=10e-3
num_oscillations=10
start_potential=-50e-3
num_steps=range/step
num_cycles=(num_steps*num_oscillations)
end_time=num_cycles/freq
interval=0.001
len_t=end_time/interval
p_per_step=len_t/num_steps
stepped=start_potential+(step*np.floor_divide(np.arange(0, len_t, 1), p_per_step))
amp=0.3
t=np.arange(0, end_time, interval)

sinusoid=amp*np.sin(freq*2*math.pi*t)
stepped_potential=np.add(stepped, sinusoid)
plt.plot(t, stepped_potential)
plt.xlabel("Time (s)")
plt.ylabel("Potential (V)")
plt.show()