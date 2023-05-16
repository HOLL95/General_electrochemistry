import numpy as np
import matplotlib.pyplot as plt
pi =np.pi
fs = 10000
ts = 1/fs
t = np.arange(0, 1-ts, step=ts)
# signal parameters
A = 5      # amplitude
f0 = 10.834095034955    # frequency
phi = -pi/8 # phase
signal = A*np.cos(2*pi*f0*t + phi)+0.5*np.cos(2*pi*200.234*t + phi)+np.random.rand(len(t))
# DTFT evaluated at f = f0
dtft_tone = np.exp(-2j*pi*f0*t)
dtft_f0 = (1/len(t))*sum(signal * dtft_tone)
# estimate parameters
A_est = 2*abs(dtft_f0)   # 4.0
phi_est = np.angle(dtft_f0) # -0.3927 == -pi/8
print(phi_est/phi, A_est/A)
plt.plot(signal)
plt.show()