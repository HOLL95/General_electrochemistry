import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.fftpack as syfp
x=np.linspace(0, 5, 1000)
hanning=np.hanning(len(x))
y=np.sin(2*math.pi*x)
plt.plot(x, y)
plt.show()
fft=np.fft.fft(x)
fft_freq=np.fft.fftfreq(len(x), x[1]-x[0])
freqs = syfp.fftfreq(len(x), x[1]-x[0])
plt.semilogy(freqs, abs(fft))
plt.axvline(1, color="red", linestyle="--")
plt.show()

import scipy as sy
import scipy.fftpack as syfp


dt = 0.02071
t = np.linspace(0, 10, 1000)            ## time at the same spacing of your data
freq=10
u = np.sin(2*np.pi*t*freq)            ## Note freq=0.01 Hz

# Do FFT analysis of array
FFT = sy.fft.fft(u)

# Getting the related frequencies
freqs = syfp.fftfreq(len(u), t[1]-t[0])     ## added dt, so x-axis is in meaningful units

# Create subplot windows and show plot
plt.subplot(211)
plt.plot(t, u)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.subplot(212)
plt.plot(freqs, sy.log10(abs(FFT)), '.')  ## it's important to have the abs here
plt.xlim(-2*freq, 2*freq)                       ## zoom into see what's going on at the peak
plt.show()
