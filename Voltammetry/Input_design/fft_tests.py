import numpy as np
x=np.linspace(0, 2*np.pi, 1000)
y=np.sin(x)
ft=np.fft.fft(y)
print(len(ft))