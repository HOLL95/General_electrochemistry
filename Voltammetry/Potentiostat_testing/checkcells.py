import numpy as np
import matplotlib.pyplot as plt
import copy
loc="/home/userfs/h/hll537/Documents/Experimental_data/Nat/checkcell/"
current=np.loadtxt(loc+"FTacV_ideal_capacitor_(no harmonics)_200_mV_cv_current")[:,1]
voltage=np.loadtxt(loc+"FTacV_ideal_capacitor_(no harmonics)_200_mV_cv_voltage")
#current=np.loadtxt(loc+"FTacV_non-ideal_capacitor_(harmonics)_200_mV_cv_current")[:,1]
#voltage=np.loadtxt(loc+"FTacV_non-ideal_capacitor_(harmonics)_200_mV_cv_voltage")
time=voltage[:,0]
voltage=voltage[:,1]
freqs=np.fft.fftfreq(len(current), time[1]-time[0])
Y=np.fft.fft(current)
get_max=abs(freqs[np.where(Y==max(Y))][0])
plt.semilogy    (freqs, abs(Y))
plt.show()
potential_Y=np.fft.fft(voltage)



#plt.plot(freqs, potential_Y)
#plt.plot(time, np.fft.ifft(potential_Y))
m=copy.deepcopy(potential_Y)
m=np.zeros(len(voltage), dtype="complex")
m[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]=potential_Y[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]
potential_Y[np.where((freqs<0.25*get_max) & (freqs>-0.25*get_max))]=0
#plt.plot(freqs, potential_Y)
ac_component=np.real(np.fft.ifft(potential_Y))

plt.plot(time, ac_component)
ax=plt.gca()
twinx=ax.twinx()
twinx.plot(time, current, color="red")
ax.set_xlim([-0.01, 0.05])

plt.show()

                # check the 3rd harmonic, it should have
                           # 1/3 amplitude, 3*10 deg

idx=np.where(time>(3/get_max))
s=np.sin(2*np.pi*get_max*time[idx])      # reference sine, note the n*t
c=np.cos(2*np.pi*get_max*time[idx])  
for sinusoid in [current[idx], ac_component[idx]]:
    xs,xc=sinusoid*s,sinusoid*c
    a,b=2*np.mean(xs),2*np.mean(xc)
    print(a,b)
    mag=np.hypot(b,a)
    rad=np.arctan2(b,a)
    deg=rad*180/np.pi
    print(deg)
