import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
import copy
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from harmonics_plotter import harmonics
loc="/home/userfs/h/hll537/Documents/Experimental_data/Nat/checkcell/"
loc="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_2/"
files=["Gamry_ideal_200mV_120Hz.txt"]
desire="Phase"
labels=["Ideal"]

for j in range(0, len(files)):
    file=files[j]    
    current=np.loadtxt(loc+file, skiprows=1)
    time=current[:,0]
    voltage=current[:,1]
    current=current[:,2]
    first_reduction=np.where(time<6.99)
    time=time[first_reduction]
    current=current[first_reduction]
    voltage=voltage[first_reduction]
    freqs=np.fft.fftfreq(len(current), time[1]-time[0])
    Y=np.fft.fft(current)     
    #plt.plot(Y)
    #plt.show()
    get_max=abs(freqs[np.where(Y==max(Y))][0])
    potential_Y=np.fft.fft(voltage)
    #plt.plot(freqs, potential_Y)
    #plt.plot(time, np.fft.ifft(potential_Y))
    m=copy.deepcopy(potential_Y)
    m=np.zeros(len(voltage), dtype="complex")
    m[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]=potential_Y[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]
    potential_Y[np.where((freqs<0.25*get_max) & (freqs>-0.25*get_max))]=0
    #plt.plot(freqs, potential_Y)
    ac_component=voltage#np.real(np.fft.ifft(potential_Y))

    if desire=="Phase":
        
        num_periods=int(np.floor(time[-1]*get_max))
        periods=list(range(1, num_periods))
        phases=np.zeros((2, num_periods-1))
        fig, ax=plt.subplots(2, 5)
        min_val=num_periods/6
        max_val=num_periods*(5/6)
        values=[int(x) for x in np.linspace(min_val, max_val, 10)]

        for i in range(0, len(values)):
            val=values[i]
            idx=np.where((time>(val/get_max))& (time<((val+1)/get_max)))
            s=np.sin(2*np.pi*get_max*time[idx])      # reference sine, note the n*t
            c=np.cos(2*np.pi*get_max*time[idx])  
            sines=[current[idx], ac_component[idx]]
            ax[i//5, i%5].plot(time[idx], sines[1], label="Potential")
            
            ax[i//5, i%5].plot(time[idx][0], sines[1][0], color="red", label="Reference")
            ax[0, -1].legend(loc="upper right")
            twinx=ax[i//5, i%5].twinx()
            twinx.plot(time[idx], s, label="Reference", color="red")
            #twinx.plot(time[idx], sines[1], label="Potential", color="Red")
            ax[i//5, i%5].set_title("Sinewave %d"% val)
            ax[i//5, i%5].set_xlabel("Time (s)")
            ax[i//5, i%5].set_ylabel("Potential (V)")
            twinx.set_yticks([])
            #twinx.set_ylabel("Potential (V)")

        
plt.show()
