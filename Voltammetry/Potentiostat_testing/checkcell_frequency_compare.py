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
files=["30_Hz_200mV_amp_non_ideal_cv_", "60_Hz_200mV_amp_non_ideal_cv_", "120_Hz_200mV_amp_non_ideal_cv_"]
desire="Fourier"
labels=["30 Hz", "60 Hz", "120 Hz"]

for j in range(0, len(files)):
    file=files[j]    
    current=np.loadtxt(loc+file+"current")[:,1]
    voltage=np.loadtxt(loc+file+"voltage")
    time=voltage[:,0]
    voltage=voltage[:,1]
    freqs=np.fft.fftfreq(len(current), time[1]-time[0])
    Y=np.fft.fft(current)
    get_max=abs(freqs[np.where(Y==max(Y))][0])
    potential_Y=np.fft.fft(voltage)
    #plt.plot(freqs, potential_Y)
    #plt.plot(time, np.fft.ifft(potential_Y))
    m=copy.deepcopy(potential_Y)
    m=np.zeros(len(voltage), dtype="complex")
    m[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]=potential_Y[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]
    potential_Y[np.where((freqs<0.25*get_max) & (freqs>-0.25*get_max))]=0
    #plt.plot(freqs, potential_Y)
    ac_component=np.real(np.fft.ifft(potential_Y))

    if desire=="Fourier":
        absY= np.fft.fftshift(abs(Y))
        plotfreq=np.fft.fftshift(freqs)
        plt.semilogy(plotfreq,absY, label=labels[j], alpha=1-(0.1*j))
        ax=plt.gca()
        """
            for i in range(1, ):
                get_freq=i*30
                max_idx=np.where((plotfreq<1.25*get_freq) & (plotfreq>0.75*get_freq))
                max_val=max(absY[max_idx])
                max_plot_idx=plotfreq[np.where(absY==max_val)]
                ax.text(max_plot_idx[0], max_val, "%d Hz"%(i*30))"""
        ax.set_xlim([0, 360])
        ax.set_xlabel("Frequency(Hz)")
        ax.set_ylabel("Amplitude (A)")
        ax.legend()
        
plt.show()
