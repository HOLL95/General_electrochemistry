import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
import copy
from pandas import DataFrame
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

files=np.flip(["30_Hz_200mV_amp_non_ideal_cv_", "120_Hz_200mV_amp_non_ideal_cv_"])
desire="Fourier"
labels=np.flip(["30 Hz", "120 Hz"])
desired_freqs=np.arange(30, 360, 30)
positions=np.zeros(len(desired_freqs))
fig, ax=plt.subplots(1,2)
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
    ax4=ax[1]
    t_idx=np.where((time>1) & (time<1.1))
    ax[0].plot(time[t_idx], ac_component[t_idx])
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Magnitude")
    if desire=="Fourier":
        absY= np.fft.fftshift(abs(potential_Y))
        plotfreq=np.fft.fftshift(freqs)
        ax[1].plot(plotfreq,absY, label=labels[j], alpha=1-(0.1*j))
        
        """
            for i in range(1, ):
                get_freq=i*30
                max_idx=np.where((plotfreq<1.25*get_freq) & (plotfreq>0.75*get_freq))
                max_val=max(absY[max_idx])
                max_plot_idx=plotfreq[np.where(absY==max_val)]
                ax.text(max_plot_idx[0], max_val, "%d Hz"%(i*30))"""
        
        ax4.set_xlabel("Frequency(Hz)")
        ax4.set_ylabel("Amplitude")
        ax4.legend()


    ax4.set_xlim([0, 360])
    #for j in range(0, len(desired_freqs)):
    #    f=desired_freqs[j]
    #    freq_idx=np.where((plotfreq>(f-5) & (plotfreq<(f+5))))
    #    Y_val=max(absY[freq_idx])
    #    fig, ax=plt.subplots()
    #    ax.plot(plotfreq[freq_idx], absY[freq_idx])
    #    positions[j]=max(Y_val, positions[j])
plt.show()