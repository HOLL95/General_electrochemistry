import os
import sys
import numpy as np
import matplotlib.pyplot as plt
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from harmonics_plotter import harmonics
file_loc="Experimental_data"
files=os.listdir(file_loc)
harm_range=list(range(1, 7))
for file in files:
    data=np.loadtxt(file_loc+"/"+file, skiprows=1)
    time=data[:,0]
    potential=data[:,2]
    current=data[:,1]
    fft=abs(np.fft.fft(current))
    fig, ax=plt.subplots()
    plt.plot(time, current)
    plt.show()
    fft_freq=np.fft.fftfreq(len(current), time[-1]-time[-2])
    max_freq=max(fft_freq[np.where(fft==max(fft))])
    plt.semilogy(fft_freq, fft)
    plt.title(file)
    for i in range(0, len(harm_range)):
        plt.axvline(max_freq*(i+1), color="black", linestyle="--")
    plt.show()
    h_class=harmonics(harm_range, max_freq, 0.1)
    h_class.plot_harmonics(time, experimental_time_series=current, hanning=True, plot_func=abs)
    plt.show()
    #plt.semilogy(fft_freq, fft)
    #plt.axvline(max_freq, color="red")
    
