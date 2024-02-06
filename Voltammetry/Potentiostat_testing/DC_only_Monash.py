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
files=[ "FTACV_MONASH_CHECK-CELL_v2_ideal_capacitor_200_mV_export_cv_"]#,"FTacV_ideal_capacitor_(no harmonics)_200_mV_cv_"]
desire="Phase"
labels=["Non-ideal","Ideal"]

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
    #plt.plot(freqs, potential_Y)
    dc_idx=np.where((freqs>0.25*get_max) | (freqs<-0.25*get_max))
    potential_Y[dc_idx]=0
    Y[dc_idx]=0

    #plt.plot(freqs, potential_Y)
    #ac_component=np.real(np.fft.ifft(potential_Y))
    #plt.plot(freqs, potential_Y)
    ifft_Y=np.fft.ifft(Y)
    ifft_v_Y=np.fft.ifft(potential_Y)
    plt.plot(ifft_v_Y, ifft_Y)
plt.show()
  