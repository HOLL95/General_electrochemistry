import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import numpy as np
import matplotlib.pyplot as plt
from harmonics_plotter import harmonics
data_loc="Experimental_data/15_11_22"
print(os.listdir(data_loc))
files=os.listdir(data_loc)
FTV_names=[]
for file in files:
    if "FTV" in file:
        FTV_names.append(file)
frequencies=np.flip([10, 50, 100, 200])
desired_harms=list(range(0, 6))
divisor=1
for f in frequencies:
    str_f=str(f)
    for file in FTV_names:
        if str_f+"_hz" in file:
            data=np.loadtxt(data_loc+"/"+file, skiprows=1)
            time=data[:,0]
            potential=data[:,1]
            current=data[:,2]
            plt.plot(time, current)
            #y=np.fft.fft(current)
            #fft_freq=np.fft.fftfreq(len(current), time[1]-time[0])
            #plt.plot(fft_freq, np.log10(np.abs(y)))
           
                
            plt.show()
plt.legend()
#plt.show()

#
#print(time[1]-time[0]/len(current))
#plt.show()