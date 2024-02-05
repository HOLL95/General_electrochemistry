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
files=["Gamry_ideal_200mV_120Hz.txt"]
desire="Timeseries"
labels=["Ideal"]

for j in range(0, len(files)):
    file=files[j]    
    current=np.loadtxt(loc+file, skiprows=1)
    time=current[:,0]
    voltage=current[:,1]
    current=current[:,2]
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
        plt.plot(freqs, abs(Y), label=labels[j])
        ax=plt.gca()
        ax.set_xlim([-2000, 2000])
        ax.set_xlabel("Frequency(Hz)")
        ax.set_ylabel("Amplitude (A)")
        ax.legend()
    elif desire=="Timeseries":   
        
        fig, ax=plt.subplots()
        if j==0:


            twinx=ax.twinx()
            plot_args={"alpha":1}
        else:
            plot_args={"linewidth":3, "alpha":0.65, "linestyle":"--"}
        ax.plot(time, ac_component, **plot_args)#
        ax.set_ylabel("Potential (V)")
        ax.set_xlabel("Time (s)")
        twinx.plot(time, current, color="red", label=labels[j], **plot_args)
        twinx.set_ylabel("Current (A)")
        #twinx.set_ylim([-0.002, 0.002])
        twinx.legend()
        ax.set_xlim([-0.01, 0.1])

        

                    # check the 3rd harmonic, it should have
                            # 1/3 amplitude, 3*10 deg
    elif desire=="Harmonics":
        h_class=harmonics(list(range(0, 10)), get_max, 0.5)
        if j==0:
            fig, ax=plt.subplots(h_class.num_harmonics, 1)
        plot_dict={"%s_time_series"%(labels[j]):current, "hanning":True, "legend":{"bbox_to_anchor":[0.5, 1.2]}}
        h_class.plot_harmonics(time, **plot_dict)

    elif desire=="Phase":
        if j==0:
            fig, ax=plt.subplots(1,2)
        num_periods=int(np.floor(time[-1]*get_max))
        periods=list(range(1, num_periods))
        phases=np.zeros((2, num_periods-1))
        for i in range(0, num_periods-1):
            print(i)
            idx=np.where((time>(i/get_max))& (time<((i+1)/get_max)))
            s=np.sin(2*np.pi*get_max*time[idx])      # reference sine, note the n*t
            c=np.cos(2*np.pi*get_max*time[idx])  
            sines=[current[idx], ac_component[idx]]
            for m in range(0, len(sines)):
                sinusoid=sines[m]
                
                xs,xc=sinusoid*s,sinusoid*c
                a,b=2*np.mean(xs),2*np.mean(xc)
                mag=np.hypot(b,a)
                rad=np.arctan2(b,a)
                deg=rad*180/np.pi
                phases[m][i]=deg
        
        ax[0].set_title("Current phase")
        ax[0].set_xlabel("Period")
        print(j, "+"*30)
        ax[0].scatter(periods, phases[0,:], label=labels[j])
        #if j==1:
        #    twinx=ax[0].twinx()
        #    twinx.scatter(periods, phases[1,:]+90,  color="red", s=0.5)
        
        #twinx=ax[0].twinx()
       
        ax[1].set_title("Potential phase")
        ax[1].set_xlabel("Period")
        ax[1].scatter(periods, phases[1,:])#
        ax[0].legend()
        ax[1].legend()
plt.show()