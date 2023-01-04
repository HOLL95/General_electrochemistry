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
experiment_funcs={"FTV":{"hanning":True, "func":abs, "xlabel":"Time (s)"}, 
                    "PSV":{"hanning":False, "func":np.real, "xlabel":"Potential (V)"}}
for experiment in ["PSV","FTV"]:
    FTV_names=[]
    for file in files:
        if experiment in file:
            FTV_names.append(file)
    frequencies=np.flip([10, 50, 100, 200])
    desired_harms=list(range(0, 6))
    fig, ax=plt.subplots(len(desired_harms), len(frequencies))
    ts_fig, ts_ax=plt.subplots(1,1)
    freq_fig, freq_ax=plt.subplots(1,len(frequencies))
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    divisor=2
    for j in range(0, len(frequencies)):
        f=frequencies[j]
        str_f=str(f)
        plot_label="{0} Hz".format(str_f)
        for file in FTV_names:
            if str_f+"_hz" in file:
                data=np.loadtxt(data_loc+"/"+file, skiprows=1)
                time=data[:,0]
                potential=data[:,1]
                current=data[:,2]
                start_time=time[-1]/3
                end_time=start_time+(2/f)
                
                t_idx=np.where((time>start_time) & (time<end_time))
                freq_ax[j].plot(time[t_idx], current[t_idx], color=colors[j])
                freq_ax[j].scatter(time[t_idx], current[t_idx],color=colors[j])
                freq_ax[j].set_xlabel("Time (s)")
                freq_ax[j].set_ylabel("Current (mA)")
                freq_ax[j].set_title(plot_label)
                if experiment=="PSV":
                    xaxis=potential
                elif experiment=="FTV":
                    xaxis=time
                ts_ax.plot(xaxis, current, label=plot_label)
                ts_ax.set_xlabel(experiment_funcs[experiment]["xlabel"])
                ts_ax.set_ylabel("Current (mA)")
                
                #y=np.fft.fft(current)
                #fft_freq=np.fft.fftfreq(len(current), time[1]-time[0])
                #plt.plot(fft_freq, np.log10(np.abs(y)))
                harms=harmonics(desired_harms, f, 0.1)
                plot_harmonics=harms.generate_harmonics(time, np.multiply(current, 1000), hanning=experiment_funcs[experiment]["hanning"])
                for i in range(0, len(plot_harmonics)):
                    ax[i,j].plot(xaxis[:len(time)//divisor], 
                                experiment_funcs[experiment]["func"](plot_harmonics[i, :len(time)//divisor]), 
                                color=colors[j])
                    if i==  len(plot_harmonics)-1:
                        ax[i,j].set_xlabel(experiment_funcs[experiment]["xlabel"])
                    if i==0:
                        ax[i,j].set_title(plot_label)
    for j in range(0, len(frequencies)):
        ax[len(plot_harmonics)//2, j].set_ylabel("Current ($\\mu$A)")
    ts_ax.legend()
    fig.subplots_adjust(top=0.959,
    bottom=0.081,
    left=0.051,
    right=0.988,
    hspace=0.1,
    wspace=0.349)
    plt.legend()
    plt.show()
#plt.show()

#
#print(time[1]-time[0]/len(current))
#plt.show()