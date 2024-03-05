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
files=[ "FTACV_Ideal_check_cell_export_cv_"]#,"FTacV_ideal_capacitor_(no harmonics)_200_mV_cv_"]
desire="Timeseries"
labels=["Non-ideal","Ideal"]
plt.figure(figsize=(12, 6))
ax1 = plt.subplot(2,3,1)
ax2 = plt.subplot(2,3,2)
ax3 = plt.subplot(2,3,3)
ax4 = plt.subplot(2,1,2)
axes = [ax1, ax2, ax3, ax4]
ax1.set_title("(A) Frequency domain")
ax2.set_title("(B) Time domain")
ax2.set_xlabel("Time (s)")
for j in range(0, len(files)):
    file="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_2/FTACV_Ideal_check_cell_V4_cv_" 
    current=np.loadtxt(file+"current")[:,1]
    voltage=np.loadtxt(file+"voltage")
    time=voltage[:,0]
    voltage=voltage[:,1]
    freqs=np.fft.fftfreq(len(current), time[1]-time[0])
    Y=np.fft.fft(current)
    freq_idx=np.where((freqs>0) & (freqs<480))
   
    ax1.set_xlabel("Frequency(Hz)")
    twinx1=ax1#.twinx()
    twinx1.set_ylabel("Amplitude")
    
    ax1.set_ylabel("Amplitude")
    get_max=abs(freqs[np.where(Y==max(Y))][0])
    potential_Y=np.fft.fft(voltage)
    ax1.semilogy(freqs[freq_idx], abs(potential_Y[freq_idx]))
    twinx1.semilogy(freqs[freq_idx], abs(Y[freq_idx]), color="red")
    a_dict={"Frequency (Hz)":freqs[freq_idx], "PotentialFFT":abs(potential_Y[freq_idx]), "CurrentFFT": abs(Y[freq_idx])}
    DataFrame(data=a_dict).to_csv("Fig2A.csv")
    #plt.plot(freqs, potential_Y)
    #plt.plot(time, np.fft.ifft(potential_Y))
    m=copy.deepcopy(potential_Y)
    m=np.zeros(len(voltage), dtype="complex")
    m[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]=potential_Y[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]
    potential_Y[np.where((freqs<0.25*get_max) & (freqs>-0.25*get_max))]=0
    #plt.plot(freqs, potential_Y)
    ac_component=np.real(np.fft.ifft(potential_Y))
    
    
    twinx=ax2.twinx()
    twinx.plot(time, ac_component)
    twinx.set_ylabel("Potential (V)")
    ax2.set_ylabel("Current (mA)")
    ax2.plot(time, current*1e3, color="red", linestyle="--")
    ax2.set_xlim([1, 1.05])
    save_idx=np.where((time>1)& (time<1.05))
    b_dict={"Time (s)":time[save_idx], "Current (mA)":current[save_idx]*1e3, "AC Potential (mV)":ac_component[save_idx]}
    DataFrame(data=b_dict).to_csv("Fig2B.csv")
            

                    # check the 3rd harmonic, it should have
  
 
        
    num_periods=int(np.floor(time[-1]*get_max))
    periods=list(range(1, num_periods))
    phases=np.zeros((2, num_periods-1))
    for i in range(0, num_periods-1):

        idx=np.where((time>(i/get_max))& (time<((i+1)/get_max)))
        s=np.sin(2*np.pi*get_max*time[idx])      # reference sine, note the n*t
        c=np.cos(2*np.pi*get_max*time[idx])  
        sines=[current[idx], ac_component[idx]]
        #plt.plot(time[idx], sines[1])
        for m in range(0, len(sines)):
            sinusoid=sines[m]
            
            xs,xc=sinusoid*s,sinusoid*c
            a,b=2*np.mean(xs),2*np.mean(xc)
            mag=np.hypot(b,a)
            rad=np.arctan2(b,a)
            deg=rad*180/np.pi
            phases[m][i]=deg
   
    ax3.set_title("(C) Phase")
    ax3.set_xlabel("Period")
    ax3.set_ylabel("Current phase")
    ax3.scatter(periods[1:], phases[0,1:], label=labels[j])
    c_dict={"Periods":periods, "Phases (deg)":phases[0,:]}
    DataFrame(data=c_dict).to_csv("2C.csv")
    #if j==1:
    #    twinx=ax[0].twinx()
    #    twinx.scatter(periods, phases[1,:]+90,  color="red", s=0.5)
    
    #twinx=ax[0].twinx()
    
    #ax[1].set_title("Potential phase")
    #ax[1].set_xlabel("Period")
    #ax[1].scatter(periods, phases[1,:])#
    #ax[0].legend()
    #ax[1].legend()
files=["30_Hz_200mV_amp_non_ideal_cv_", "60_Hz_200mV_amp_non_ideal_cv_", "120_Hz_200mV_amp_non_ideal_cv_"]
desire="Fourier"
labels=["30 Hz", "60 Hz", "120 Hz"]
desired_freqs=np.arange(30, 360, 30)
positions=np.zeros(len(desired_freqs))
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
        ax4.semilogy(plotfreq,absY, label=labels[j], alpha=1-(0.1*j))
        
        """
            for i in range(1, ):
                get_freq=i*30
                max_idx=np.where((plotfreq<1.25*get_freq) & (plotfreq>0.75*get_freq))
                max_val=max(absY[max_idx])
                max_plot_idx=plotfreq[np.where(absY==max_val)]
                ax.text(max_plot_idx[0], max_val, "%d Hz"%(i*30))"""
        
        ax4.set_xlabel("Frequency(Hz)")
        ax4.set_ylabel("Amplitude (A)")
        ax4.legend()
    save_idx=np.where((plotfreq>0) & (plotfreq<480))
    if j==0:
        plot_dict={"Frequency (Hz)":plotfreq[save_idx], "CurrentFFT %d Hz"%desired_freqs[j]:absY[save_idx]}
    else:
        plot_dict["CurrentFFT %d Hz"%desired_freqs[j]]=absY[save_idx]
    ax4.set_xlim([0, 360])
    #for j in range(0, len(desired_freqs)):
    #    f=desired_freqs[j]
    #    freq_idx=np.where((plotfreq>(f-5) & (plotfreq<(f+5))))
    #    Y_val=max(absY[freq_idx])
    #    fig, ax=plt.subplots()
    #    ax.plot(plotfreq[freq_idx], absY[freq_idx])
    #    positions[j]=max(Y_val, positions[j])
    #plt.show()
for i in range(0, len(desired_freqs)):
    ax4.text(desired_freqs[i], positions[j], "%d Hz" % desired_freqs[i])
plt.show()
df=DataFrame(data=plot_dict)
df.to_csv("Fig2D.csv")