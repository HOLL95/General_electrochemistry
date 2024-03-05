import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
import copy#
from scipy.optimize import fmin
from pandas import DataFrame
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from harmonics_plotter import harmonics
loc="/home/henryll/Documents/Experimental_data/Henry/checkcells_2/"
files=["FTACV_Ideal_check_cell_V4_cv_","Post_meeting_14_2_24_2_new_params.txt","Ideal_cell_ivium_7.txt" ]#,"FTacV_ideal_capacitor_(no harmonics)_200_mV_cv_"]
desire="Timeseries"
labels=["(A) Monash","(B) Gamry", "(C) Ivium"]
fig, ax=plt.subplots(3,2)
class pure_sine:
    def __init__(self, time, sine):
        self.time=time
        self.sine=sine
    def sine_calc(self, amp, freq, phase):
        
        freq*=2*np.pi
        return amp*np.sin(freq*self.time+phase)
    def fsine(self, params):
        amp, freq, phase=params
        simsine=self.sine_calc(amp, freq, phase)
        return RMSE(simsine, self.sine)
def RMSE(y, y_data):
    return np.sqrt(np.sum(np.square(np.subtract(y, y_data))))

for j in range(0, len(files)):
    file=files[j]    
    if j==0:
        current=np.loadtxt(loc+file+"current")[:,1]
        voltage=np.loadtxt(loc+file+"voltage")
        time=voltage[:,0]
        voltage=voltage[:,1]
    elif j==1:
        current=np.loadtxt(loc+file, skiprows=1)
        time=current[:,0]
        voltage=current[:,1]
        current=current[:,2]
        
    elif j==2:
        current=np.loadtxt(loc+file, skiprows=1)
        time=current[:,0]
        voltage=current[:,2]
        current=current[:,1]
    
    freqs=np.fft.fftfreq(len(current), time[1]-time[0])
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

  
        
    num_periods=int(np.floor(time[-1]*get_max))
    periods=list(range(1, num_periods))
    phases=np.zeros((2, num_periods-1))
    for i in range(0, num_periods-1):
        print(i)
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
    period_dict={"Periods":periods, "Potential phases":phases[1,:]}
   
    DataFrame(data=period_dict).to_csv(labels[j]+"_phase_fig5.csv")
   

    ax[j,0].plot(periods[100:-100], phases[1,100:-100])
    ax[j,0].set_xlabel("Period")
    ax[j,0].set_ylabel("Phase ($^\\circ$)")
    ax[j,0].set_title(labels[j])
    fit_time=np.where((time>1) & (time<1.1))
    
    ps=pure_sine(time[fit_time], ac_component[fit_time])
    
    init_guess=[0.2, get_max, 4.91]
    if j==1:
        init_guess[0]=0.1
    minimum=fmin(ps.fsine, init_guess)
    print(minimum)
    if j==1:
        minimum[0]=0.1
    simsine=ps.sine_calc(*minimum)
    if j==1:
        fac=1
    else:
        fac=1
    ax[j,1].plot(time[fit_time], fac*ac_component[fit_time], label="E$_{app}$")
    ax[j,1].plot(time[fit_time], fac*simsine, linestyle="--", label="sin")
    ax[j,1].plot(time[fit_time], fac*(simsine-ac_component[fit_time]), label="Residual")
    ax[j,1].set_xlabel("Time(s)")
    ax[j,1].set_ylabel("Potential (V)")
    time_series_dict={"Times (s)":time[fit_time], "Potential":fac*ac_component[fit_time], "Fitted sin":fac*simsine, "Residual": fac*(simsine-ac_component[fit_time])}
    DataFrame(data=time_series_dict).to_csv(labels[j]+"_ts_fig5.csv")
    #ax[j].set_xlim([0, 800])
ax[0,1].legend()
plt.show()
