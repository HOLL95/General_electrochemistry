import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
import copy
from scipy.optimize import fmin
import cma
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
desire="Timeseries"
labels=["Ideal"]
class pure_sine:
    def __init__(self, time, sine):
        self.time=time
        self.sine=sine
    def sine_calc(self, amp, freq, phase, m, c):
        
        freq*=2*np.pi
        phase=phase#m*(self.time+c)
        #fig, ax=plt.subplots()
        #ax.plot(phase)
        #plt.show()
        return_arg=abs(amp)*np.sin(freq*self.time+phase)
        
        #plt.plot(return_arg)
        #plt.plot(self.sine)
        #plt.show()
        #plt.plot(time, phase)
        #plt.show()
        return return_arg
    def fsine(self, params):
        amp, freq, phase, m, c=params
        amp=0.1
        #print(list(params))
        simsine=self.sine_calc(amp, freq, phase, m, c)
        
        return RMSE(simsine, self.sine)
def RMSE(y, y_data):
    return np.sqrt(np.sum(np.square(np.subtract(y, y_data))))
def constraints(x):
    return [x[1] + 0.1, ]  # constrain the second variable to <= -1, the second constraint is superfluous
 # unconstrained function with adaptive Lagrange multipliers


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
    get_max=abs(freqs[np.where(Y==max(Y))][0])
    print(get_max)
    potential_Y=np.fft.fft(voltage)
    #plt.plot(freqs, potential_Y)
    #plt.plot(time, np.fft.ifft(potential_Y))
    m=copy.deepcopy(potential_Y)
    m=np.zeros(len(voltage), dtype="complex")
    m[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]=potential_Y[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]
    potential_Y[np.where((freqs<0.25*get_max) & (freqs>-0.25*get_max))]=0
    #plt.plot(freqs, potential_Y)
    #plt.show()

    ac_component=np.real(np.fft.ifft(potential_Y))
    second_reduction =np.where((time>0.5) & (time<6))
    time=time[second_reduction]
    ac_component=ac_component[second_reduction]
    current=current[second_reduction]
    ps=pure_sine(time, ac_component)
  
    #plt.plot(ac_component)
    #plt.plot(0.1*np.sin(2*np.pi*100*time), linestyle="--")
    #for m in [-50, -50, -80]:
    #    var_sine=ps.sine_calc(0.1, get_max, 0, m, -20)
    #    plt.plot(var_sine, label=m)
    #plt.legend()
    #plt.show()
    cfun = cma.ConstrainedFitnessAL(ps.fsine, constraints) 
    init_guess=[0.1, get_max, 5, -50, -20]
    xopt, es = cma.fmin2(ps.fsine, init_guess, 1, callback=cfun.update)
    result=dict(es.result._asdict())

    top_vals=result["xbest"]
    #top_vals=[-3.52551985856492, 127.52956765650454, 3.125341835416577, -47.30391320263106, -28.005164843932178]
    #minimum=fmin(ps.fsine, init_guess)
    #print(minimum)
    top_vals[0]=0.1
    simsine=ps.sine_calc(*top_vals)
    fig, ax=plt.subplots(1,2)
    ax[0].plot(time, ac_component)
    ax[0].plot(time, simsine)
    ax[1].plot(time, simsine-ac_component)
    plt.show()
   