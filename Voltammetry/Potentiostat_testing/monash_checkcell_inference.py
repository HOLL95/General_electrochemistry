import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
import os
import copy
from scipy.optimize import fmin
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from harmonics_plotter import harmonics
loc="/home/userfs/h/hll537/Documents/Experimental_data/Nat/Figure_2/"
#loc="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_2/"
files=[ "FTACV_MONASH_CHECK-CELL_v2_ideal_capacitor_200_mV_export_cv_"]#,"FTacV_ideal_capacitor_(no harmonics)_200_mV_cv_"]
desire="Timeseries"
labels=["Non-ideal","Ideal"]

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
    ps=pure_sine(time, ac_component)
    init_guess=[0.2, get_max, 4.91]
    minimum=fmin(ps.fsine, init_guess)
    print(minimum)
    simsine=ps.sine_calc(*minimum)
    fig, ax=plt.subplots(1,2)
    ax[0].plot(time, ac_component)
    ax[0].plot(time, simsine)
    ax[1].plot(time, simsine-ac_component)
    plt.show()