import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
print(dir_list)
src_index=[x for x in range(0, len(dir_list)) if dir_list[x]=="General_electrochemistry"][0]
sys.path.append(("/").join(dir_list[:src_index+1]+["src"]))
from EIS_class import EIS
def potential(amp,frequency, time, phase):
    return amp*np.sin(2*np.pi*frequency*time+phase)
def current(cdl, amp, frequency, time, phase):
    return (cdl)*frequency*amp*np.cos(2*np.pi*frequency*time+phase)
def impedance_response(min_f, max_f, points_per_decade, num_osc, cdl, sf=200, amplitude=5e-3):
    if (np.log2(sf)%2)!=0:
        sf=2**np.ceil(np.log2(sf))
    
    num_points=int(num_osc*sf)
    frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
    freqs=[10**x for x in frequency_powers]
    Z=np.zeros((len(freqs), num_points), dtype="complex")
    threshold=0.5
    phase=0
    impedances=np.zeros(len(freqs), dtype="complex")
    magnitudes=np.zeros(len(freqs))
    phases=np.zeros(len(freqs))
    print(cdl, amplitude, phase)
    for i in range(0, len(frequency_powers)):
        time_end=num_osc/freqs[i]
        times=np.linspace(0, time_end, num_points, endpoint=False)
        V=potential(amplitude, freqs[i], times, phase)
        I=current(cdl, amplitude, freqs[i], times,phase)
        #I+=+0.01*max(I)*np.random.rand(num_points)
        ffts=[]
        for dataset in [V, I]:
            fft=1/num_points*np.fft.fftshift(np.fft.fft(dataset))
            abs_fft=abs(fft)
            fft[abs_fft<threshold*max(abs_fft)]=1
            ffts.append(fft)


        Z_f=np.divide(ffts[0], ffts[1])
        #plt.plot(times,V)
        #plt.show()

        abs_fft=np.abs(Z_f)
        #abs_V=np.abs(fft_V)
        #Z[i,:][abs_fft<threshold*max(abs_fft)]=0
        fft_freq=np.fft.fftshift(np.fft.fftfreq(num_points, times[1]-times[0]))
       
        plt_idx=np.where((fft_freq>(freqs[i]-(0.5*freqs[i]))) & (fft_freq<(freqs[i]+(0.5*freqs[i]))))
        #plt.loglog(fft_freq[plt_idx], abs_fft[plt_idx])
        #plt.plot(fft_freq, np.angle(ffts[0]))
        #plt.show()
        subbed_f=abs(np.subtract(fft_freq, freqs[i]))
        freq_idx=np.where(subbed_f==min(subbed_f))
        #plt.axvline(fft_freq[freq_idx], linestyle="--")
        
        impedances[i]=Z_f[freq_idx][0]
        #print(impedances)
        phases[i]=abs(np.angle(fft, deg=True))[freq_idx][0]
        #plt.show()
        #plt.plot(np.angle(fft, deg=True))
        #plt.show()
        magnitudes[i]=abs_fft[freq_idx][0]
    return phases, magnitudes,impedances, freqs
#plt.show()
cdl=1e-4
p,m,z,f=impedance_response(0, 8, 10, 10, cdl)
real=z.real
real[np.where(real<1e-3)]=0

data=np.column_stack((real, z.imag))
sim_class=EIS(circuit={ "z1":"C1"})
zsim=sim_class.test_vals({"C1":cdl}, f)


plt.scatter(z.real, -z.imag)
#ax=plt.gca()
#sim_class.nyquist(zsim, ax=ax, orthonormal=False, s=2)
plt.show()