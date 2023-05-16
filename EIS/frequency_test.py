import matplotlib.pyplot as plt
import numpy as np
def potential(amp,frequency, time, phase):
    return amp*np.sin(2*np.pi*frequency*time+phase)
def current(cdl,frequency, time, phase):
    return (cdl)*frequency*np.cos(2*np.pi*frequency*time+phase)
def impedance_response(min_f, max_f, points_per_decade, num_osc, sf=200, amplitude=5e-3):
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
    for i in range(0, len(frequency_powers)):
        time_end=num_osc/freqs[i]
        times=np.linspace(0, time_end, num_points, endpoint=False)
        V=potential(amplitude, freqs[i], times, phase)
        I=current(cdl, freqs[i], times,phase)
        I+=+0.01*max(I)*np.random.rand(num_points)
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
    return phases, magnitudes,impedances
#plt.show()
cdl=1e-3
p,m,z=impedance_response(0, 8, 10, 10)
real=z.real
real[np.where(real<1e-3)]=0
plt.plot(z.real, -z.imag)
plt.show()
fig, ax =plt.subplots(1,2)
ax[0].plot(p)
ax[1].semilogy(m)
plt.show()