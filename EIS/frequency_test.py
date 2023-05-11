import matplotlib.pyplot as plt
import numpy as np
def potential(amp,frequency, time, phase):
    return amp*np.sin(2*np.pi*frequency*time+phase)
def current(amp,frequency, time, phase):
    return amp*np.cos(2*np.pi*frequency*time+phase)
def impedance_response(min_f, max_f, points_per_decade, num_osc, sf=200, amplitude=5e-3):
    if (np.log2(sf)%2)!=0:
        sf=2**np.ceil(np.log2(sf))
    
    num_points=int(num_osc*sf)
    frequency_powers=np.linspace(min_f, max_f, (max_f-min_f)*points_per_decade)
    freqs=[10**x for x in frequency_powers]
    Z=np.zeros((len(freqs), num_points))
    threshold=0.5
    phase=0
    impedances=np.zeros(len(freqs), dtype="complex")
    magnitudes=np.zeros(len(freqs))
    phases=np.zeros(len(freqs))
    for i in range(0, len(frequency_powers)):
        time_end=num_osc/freqs[i]
        times=np.linspace(0, time_end, num_points, endpoint=False)
        V=potential(amplitude, freqs[i], times, phase)
        I=current(amplitude, freqs[i], times,phase)#+0.05*amplitude*np.random.rand(num_points)
        Z[i,:]=np.divide(V, I)
        #plt.plot(times,V)
        #plt.show()
        fft=1/num_points*np.fft.fftshift(np.fft.fft(V))
        abs_fft=np.abs(fft)
        fft[abs_fft<threshold*max(abs_fft)]=0

        fft[np.where(abs_fft<threshold*max(abs_fft))]=0
        max_idx=np.where(abs_fft==max(abs_fft))
        impedances[i]=fft[max_idx][0]
        phases[i]=abs(np.angle(fft, deg=True))[max_idx][0]
        plt.plot(np.angle(fft, deg=True))
        plt.show()
        magnitudes[i]=abs_fft[max_idx][0]
    return phases, magnitudes
p,m=impedance_response(0, 8, 10, 10)
plt.subplot(1,2,1)
plt.plot(p)
plt.subplot(1,2,2)
plt.plot(m)
plt.show()
