freqs=np.fft.fftfreq(len(voltage), time[1]-time[0])
potential_Y=np.fft.fft(voltage)
current_Y=np.fft.fft(current)
fundamental_hz=freqs[np.where(Y==max(Y[np.where(freqs>10)]))][0]#obtaining fundamental frequency
potential_Y[np.where((freqs<0.25*fundamental_hz) & (freqs>-0.25*fundamental_hz))]=0
ac_component=np.real(np.fft.ifft(potential_Y))#Removing the DC component
#If the signal is aperiodic you will get "ringing" elements at the start 
#amd end of the timeseries. For this analysis, it is recommended you
#truncate the signal accordingly
num_periods=int(np.floor(time[-1]*fundamental_hz))
periods=list(range(1, num_periods))
phases=np.zeros((2, num_periods-1))
for i in range(0, num_periods-1):
    idx=np.where((time>(i/fundamental_hz))& (time<((i+1)/fundamental_hz)))
    s=np.sin(2*np.pi*fundamental_hz*time[idx])
    c=np.cos(2*np.pi*fundamental_hz*time[idx])  
    sines=[current[idx], ac_component[idx]]
    #To verify this method works uncomment these two lines
    #check_phase=70
    
    #sines[1]=np.sin(2*np.pi*fundamental_hz*time[idx]+(check_phase*(np.pi/180)))

    for m in range(0, len(sines)):
        
        sinusoid=sines[m]
        xs=sinusoid*s
        xc=sinusoid*c
        a=2*np.mean(xs)
        b=2*np.mean(xc)
        mag=np.hypot(b,a)
        rad=np.arctan2(b,a)
        deg=rad*180/np.pi
        phases[m][i]=deg
