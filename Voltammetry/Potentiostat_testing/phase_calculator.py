import numpy as np
import matplotlib.pyplot as plt
import copy

files=[r"C:\Gamry\JB 120Hz 2uF 10 Ohm.txt"]
desire="Timeseries"
labels=["(B) Gamry"]
fig, ax=plt.subplots(1,2)


for j in range(0, len(files)):
    file=files[j]    
 
    
    current=np.loadtxt(file, skiprows=1, max_rows=1048545)
    time=current[:,0]
    reduction=np.where(time>2)
    voltage=current[:,1][reduction]
    current=current[:,2][reduction]
    time=time[reduction]
    ax[0].plot(time, current)
    ax[0].set_title("Current")
    ax[1].plot(time, voltage)
    ax[1].set_title("Potential")
    ax[0].set_xlabel("Time (S)")
    plt.show()

    
    freqs=np.fft.fftfreq(len(current), time[1]-time[0])
    freqs=np.fft.fftfreq(len(current), time[1]-time[0])
    Y=np.fft.fft(current)
    plt.plot(freqs, Y)
    plt.show()
    potential_Y=np.fft.fft(voltage)
    get_max=abs(freqs[np.where(Y==max(Y[np.where(freqs>10)]))][0])
    print("The input frequency is %.2f" %get_max)
    #plt.plot(freqs, potential_Y)
    #plt.plot(time, np.fft.ifft(potential_Y))
    m=copy.deepcopy(potential_Y)
    m=np.zeros(len(voltage), dtype="complex")
    m[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]=potential_Y[np.where((freqs<1.5*get_max) & (freqs>0.5*get_max))]
    potential_Y[np.where((freqs<0.25*get_max) & (freqs>-0.25*get_max))]=0
    #plt.plot(freqs, potential_Y)
    ac_component=voltage#np.real(np.fft.ifft(potential_Y))

  
        
    num_periods=int(np.floor(time[-1]*get_max))
    periods=list(range(1, num_periods))
    phases=np.zeros((2, num_periods-1))
    for i in range(0, num_periods-1):
        #print(i)
        idx=np.where((time>(i/get_max))& (time<((i+1)/get_max)))
        s=np.sin(2*np.pi*get_max*time[idx])      # reference sine, note the n*t
        c=np.cos(2*np.pi*get_max*time[idx])  
        sines=[current[idx], ac_component[idx]]
        #To verify this method works uncomment these two lines
        #check_phase=70
        
        #sines[1]=np.sin(2*np.pi*get_max*time[idx]+(check_phase*(np.pi/180)))

        for m in range(0, len(sines)):
            
            sinusoid=sines[m]
            
            xs,xc=sinusoid*s,sinusoid*c
            a,b=2*np.mean(xs),2*np.mean(xc)
            mag=np.hypot(b,a)
            rad=np.arctan2(b,a)
            deg=rad*180/np.pi
            phases[m][i]=deg
   
    
   
    fig,ax=plt.subplots(1,3)
    
    labels=["Voltage-time shift", "Current-time shift", "Voltage-current shift"]
    datasets=[phases[1,:], phases[0,:],phases[1,:]-phases[0,:]]
    for m in range(0, 3):
        ax[m].set_xlabel("Period")
        ax[m].set_ylabel("Phase ($^\\circ$)")
        ax[m].set_title(labels[m])
        ax[m].plot(periods, datasets[m])

plt.show()
