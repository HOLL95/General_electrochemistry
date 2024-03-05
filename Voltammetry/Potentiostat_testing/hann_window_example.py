import numpy as np
#hann_window=np.hanning(len(total_current))
#windowed_current=np.multiply(total_current, hann_window)


#import numpy as np
#import matplotlib.pyplot as plt
#FT=np.fft.fft(windowed_current)
#timestep=time[1]-time[0]
#frequencies=np.fft.fftfreq(len(windowed_current), timestep)
#plt.plot(np.fft.fftshift(frequencies), np.fft.fftshift(windowed_current))
#plt.show()


import numpy as np
import matplotlib.pyplot as plt
desired_harmonics= np.arange(0, 8, 1)#Plotting harmonics 0 to 7
num_desired_harmonics=len(desired_harmonics)
fft_harmonics=np.zeros((num_desired_harmonics, frequencies), dtype="complex") # The fourier spectrum is made up of complex numbers
box_width=0.5*desired_Hz # The region of the Fourier spectrum that we will extract around each harmonic as a fraction of the input frequency
envelope=True
for i in range(0, num_desired_harmonics):
    harmonic_i=i*desired_Hz
    
    if envelope==True:
        signs=[1]
    else:
        signs=[-1,1]
    for sign in signs:
        box_lower_bound=sign*harmonic_i-box_width
        box_upper_bound=sign*harmonic_i+box_width
        box_region=np.where((one_tail_frequency>box_lower_bound) & (one_tail_frequency<box_upper_bound))
        fft_harmonics[i,box_region]+=FT[box_region]#Only the frequency regions that are defined by the box are copied over - everything else stays as complex 0


for i in range(0, num_desired_harmonics):
    if envelope==True:
        plot_harmonic= 2*np.abs(np.fft.ifft(fft_harmonics[i,:]))
    else:
         plot_harmonic= np.real(np.fft.ifft(fft_harmonics[i,:]))
    plt.plot(time,) # We can now plot the time-domain representation (obtained using ifft) using an approrpriate function
    plt.show()