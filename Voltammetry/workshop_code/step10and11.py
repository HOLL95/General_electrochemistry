##########STEP1#############
import numpy as np
data_location="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_3"
current_file=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_current")
potential_file=current=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_voltage")
current=current_file[:,1]
time=current_file[:,0]
potential=potential_file[:,1]
#########STEP2###############
import matplotlib.pyplot as plt
########STEP3###############
fft=np.fft.fft(current)
Fourier_frequencies=np.fft.fftfreq(len(current), time[1]-time[0])
positive_index=np.where(Fourier_frequencies>0)
positive_frequencies=Fourier_frequencies[positive_index]
one_sided_FT=fft[positive_index]
########STEP4###############
max_index=np.where(fft==max(one_sided_FT))
max_frequency=Fourier_frequencies[max_index][0]
########STEP5###############
########STEP6###############
amplitude=80e-3
phase=0
AC_component=amplitude*np.sin(2*np.pi*max_frequency*time+phase)
E_start=0.35
E_reverse=1.05
v=104.31e-3
DC_range=E_reverse-E_start
tr=(E_reverse-E_start)/v
DC_component=E_reverse-(np.abs(time-tr))*v
synthetic_potential=DC_component+AC_component
########STEP10+11###############
import copy
def generate_harmonics(time, current, max_harmonic, minimum_harmonic, plot_function, window_extent=0.5):
    if plot_function=="envelope":
        sign_list=[1]
        factor=2
        function=np.abs
    elif plot_function=="real" or plot_function=="imag":
        sign_list=[-1,1]
        factor=1
        if  plot_function=="real":
            function=np.real
        else:
            function=np.imag
        
    else:
        raise ValueError("{0} is not a valid plot function".format(plot_function))
    fft=np.fft.fft(current)
    num_harmonics=max_harmonic-minimum_harmonic
    harmonic_range=list(range(minimum_harmonic, max_harmonic))
    filtered_spectra_array=np.zeros((len(fft), num_harmonics), dtype="complex")
    time_domain_harmonic_array=np.zeros((len(time), num_harmonics))
    for i in range(0, num_harmonics):
        for sign in sign_list:
            peak_position=max_frequency*i*sign
            box_min=peak_position-(window_extent*max_frequency)
            box_max=peak_position+(window_extent*max_frequency)
            window_index=np.where((Fourier_frequencies>box_min) & (Fourier_frequencies<box_max))
            filtered_spectra_array[window_index, i]=copy.deepcopy(fft[window_index])
        time_domain_harmonic_array[:,i]=factor*function(np.fft.ifft(filtered_spectra_array[:,i])) 
    return time_domain_harmonic_array
max_harmonic=7
minimum_harmonic=0
window_extent=0.5
num_harmonics=max_harmonic-minimum_harmonic
fig,axes_list=plt.subplots(num_harmonics, 1)
real=generate_harmonics(time, current*np.hanning(len(current)), max_harmonic, minimum_harmonic, "envelope")
envelope=generate_harmonics(time, current, max_harmonic, minimum_harmonic, "envelope")
xaxis=time
for i in range(0, num_harmonics):
    axes_list[i].plot(xaxis, real[:,i])
    #axes_list[i].plot(xaxis, envelope[:,i])
    
plt.show()