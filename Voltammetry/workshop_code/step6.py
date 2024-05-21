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
plt.plot(time, potential)
plt.plot(time, synthetic_potential)
plt.show()