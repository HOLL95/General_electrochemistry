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
plt.plot(np.real(fft))
plt.show()
plt.plot(np.imag(fft))
plt.show()
plt.plot(np.abs(fft))
plt.show()
Fourier_frequencies=np.fft.fftfreq(len(current), time[1]-time[0])
plt.plot(Fourier_frequencies, np.real(fft))
plt.show()
positive_index=np.where(Fourier_frequencies>0)
positive_frequencies=Fourier_frequencies[positive_index]
one_sided_FT=fft[positive_index]
plt.plot(positive_frequencies, np.imag(one_sided_FT))
plt.show()
plt.semilogy(positive_frequencies, np.abs(one_sided_FT))
plt.show()