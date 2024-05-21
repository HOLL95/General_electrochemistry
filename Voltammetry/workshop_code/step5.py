##########STEP1#############
import numpy as np
data_location="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_3"
current_file=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_current")
potential_file=current=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_voltage")
current=current_file[:,1]
time=current_file[:,0]
potential=potential_file[:,1]
########STEP5################(Changes the results from previous steps and so is placed above them in the code)
import scipy as sp
dec_amount=13
print("Pre-decimation")
print(len(current), len(potential), len(time))
current=sp.signal.decimate(current, dec_amount)
potential=sp.signal.decimate(potential, dec_amount)
time=sp.signal.decimate(time, dec_amount)
print("Post-decimation")
print(len(current), len(potential), len(time))#

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
