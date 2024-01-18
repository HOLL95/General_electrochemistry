import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
from PIL import Image
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from harmonics_plotter import harmonics
from multiplotter import multiplot
num_harms=6
figure=multiplot(3, 4, **{"harmonic_position":[1,2], "num_harmonics":num_harms, "orientation":"landscape", "plot_width":6, "row_spacing":2,"col_spacing":2, "plot_height":1})

bigax=figure.merge_harmonics(2, 1)

axes=figure.axes_dict
loc=loc="/home/userfs/h/hll537/Documents/Experimental_data/Nat/"
file1="NGB-ECHEM(01)-025 FTacV ELTON 0.01 mM Fc 104.31 mVs-1 80 mV amp 72 Hz @ GC_data_export_cv_current"
file2="NGB-ECHEM(01)-025 FTacV ELTON 0.01 mM Fc 104.31 mVs-1 80 mV amp 72 Hz @ GC_data_export_cv_voltage"
harmonics_range=list(range(0, num_harms))
h_class=harmonics(harmonics_range, 72.04862601258495, 0.25)
current_data=np.loadtxt(loc+file1)
current=current_data[:,1]
time=current_data[:,0]
potential=np.loadtxt(loc+file2)[:,1]
axes["row1"][0].plot(time, potential)
axes["row1"][2].plot(time,  current)
frequency=np.fft.fftfreq(len(time), time[1]-time[0])
hanning=np.hanning(len(time))
Y=np.fft.fft(current*hanning)

axes["row1"][3].semilogy(np.fft.fftshift(frequency), np.fft.fftshift(abs(Y)))
axes["row1"][3].set_xlim([-500, 500])
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
h_class.harmonic_selecter(bigax, current, time, extend=1, log=True, arg=abs)
real_axis=axes["row2"][h_class.num_harmonics*2:h_class.num_harmonics*3]
plot_dict={"Real_time_series":current, "plot_func":np.real, "axes_list":real_axis, "legend":None}#"hanning(Real)_time_series":hanning*current,
h_class.plot_harmonics(time, **plot_dict  )
imag_axis=axes["row2"][h_class.num_harmonics*3:]
plot_dict={"Imag_time_series":current, "plot_func":np.imag, "axes_list":imag_axis, "colour":colours[1], "legend":None}#"hanning(Imag)_time_series":hanning*current,
h_class.plot_harmonics(time, **plot_dict  )
for arg,linestyle in zip([np.real, np.imag], ["-","--"]):
    fourier_bits, freqs=h_class.generate_harmonics(time, current, hanning=False, return_fourier=True)
    for i in range(0, h_class.num_harmonics):
        axes["row2"][h_class.num_harmonics+i].plot(freqs, arg(fourier_bits[i]), linestyle=linestyle)
abs_axis=axes["row3"][:h_class.num_harmonics]
plot_dict={"Abs_time_series":current,"Hanning_time_series":hanning*current, "plot_func":np.abs, "axes_list":abs_axis, "legend":None}#"hanning(Imag)_time_series":hanning*current,
h_class.plot_harmonics(time, **plot_dict  )
Dc_axis=axes["row3"][h_class.num_harmonics:h_class.num_harmonics*2]
plot_dict={"DC_axis_time_series":current*hanning,"plot_func":np.abs, "axes_list":Dc_axis, "DC_component":True, "xaxis":potential, "legend":None}#"hanning(Imag)_time_series":hanning*current,
h_class.plot_harmonics(time, **plot_dict  )
axes["row1"][0].plot(time, h_class.dc_pot)
image=np.asarray(Image.open(loc+"3elec.png"))
axes["row1"][1].imshow(image)
plt.subplots_adjust(top=0.95,
                    bottom=0.11,
                    left=0.07,
                    right=0.955,
                    hspace=0.2,
                    wspace=0.2)
plt.show()