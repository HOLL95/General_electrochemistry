import matplotlib.pyplot as plt
import math
import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_class import EIS
from EIS_optimiser import EIS_genetics
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from scipy.signal import decimate
from multiplotter import multiplot

data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"


file_name="2023-10-10_FTV_GC-Green_Fc_cv_"

#figure=multiplot(3, 1, **{"harmonic_position":0, "num_harmonics":4, "orientation":"portrait", "fourier_position":1, "plot_width":5, "row_spacing":1, "plot_height":1})
dec_amount=8
harm_range=list(range(4, 8))
ramped_dec_amount=32
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
volt_data=voltage_data_file[0::dec_amount, 1]
plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}



plot_dict={"current":current_data_file[0::ramped_dec_amount,1], "time":current_data_file[0::ramped_dec_amount,0], "potential":current_data_file[0::ramped_dec_amount,1]}
for i in range(1, 3):
    axes_list=figure.axes_dict["col1"][i*ramped_h_class.num_harmonics:(i+1)*ramped_h_class.num_harmonics]
    ramped_h_class.plot_harmonics(plot_dict["time"], current_time_series=plot_dict["current"],hanning=True, plot_func=abs, axes_list=axes_list)
plt.show()