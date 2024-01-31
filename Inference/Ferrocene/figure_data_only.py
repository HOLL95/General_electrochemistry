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
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
from harmonics_plotter import harmonics
import numpy as np
import pints
from scipy.optimize import minimize
from pints.plot import trace
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
#data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)

truncate=10
truncate_2=1

real=np.flip(data[truncate:,0])
imag=np.flip(data[truncate:,1])

frequencies=np.flip(data[truncate:,2])
trunc_real=np.flip(data[:truncate+1,0])
trunc_imag=np.flip(data[:truncate+1,1])

trunc_frequencies=np.flip(data[:truncate+1,2])
spectra=np.column_stack((real, imag))
trunc_spectra=np.column_stack((trunc_real, trunc_imag))


fig,ax =plt.subplots(1, 2)
twinx=ax[1].twinx()
EIS().nyquist(spectra, orthonormal=False, scatter=1, ax=ax[0])
EIS().bode(spectra, frequencies, ax=ax[1], twinx=twinx, compact_labels=True)
EIS().nyquist(trunc_spectra, orthonormal=False, scatter=1, ax=ax[0], colour="red")
EIS().bode(trunc_spectra, trunc_frequencies, ax=ax[1], twinx=twinx, compact_labels=True, colour="red")
fig=plt.gcf()
fig.set_size_inches(7, 4.5)

plt.tight_layout()
letters=["A", "B"]
for i in range(0, len(letters)):
    axes=ax[i]
    axes.text(0.2+(i*0.6), 0.9,
         "(%s)"%letters[i],
         fontweight="bold",
         fontsize=12,
         transform=axes.transAxes)
plt.show()
fig.savefig("Data_only.png", dpi=500)