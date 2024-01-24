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


loc=loc="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_1/"
file1="FTacV_(MONASH)_0.1_mM_Fc_72.05_Hz_cv_current"
file2="FTacV_(MONASH)_0.1_mM_Fc_72.05_Hz_cv_voltage"
harmonics_range=list(range(0, num_harms))
num_harmonics=len(harmonics_range)
h_class=harmonics(harmonics_range, 72.04862601258495, 0.25)
current_data=np.loadtxt(loc+file1)
current=current_data[:,1]
time=current_data[:,0]
potential=np.loadtxt(loc+file2)[:,1]
fig, ax=plt.subplots(num_harmonics, 1)
one_sided=h_class.generate_harmonics(time, current, one_sided=True)
two_sided=h_class.generate_harmonics(time, current, one_sided=False)
for i in range(0,num_harmonics):
    
    ax[i].plot(time, np.real(two_sided[i,:]))
    ax[i].plot(time, 2*np.abs(one_sided[i,:]))
plt.show()