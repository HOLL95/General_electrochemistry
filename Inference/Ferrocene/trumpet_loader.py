

import matplotlib.pyplot as plt
import pints

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
from heuristic_class import Automated_trumpet

data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Immobilised_Fc/GC-Yellow_(2023-10-10)/Fc/TP_Exported/"
all_files=os.listdir(data_loc)
filename="fc_dc_positions_1.txt"
Automated_trumpet(file_list=all_files, trumpet_file_name=filename,data_loc=data_loc, area=0.07, skiprows=2)