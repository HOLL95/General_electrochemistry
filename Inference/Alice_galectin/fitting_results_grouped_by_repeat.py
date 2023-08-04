import matplotlib.pyplot as plt
import math
import os
import re
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
import numpy as np
import pints
from pandas import read_csv
data_loc="/home/henney/Documents/Oxford/Experimental_data/Alice/Galectin_31_7/"
files=os.listdir(data_loc)
header=6
footer=2

model_dict={"CPE":{"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", "W1"]}},
            "2CPE":{"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", ("Q2", "alpha2")]}},
            #"C":{"z1":"R0", "z2":{"p_1":"C2","p_2":["R1", "W1"]}}
            }
names={"CPE":["R0", "R1", "Q1", "alpha1", "W1"],
        "2CPE":["R0", "R1", "Q1", "alpha1" ,"Q2", "alpha2"]
}
marker_dict={"CPE":{"line":"-", "marker":"o"},
        "2CPE":{"line":"--", "marker":"v"},
}
model_names=list(names.keys())
boundaries={"R0":[0, 1e4,],
            "R1":[1e-6, 1e6], 
            "C2":[0,1],
            "Q2":[0,1], 
            "alpha2":[0,1],
            "Q1":[0,1], 
            "alpha1":[0,1],
            "W1":[0,1e6]}

get_conc=re.compile(r"0\.[0-9]+(?=\.csv$)")

#
#concentration
#->Repeat
#---->Data
#---->Model
#------->Data type
#---------->Parameter values
#---------->Generated fits
results_file=np.load("alice_fitting_results_2.npy", allow_pickle=True).item()
concentrations=list(results_file.keys())
number_list=[float(x) for x in concentrations]
enumerated_list=sorted(enumerate(number_list), key=lambda x:x[1])
concentrations=[concentrations[x[0]] for x in enumerated_list]

num_repeats=3

modes=["bode", "nyquist"]
plot_class=EIS()
repeat_dict={"2023-07-14_SPE-P-DS_1":"1", 
            "2023-07-27_SPE-P-DS_1":"2",
            "2023-07-27_SPE-P-DS_5":"3"}
repeat_dict={v: k for k, v in repeat_dict.items()}
for model in ["CPE", "2CPE"]:
    for mode in modes:
        plots=plt.subplots(num_repeats, 2)
        for repeat in range(0, num_repeats):
            current_axes=plots[1]
            bode_axes=current_axes[repeat,0]
            #bode_axes.text(x=1.2, y=0, s=text, fontsize=12, transform=bode_axes.transAxes)
            bode_twinx=bode_axes.twinx()
            nyquist_axes=current_axes[repeat,1]
            for i in range(0, len(concentrations)):
                if concentrations[i]=="0":
                    text="Blank"
                else:
                    text=concentrations[i]

            

                
            

                
                if i==0 and repeat==0:
                    no_labels=False
                else:
                    no_labels=False
                repeat_str=str(repeat+1)
                fitting_data=simulated_data=results_file[concentrations[i]][repeat_str]["Data"]
                plot_frequencies=simulated_data=results_file[concentrations[i]][repeat_str]["Frequencies"]
                plot_class.bode(fitting_data, plot_frequencies, compact_labels=True, ax=bode_axes, twinx=bode_twinx, lw=1, no_labels=no_labels, alpha=0.75, scatter=1, markersize=10, line=False)
                plot_class.nyquist(fitting_data, ax=nyquist_axes, label=text, orthonormal=False, lw=1, alpha=0.75, scatter=1, markersize=10, line=False)
                bode_axes.set_title(repeat_dict[repeat_str])
                nyquist_axes.set_title(repeat_dict[repeat_str])

                #sim_class=EIS(circuit=model_dict[model], fitting=True, parameter_bounds=boundaries, normalise=True)
                simulated_data=results_file[concentrations[i]][repeat_str][model][mode]["Fit"]
                #print(simulated_data)
                plot_class.bode(simulated_data, plot_frequencies, compact_labels=True, ax=bode_axes, twinx=bode_twinx, no_labels=no_labels)
                plot_class.nyquist(simulated_data, ax=nyquist_axes, orthonormal=False, label=text)
        plots[1][0, -1].legend()
        plots[0].set_size_inches(9, 9)
        plots[0].subplots_adjust(left  = 0.09,
                            right = 0.965,  
                            bottom = 0.05, 
                            top = 0.97,    
                            wspace = 0.35, 
                            hspace = 0.35)
        plots[0].savefig("{0}-Model_{1}_Mode_fits.png".format(model, mode), dpi=500)
                
       