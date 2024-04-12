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
import numpy as np
import pints
from pandas import read_excel
data_loc="/home/henryll/Documents/Experimental_data/Alice/Galectin_5_4/"
directories=os.listdir(data_loc)
header=2
footer=1
fig, ax=plt.subplots(1,1)
def get_conc(files):
    concs=["" for x in range(0, len(files))]
    

fig, axes=plt.subplots(3, 5)
directory_counter=-1
for directory in directories:
    if "GFF" in directory:
        directory_counter+=1
        new_loc=data_loc+directory+"/"
        files=os.listdir(new_loc)
        conc_list=[]

        for i in range(0, len(files)):
            if ".xlsx" in files[i]:
                removed=files[i].split(".")
                curr_file=("").join(removed[:-1])
                id_list=curr_file.split("_")
                conc_idx=id_list.index("SPE")+2
                decimal_idx=conc_idx+1
                try:
                    decimal=int(id_list[decimal_idx])*0.1
                    conc_list.append((int(id_list[conc_idx])+decimal,i))
                except:
                    conc_list.append((int(id_list[conc_idx]),i))
        concs_only=[x[0] for x in conc_list]
        sort_index = np.argsort(concs_only)
        sorted_concs=[concs_only[x] for x in sort_index]
        file_list=[files[conc_list[x][1]] for x in sort_index]
        for i in range(0, len(file_list)):
           
            ax=axes[directory_counter//5, directory_counter%5]
            name=file_list[i] 
            pd_data=read_excel(new_loc+name, skiprows=header, skipfooter=footer)
            data=pd_data.to_numpy(copy=True, dtype='float')
            fitting_data=np.column_stack((np.flip(data[:,3]), -np.flip(data[:,4])))

            frequencies=np.flip(data[:,0])*2*np.pi
            EIS().nyquist(fitting_data, ax=ax, label=sorted_concs[i], orthonormal=False)
        ax.legend()
plt.show()
