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
from pints.plot import trace
def get_conc(files):
    concs=["" for x in range(0, len(files))]
    
import copy
conc_dict={}
fig, axes=plt.subplots(3, 5)
param_fig, param_axes=plt.subplots(3,5)
directory_counter=-1
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
zero_loc="/home/henryll/Documents/Experimental_data/Alice/Galectin_5_4/zero_points/"
mode="nyquist"
big_param_fig, bg_pax=plt.subplots()
tc_dict={}
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
        #twinx=axes[directory_counter//5, directory_counter%5].twinx()
        time_constant_array=[]
        for i in range(0, len(file_list)):
            name=file_list[i]
            associated_conc=sorted_concs[i]
            ax=axes[directory_counter//5, directory_counter%5]
            removed=(new_loc+name).split(".")
            curr_file=("").join(removed[:-1])
            
            
            pd_data=read_excel(new_loc+name, skiprows=header, skipfooter=footer)
            
            data=pd_data.to_numpy(copy=True, dtype='float')
            if i==0:
                reference=np.column_stack((np.flip(data[:,3]), -np.flip(data[:,4])))
                removed=(zero_loc+name).split(".")
                curr_file=("").join(removed[:-1])
                params=np.load(curr_file+".npy")
            else:
                params=np.load(curr_file+".npy")
                #trace(params)
                #plt.show()
            r1=np.mean(params[0, 5000:, 2])
            q1_mean=np.mean(params[0, 5000:, 3])
            alpha=np.mean(params[0, 5000:, 4])
            c_eff=(q1_mean/(r1**-alpha))
            time_constant=10**(np.log10(r1*c_eff)/(1/(alpha-1)))
            time_constant_array.append(time_constant)
            param_axes[directory_counter//5, directory_counter%5].scatter(sorted_concs[i], time_constant)
                #print(np.mean(r_chain), np.mean(q_chain), np.mean(alpha_chain))
            
            fitting_data=EIS().normalise_spectra(np.column_stack((np.flip(data[:,3]), -np.flip(data[:,4]))),  method=None)
            frequencies=np.flip(data[:,0])*2*np.pi
            if i==0:
                lw=2
            else:
                lw=1
            if mode=="nyquist":
                EIS().nyquist(fitting_data, ax=ax, label=sorted_concs[i], orthonormal=False, disable_negative=True,lw=lw), 
            elif mode=="bode":
                
                fitting_data=EIS().convert_to_bode(fitting_data)
                
                EIS().bode(fitting_data,frequencies, ax=ax, label=sorted_concs[i], data_type="phase_mag", twinx=twinx)
                fitting_data=np.column_stack((fitting_data, frequencies))
            current_keys=list(conc_dict.keys())
            if associated_conc in current_keys:
                conc_dict[associated_conc].append(fitting_data)
                tc_dict[associated_conc].append(time_constant)
            else:
                conc_dict[associated_conc]=[fitting_data]
                tc_dict[associated_conc]=[time_constant]
        bg_pax.plot(sorted_concs[1:], time_constant_array[1:], marker="o")
        
        ax.legend()
all_concs_sorted=sorted(list(tc_dict.keys()))
all_data=np.array([[np.mean(tc_dict[conc]), np.std(tc_dict[conc])] for conc in all_concs_sorted ])
bg_pax.set_xlabel("Galectin concentration (mg/ml)")
bg_pax.set_ylabel("Estimated time constantn")
#bg_pax.plot(all_concs_sorted, all_data[:,0], color="blue", lw=2, linestyle="--")
#bg_pax.fill_between(all_concs_sorted, all_data[:,0]+all_data[:,1], all_data[:,0]-all_data[:,1])
bg_pax.legend()     
            
full_conc_list=sorted(list(conc_dict.keys()))
fig2, axes2=plt.subplots(2,5)
monster_fig, monster_ax=plt.subplots()
monster_twinx=monster_ax.twinx()
for j in range(0, len(full_conc_list)):
    ax=axes2[j//5, j%5]
    #twinx=ax.twinx()
    ax.set_title(full_conc_list[j])
    file_list=conc_dict[full_conc_list[j]]
    for i in range(0, len(file_list)):
        if mode=="nyquist":
            EIS().nyquist(file_list[i], ax=monster_ax, orthonormal=False, disable_negative=True, colour=colours[j])
            EIS().nyquist(file_list[i], ax=ax, orthonormal=False, disable_negative=True)
        elif mode=="bode":
            EIS().bode(file_list[i][:,0:2], file_list[i][:,-1], ax=ax, twinx=twinx,data_type="phase_mag")
            EIS().bode(file_list[i][:,0:2], file_list[i][:,-1], ax=monster_ax,  colour=colours[j], twinx=monster_twinx,data_type="phase_mag")
import matplotlib.lines as mlines
lines = [mlines.Line2D([], [], color=colours[x],label=full_conc_list[x]) for x in range(0, len(full_conc_list))]
monster_ax.legend(handles=lines)
plt.show()
