import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser, EIS_genetics
from circuit_drawer import circuit_artist
from pandas import read_csv
mv5_plots={"Henry":[0,3], "Alice":[0], "filename":"Electroactive_protein.png"}
henry_plots={"Henry":[0,1,2], "Alice":[], "filename":"Henry_cjx.png"}
alice_plots={"Henry":[], "Alice":[0,1,2], "filename":"Alice_cjx.png"}
henry_files=["EIS_WT_0.005V.csv",
        "EIS_WT_0.2V.csv",
        "EIS_WT_-0.3V.csv",
        "EIS_blank_0.005V_wide_window.csv",]
henry_labels=["5mV (Henry)",
        "200mV (Henry)",
        "-300mV (Henry)",
        "Blank (Henry)"]
alice_files=[
        '0.1_M_Phosphate_CjX183_PGE_EIS_0.005_V_5deg.csv',
        '0.1_M_Phosphate_CjX183_PGE_EIS_0.2_V_5deg.csv',
        '0.1_M_Phosphate_CjX183_PGE_EIS_-0.3_V_5deg.csv'
        ]
alice_labels=["5mV (Alice)",
        "200mV (Alice)",
        "-300mV (Alice)"
        ]
for info_dict in [mv5_plots, henry_plots, alice_plots]:
    alice_dict={
                "file_loc":"Experimental_data/Alice_15_8_22/",
                "files":[alice_files[x] for x in info_dict["Alice"]],
                "real_loc":4,
                "imag_loc":5,
                "linestyle":"-",
                "labels":[alice_labels[x] for x in info_dict["Alice"]],
                "encoding":"UTF-16",
                "skiprows":5,
                "DCV_file":"0.1_M_Phosphate_CjX183_PGE_DCV_5deg.csv",
                "marker":"x"

    }

    henry_dict={
                "file_loc":"Experimental_data/5_7_22/",
                "files":[henry_files[x] for x in info_dict["Henry"]],
                "real_loc":6,
                "imag_loc":7,
                "linestyle":"-",
                "labels":[henry_labels[x] for x in info_dict["Henry"]],
                "encoding":"unicode-escape",
                "skiprows":2,
                "DCV_file":"DCV_WT_pre_EIS.csv",
                "marker":"o"
    }
    plot_dicts=[henry_dict, alice_dict]
    DCV_labels=["Henry" ,"Alice"]

    get_color=plt.rcParams['axes.prop_cycle'].by_key()['color']
    get_color[3]="lightslategrey"
    labels=["WT", "blank"]
    get_color[2]="red"
    c_idx=0
    #twinx=ax[2].twinx()
    plotter=EIS()
    encoding=["unicode-escape", "UTF-16"]

    xaxis=["Potential(V)", "$Z_{re}$", "$log_{10}$(Freq)", "Potential(V)"]
    yaxis=["Current($\\mu A$)", "$Z_{im}$", "$Z_{mag}$", "Current($\\mu A$)"]
    fig, ax=plt.subplots(1, 3)
    plot_locs={"5mV":[1, 0], "200mV":[1, 1], "-300mV":[1, 2]}

    plot_loc_keys=plot_locs.keys()
    for i in range(0, len(plot_dicts)):
        current_dict=plot_dicts[i]
        for j in range(0, len(current_dict["files"])):

                data=read_csv(current_dict["file_loc"]+current_dict["files"][j], sep=",", encoding=current_dict["encoding"], engine="python", skiprows=current_dict["skiprows"], skipfooter=1)
                if j==0:

                    DCV_data=read_csv(current_dict["file_loc"]+current_dict["DCV_file"], sep=",", encoding=current_dict["encoding"], engine="python", skiprows=current_dict["skiprows"], skipfooter=1)
                    DCV_numpy=DCV_data.to_numpy(copy=True, dtype='float')
                    potential=DCV_numpy[:,0]
                    current=DCV_numpy[:,1]
                    ax[2].plot(potential, current, linestyle=current_dict["linestyle"],color= get_color[i+2], label=DCV_labels[i])
                    ax[2].set_xlabel("Potential (V)")
                    ax[2].set_ylabel("Current ($\\mu$A)")
                numpy_data=data.to_numpy(copy=True, dtype='float')

                real=(numpy_data[:, current_dict["real_loc"]])
                imag=-(numpy_data[:,current_dict["imag_loc"]])
                freq=np.log10(numpy_data[:,0])
                #phase=numpy_data[:,2]
                #mag=numpy_data[:,5]
                spectra=np.column_stack((real, imag))
                trunc_vals=[35, 25]
                for k in range(0, len(trunc_vals)):
                    plot_spec=np.column_stack((real[:-trunc_vals[k]], imag[:-trunc_vals[k]]))
                    plotter.nyquist(plot_spec, ax=ax[k], scatter=1,  linestyle=current_dict["linestyle"], marker=current_dict["marker"],label=current_dict["labels"][j],)
                #ax[0,2].plot(0,0,  color=get_color[j], linestyle=current_dict["linestyle"])
                #if current_dict["labels"][j] in plot_loc_keys:
            #        current_label=current_dict["labels"][j]
        #            plotter.nyquist(spectra, label=current_dict["labels"][j], ax=ax[plot_locs[current_label][0],plot_locs[current_label][1]], scatter=1, colour=get_color[j], linestyle=current_dict["linestyle"], marker=current_dict["marker"])
        #            ax[plot_locs[current_label][0],plot_locs[current_label][1]].set_title(current_label)
                """ax[1].scatter(real, imag, color=get_color[c_idx])
                ax[1].plot(real,imag, label=label_dict[file], color=get_color[c_idx])
                ax[2].plot(freq,mag, color=get_color[c_idx], label=label_dict[file],)
                ax[2].scatter(freq,mag, color=get_color[c_idx])
                twinx.plot(freq, phase, color=get_color[c_idx])
                twinx.set_ylabel("Phase")
                twinx.scatter(freq, phase, color=get_color[c_idx])
                c_idx+=1"""
    #ax[0,2].legend()
    ax[0].legend()
    ax[2].legend()
    #ax[0,2].set_axis_off()
    plt.subplots_adjust(top=0.955,
    bottom=0.110,
    left=0.085,
    right=0.96,
    hspace=0.315,
    wspace=0.315)
    fig.set_size_inches(7, 4.5)
    plt.show()
    fig.savefig(info_dict["filename"], dpi=500)
