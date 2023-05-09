import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:dir_list.index("General_electrochemistry")+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import math
import numpy as np
import matplotlib.pyplot as plt
from single_e_class_unified import single_electron
from square_scheme import square_scheme
from multiplotter import multiplot
from harmonics_plotter import harmonics
import time
import pints
from pints import plot

harm_range=list(range(1, 13))
cc=0

param_list={
       "E_0":0.0,
        'E_start':  -500e-3, #(starting dc voltage - V)
        'E_reverse':300e-3,
        'omega':10,
        "v":50e-3,  #    (frequency Hz)
        'd_E': 150e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 0,  #     (uncompensated resistance ohms)
        'Cdl': 1e-4, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        'sampling_freq' : (1.0/200),
        'phase' :3*math.pi/2,
        "time_end": None,
        'num_peaks': 30,

    }


simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "disperson":False,
    "dispersion_bins":[16],
    "test": False,
    "method": "ramped",
    "phase_only":False,
    "likelihood":"timeseries",
    "numerical_method": "Brent minimisation",
    "label": "MCMC",
    "optim_list":[],

}
other_values={
    "filter_val": 0.5,
    "harmonic_range":harm_range,
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],
    'Ru': [0, 1e3],  
    'Cdl': [0,1e-3], 
    'CdlE1': [-1e-2,1e-2],
    'CdlE2': [-5e-4,5e-4],
    'CdlE3': [-1e-4,1e-4],
    'gamma': [0.1*param_list["original_gamma"],1e-9],
    'k_0': [0.1, 1000], 
    'alpha': [0.35, 0.65],
    "cap_phase":[math.pi/2, 2*math.pi],
    'phase' : [0, 2*math.pi],
}
linkage_dict=elements=["AoRo", "AoRr", "AiRo", "AiRr", "ArRo", "ArRr"]
EEEEE={   
            "AoRo":{"AoRr":{"type":"BV_red", "group":None},},
            "AoRr":{"AoRo":{"type":"BV_ox", "group":None} ,"AiRo":{"type":"BV_red", "group":None}, },
            "AiRo":{"AoRr":{"type":"BV_ox", "group":None},"AiRr":{"type":"BV_red", "group":None}},
            "AiRr":{"AiRo":{"type":"BV_ox", "group":None},"ArRo":{"type":"BV_red", "group":None}},
            "ArRo":{"AiRr":{"type":"BV_ox", "group":None},"ArRr":{"type":"BV_red", "group":None}},
            "ArRr":{"ArRo":{"type":"BV_ox", "group":None}},
}
EECR={   
            "AoRo":{"AoRr":{"type":"BV_red", "group":None},"ArRo" :{"type":"Cat", "group":None}},
            "AoRr":{"AoRo":{"type":"BV_ox", "group":None} ,"AiRo":{"type":"Cat", "group":None}, "ArRr":{"type":"Cat", "group":None}},
            "AiRo":{"AoRr":{"type":"Cat", "group":None},"AiRr":{"type":"BV_red", "group":None}},
            "AiRr":{"AiRo":{"type":"BV_ox", "group":None},"ArRo":{"type":"Cat", "group":None}},
            "ArRo":{"AoRo":{"type":"Cat", "group":None},"AiRr":{"type":"Cat", "group":None},"ArRr":{"type":"BV_red", "group":None}},
            "ArRr":{"ArRo":{"type":"BV_ox", "group":None},"AoRr":{"type":"Cat", "group":None}},
}

EEC={   
    "Ao":{"Ai":{"type":"BV_red", "group":None}, "Ar":{"type":"Cat", "group":None}},
    "Ai":{"Ao":{"type":"BV_ox", "group":None}, "Ar":{"type":"BV_red", "group":None}},
    "Ar":{"Ai":{"type":"BV_ox", "group":None}, "Ao":{"type":"Cat", "group":None}},
    }
ECEC={
    "Ao":{"Ai":{"type":"BV_red", "group":None}, "Ar":{"type":"Cat", "group":None}},
    "Ai":{"Ao":{"type":"BV_ox", "group":None}, "X1":{"type":"Cat", "group":None}},
    "X1":{"Ai":{"type":"Cat", "group":None}, "Ar":{"type":"BV_red", "group":None}},
    "Ar":{"X1":{"type":"BV_ox", "group":None}, "Ao":{"type":"Cat", "group":None}}
}
EECC={
    "Ao":{"Ai":{"type":"BV_red", "group":None}, "X2":{"type":"Cat", "group":None}},
    "Ai":{"Ao":{"type":"BV_ox", "group":None}, "Ar":{"type":"BV_red", "group":None}},
    "X2":{"Ao":{"type":"Cat", "group":None}, "Ar":{"type":"Cat", "group":None}},
    "Ar":{"X2":{"type":"Cat", "group":None}, "Ai":{"type":"BV_ox", "group":None}}
}

mechanisms=[EECR, EEEEE, EEC, ECEC, EECC]
name_list=["EECR", "EEEEE", "EEC", "ECEC", "EECC"]
subtracted_keys=["ArRr", "ArRr", "Ar", "Ar", "Ar"]
class_list=[]
for i in range(0, len(mechanisms)):
    simulation_options["linkage_dict"]=mechanisms[i]
    simulation_options["subtracted_species"]=subtracted_keys[i]
    class_list.append(square_scheme(param_list, simulation_options, other_values, param_bounds))
farad_param_values=[
    {
    'k0_1':None, #AoRo->AoRr, AoRr->AoRo
    'E0_1':None, #AoRo->AoRr, AoRr->AoRo
    'alpha_1':None, #AoRo->AoRr, AoRr->AoRo
    'k_1':None, #AoRo->ArRo
    'k_2':None, #AoRr->AiRo
    'k_3':None, #AoRr->ArRr
    'k_4':None, #AiRo->AoRr
    'k0_2':None, #AiRo->AiRr, AiRr->AiRo
    'E0_2':None, #AiRo->AiRr, AiRr->AiRo
    'alpha_2':None, #AiRo->AiRr, AiRr->AiRo
    'k_5':None, #AiRr->ArRo
    'k_6':None, #ArRo->AoRo
    'k_7':None, #ArRo->AiRr
    'k0_3':None, #ArRo->ArRr, ArRr->ArRo
    'E0_3':None, #ArRo->ArRr, ArRr->ArRo
    'alpha_3':None, #ArRo->ArRr, ArRr->ArRo
    'k_8':None, #ArRr->AoRr
    },
    {
    'k0_1':None, #AoRo->AoRr, AoRr->AoRo
    'E0_1':None, #AoRo->AoRr, AoRr->AoRo
    'alpha_1':None, #AoRo->AoRr, AoRr->AoRo
    'k0_2':None, #AoRr->AiRo, AiRo->AoRr
    'E0_2':None, #AoRr->AiRo, AiRo->AoRr
    'alpha_2':None, #AoRr->AiRo, AiRo->AoRr
    'k0_3':None, #AiRo->AiRr, AiRr->AiRo
    'E0_3':None, #AiRo->AiRr, AiRr->AiRo
    'alpha_3':None, #AiRo->AiRr, AiRr->AiRo
    'k0_4':None, #AiRr->ArRo, ArRo->AiRr
    'E0_4':None, #AiRr->ArRo, ArRo->AiRr
    'alpha_4':None, #AiRr->ArRo, ArRo->AiRr
    'k0_5':None, #ArRo->ArRr, ArRr->ArRo
    'E0_5':None, #ArRo->ArRr, ArRr->ArRo
    'alpha_5':None, #ArRo->ArRr, ArRr->ArRo
    },
    {
    'k0_1':None, #Ao->Ai, Ai->Ao
    'E0_1':None, #Ao->Ai, Ai->Ao
    'alpha_1':None, #Ao->Ai, Ai->Ao
    'k_1':None, #Ao->Ar
    'k0_2':None, #Ai->Ar, Ar->Ai
    'E0_2':None, #Ai->Ar, Ar->Ai
    'alpha_2':None, #Ai->Ar, Ar->AiEECC
    'k_2':None, #Ar->Ao
    },
    {
    'k0_1':None, #Ao->Ai, Ai->Ao
    'E0_1':None, #Ao->Ai, Ai->Ao
    'alpha_1':None, #Ao->Ai, Ai->Ao
    'k_1':None, #Ao->Ar
    'k_2':None, #Ai->X1
    'k_3':None, #X1->Ai
    'k0_2':None, #X1->Ar, Ar->X1
    'E0_2':None, #X1->Ar, Ar->X1
    'alpha_2':None, #X1->Ar, Ar->X1
    'k_4':None, #Ar->Ao
    },
    {
    'k0_1':None, #Ao->Ai, Ai->Ao
    'E0_1':None, #Ao->Ai, Ai->Ao
    'alpha_1':None, #Ao->Ai, Ai->Ao
    'k_1':None, #Ao->X2
    'k0_2':None, #Ai->Ar, Ar->Ai
    'E0_2':None, #Ai->Ar, Ar->Ai
    'alpha_2':None, #Ai->Ar, Ar->Ai
    'k_2':None, #X2->Ao
    'k_3':None, #X2->Ar
    'k_4':None, #Ar->X2
    }
]



xaxis="time" 
xaxis="potential"
mode="together"
mode="seperate"
if mode=="seperate":
    figure=multiplot(1, len(class_list), **{"harmonic_position":list(range(0, len(class_list))), "num_harmonics":7, "orientation":"portrait",  "plot_width":5, "row_spacing":2,"col_spacing":2, "plot_height":1})
    fig,ax=plt.subplots(1, len(class_list))

else:
    figure=multiplot(1, 1, **{"harmonic_position":0, "num_harmonics":7, "orientation":"portrait",  "plot_width":5, "row_spacing":2,"col_spacing":2, "plot_height":1})
    fig, ax=plt.subplots(1,1)
for j in range(0, len(class_list)):
    farad_params=farad_param_values[j]
    keys=list(farad_params.keys())
    param_vals=np.random.rand(len(keys))
    e0_counter=1
    e0_vals=[]
    #Comment this section out if you want to define value manually in the dictionary above
    for i in range(0, len(param_vals)):
        if "k" in keys[i]:
            param_vals[i]=class_list[j].un_normalise(param_vals[i], param_bounds["k_0"])
        elif "E0" in keys[i]:
            param_vals[i]=param_list["E_reverse"]-(e0_counter*0.15)
            e0_vals.append(param_vals[i])
            e0_counter+=1
        elif "alpha" in keys[i]:
            param_vals[i]=class_list[j].un_normalise(param_vals[i], param_bounds["alpha"])
        farad_params[keys[i]]=param_vals[i]
    ####################################################################################
    class_list[j].def_optim_list(keys)
    current=class_list[j].simulate(param_vals, [])*1e6
    h_class=harmonics(list(range(1, 8)), param_list["omega"], 0.25)
    if mode=="seperate":
       
        if xaxis=="potential":
            ax[j].plot(class_list[j].e_nondim(class_list[j].define_voltages())[5:], current[5:])
        elif xaxis=="time":
            ax[j].plot(class_list[j].t_nondim(class_list[j].time_vec)[5:], current[5:])
        arg_dict=dict(hanning=True, axes_list=figure.axes_dict["col{0}".format(j+1)], plot_func=abs, xlabel="time", ylabel="Current ($\\mu A$)")
        arg_dict[name_list[j]+"_time_series"]=current
        h_class.plot_harmonics(class_list[j].t_nondim(class_list[j].time_vec), **arg_dict)
        ax[j].set_xlabel(xaxis)
        ax[j].set_ylabel("Current ($\\mu A$)")
        ax[j].set_title(name_list[j])
    elif mode=="together":
        if xaxis=="potential":
            ax.plot(class_list[j].e_nondim(class_list[j].define_voltages())[5:], current[5:], label=name_list[j])
        elif xaxis=="time":
            ax.plot(class_list[j].t_nondim(class_list[j].time_vec)[5:], current[5:], label=name_list[j])
        ax.set_xlabel(xaxis)
        ax.set_ylabel("Current ($\\mu A$)")
        arg_dict=dict(hanning=True, axes_list=figure.axes_dict["col1"], plot_func=abs, xlabel="time", ylabel="Current ($\\mu A$)")
        arg_dict[name_list[j]+"_time_series"]=current
        h_class.plot_harmonics(class_list[j].t_nondim(class_list[j].time_vec), **arg_dict)
        ax.legend()
plt.show()

