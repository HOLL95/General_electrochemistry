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
from multiplotter import multiplot
import numpy as np
import pints
from scipy.signal import decimate
from pints.plot import trace
data_loc="/home/henryll/Documents/Experimental_data/Alice/Immobilised_Fc/GC-Green_(2023-10-10)/Fc"
#data_loc="/home/userfs/h/hll537/Documents/Experimental_data"
file_name="2023-10-10_EIS_GC-Green_Fc_240_1"
data=np.loadtxt(data_loc+"/"+file_name)

truncate=10
truncate_2=1
real=np.flip(data[truncate:-truncate_2,0])
imag=np.flip(data[truncate:-truncate_2,1])

frequencies=np.flip(data[truncate:-truncate_2,2])
file_name="2023-10-10_FTV_GC-Green_Fc_cv_"
current_data_file=np.loadtxt(data_loc+"/"+file_name+"current")
voltage_data_file=np.loadtxt(data_loc+"/"+file_name+"voltage")
dec_amount=8

volt_data=voltage_data_file[0::dec_amount, 1]

plot_dict={"current":current_data_file[0::dec_amount,1], "time":current_data_file[0::dec_amount,0], "potential":volt_data}





curr_dict=plot_dict
for key in curr_dict:
    curr_dict[key]=decimate(curr_dict[key], dec_amount)





DC_val=0
param_list={
        "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':0,
        "v":100e-3,  #    (frequency Hz)
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-11,
        "original_gamma":1e-11,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "Cfarad":0,
        "E0_mean":0,
        "E0_std": 0.025,
        "k0_shape":0.4,
        "k0_scale":75,
        "num_peaks":10,
        "cap_phase":0
}
print(param_list["E_start"], param_list["E_reverse"])
print(param_list)
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[300],
    "GH_quadrature":False,
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[1],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
    "C_sim":True,
    "EIS_Cf":"C",
    "EIS_Cdl":"CPE",
    "DC_pot":240e-3,
    "Rct_only":False,
}
DC_pot=240e-3
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,9,1)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[0.15, 0.35],
    "E0_mean":[0.15, 0.35],
    "E0_std":[0.001, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-7],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0.65,1],
    "k0_shape":[0,10],
    "k0_scale":[0,1e4],
    "phase":[-180, 180],
    "cap_phase":[-180,190]
}
import copy


laviron=Laviron_EIS(param_list, simulation_options, other_values, param_bounds)


#laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E_0", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic", "Cfarad"])
laviron.def_optim_list(["E0_mean", "E0_std", "k_0", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E_0",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
#laviron.def_optim_list(["E0_mean","E0_std",  "k0_scale","k0_shape", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl", "cpe_alpha_faradaic"])
banned_param={"cpe_alpha_cdl", "cpe_alpha_faradaic", "Cfarad"}
get_set=list(set(laviron.optim_list)-banned_param)
ramped_optim_list=[x for x in laviron.optim_list if x in get_set]#+["phase","cap_phase"]

free_params=dict(zip(["phase","cap_phase"], [0,0]))

#"E0_mean","E0_std","k0_scale","k0_shape"
spectra=np.column_stack((real, imag))
#EIS().bode(spectra, frequencies)
#plt.show()
fitting_frequencies=2*np.pi*frequencies
#EIS().nyquist(spectra, orthonormal=False)

"""sim_data=laviron.simulate([cdl_only[x] for x in laviron.optim_list], fitting_frequencies)
fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(spectra, frequencies, ax=ax, twinx=twinx, label="Data")
EIS().bode(sim_data, frequencies, ax=ax, twinx=twinx, label="Simulation")
ax.legend()
ax.set_title("C fit")
plt.show()

"""

#laviron.simulation_options["label"]="cmaes"
laviron.simulation_options["data_representation"]="bode"
data_to_fit=EIS().convert_to_bode(spectra)
#C

EIS_params1={'E_0': 0.23708843969139082-DC_val, 'k_0': 4.028523388682444, 'gamma': 7.779384163661676e-10, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#EIS_params1={'E_0': 0.23708843969139082-DC_val, 'k_0': 75, 'gamma': 2e-11, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#CPE

EIS_params2={'E_0': 0.3047451457126534-DC_val, 'k_0': 39.40663787158313, 'gamma': 1.0829517784499947e-10, 'Cdl': 8.7932554621096e-06, 'alpha': 0.5394294479538084, 'Ru': 80.76397847517714,"omega":8.794196510802587}
#CPEonlycdl
EIS_params2={'E_0': 0.48473683666709744, 'k_0': 748.8829198367846, 'gamma': 1.8360361398144898e-08, 'Cdl': 6.160194675577306e-05, 'alpha': 0.44425990992099207, 'Ru': 62.57692789363438, 'cpe_alpha_cdl': 0.4804715757637764, 'cpe_alpha_faradaic': 0.37647280608421163}
EIS_params2={"E_0":0.24,'k_0': 2.7264466013265967, 'gamma': 6.312795515810845e-10, 'Cdl': 1.4793284277931158e-05, 'alpha': 0.47125898955325585, 'Ru': 75.59649718628145, 'cpe_alpha_cdl': 0.690583155657475, 'cpe_alpha_faradaic': 0.9740515699839547}

#CPEboth
EIS_params2a={'E_0': 0.45787769024326985, 'k_0': 1.9399798351038797, 'gamma': 3.609270985983326e-08, 'Cdl': 8.793255414148275e-06, 'alpha': 0.5243005057452185, 'Ru': 80.76397850768745, 'cpe_alpha_cdl': 0.7557057511156808, 'cpe_alpha_faradaic': 0.8471965362771445}


#Cfarad
EIS_params3={'E_0': 0.4991982785842859, 'k_0': 1545.3528723742425, 'gamma': 4.0289036924436e-11, 'Cdl': 8.793255423745104e-06, 'alpha': 0.6117535872755911, 'Ru': 80.76397862233247, 'cpe_alpha_cdl': 0.7557057511591937, 'cpe_alpha_faradaic': 0.8471965348601926, 'Cfarad': 4.98359157906545e-05}
EIS_params3={"E_0":0.24, 'k_0': 42.3589742547644, 'gamma': 5.839640664055033e-11, 'Cdl': 8.793255424507551e-06, 'alpha': 0.3954483294530293, 'Ru': 80.76397847166463, 'cpe_alpha_cdl': 0.7557057509996593, 'cpe_alpha_faradaic': 0.8471965357461846, 'Cfarad': 4.983591555543026e-05}

#EIS_params3={'E_0': 0.2161051668499098-DC_val, 'k_0': 106.6602436491309, 'gamma': 2.5360979030661595e-11, 'Cdl': 8.751614540745486e-06, 'alpha': 0.47965820103670564, 'Ru': 80.92159716231082,"omega":8.794196510802587}
#No CPE
EIS_params_4={'E_0': 0.2014214483444881-DC_val, 'k0_scale': 1.0950956335756536, 'k0_shape': 1.043401547065882, 'gamma': 1.4645920920242938e-09, 'Cdl': 7.945475589121264e-06, 'alpha': 0.359816590101354, 'Ru': 81.69086207153816, 'cpe_alpha_cdl': 0.768033972041215, 'cpe_alpha_faradaic': 0.9119951540999298, 'phase': -0.20113351781378697,"omega":8.794196510802587} 
EIS_params_4={'E_0': 0.19463469159444804-DC_val, 'k0_shape': 2.392919243565184, 'k0_scale': 0.2064967806796385, 'gamma': 3.924680690469505e-09, 'Cdl': 9.90894008867702e-07, 'alpha': 0.5883138432002848, 'Ru': 90.57829277724034, 'cpe_alpha_cdl': 0.5659794458625567, 'cpe_alpha_faradaic': 0.14317733968270077}
EIS_params_4={'E_0': 0.2226369985285386, 'k0_shape': 2.392919250556589, 'k0_scale': 0.24611646420768749, 'gamma': 2.189348988012857e-09, 'Cdl': 9.908940071648933e-07, 'alpha': 0.5587056057336539, 'Ru': 90.57829286855551, 'cpe_alpha_cdl': 0.8941354153164397, 'cpe_alpha_faradaic': 0.8163866289424468}


#CPE but no extra phase
EIS_params_5={'E_0': 0.15993573511018355-DC_val, 'k0_shape': 1.0143738017899193, 'k0_scale': 0.4719318521743649, 'gamma': 5.364902729622656e-09, 'Cdl': 7.636444325985663e-06, 'alpha': 0.364166232234114, 'Ru': 81.92130789951946, 'cpe_alpha_cdl': 0.7725503185958909, 'cpe_alpha_faradaic': 0.008866416527890587}
EIS_params_5={'E_0': 0.19066872485338204-DC_val, 'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}
#E0_mean with CPE
EIS_params_6={'E0_mean': 0.3499999999999999-DC_val, 'E0_std': 0.045854752108924646, 'k_0': 0.9166388879743895, 'gamma': 5.405583319246063e-09, 'Cdl': 9.23395426422013e-06, 'alpha': 0.6499999999999999, 'Ru': 80.37330196288727, 'cpe_alpha_cdl': 0.7495664487939422, 'cpe_alpha_faradaic': 0.1477882191366903}
#[0.2591910307724134, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, ramped_h_class.input_frequency, 0, 0, 0.5999998004431891]
EIS_params_6={'E0_mean': 0.2591910307724134, 'E0_std': 0.0674086382052161,  'k_0':  177.04633092062943, 'gamma': 6.037508022415195e-11, 'Cdl': 9.23395426422013e-06, 'alpha': 0.6, 'Ru': 88.31972285297374,'cpe_alpha_cdl': 0.7495664487939422, 'cpe_alpha_faradaic': 0.1477882191366903}
#E0_mean with C
EIS_params_7={'E0_mean': 0.35-DC_val, 'E0_std': 0.059192130273338424, 'k_0': 1.678514486845095, 'gamma': 4.586111119553072e-09, 'Cdl': 1.3948590417154984e-06, 'alpha': 0.65, 'Ru': 96.39939088176911, 'cpe_alpha_cdl': 0.6612954622562719, 'cpe_alpha_faradaic': 0.9995161877220936}




#EIS_params_5={'E_0': 0.17367485939633537, 'k0_shape': 2.4360726160882744, 'k0_scale': 0.15942388883368433, 'gamma': 8.019805323074907e-09, 'Cdl': 9.844551737915205e-07, 'alpha': 0.6465984705927521, 'Ru': 91.5686733543809, 'cpe_alpha_cdl': 0.9169107302998178, 'cpe_alpha_faradaic': 0.9863374008502952, 'phase': -1.9689345255496846}
#Both no C
EIS_params_9={'E0_mean': 0.2552237543929984-DC_val, 'E0_std': 0.0010014967867590788, 'k0_shape': 2.4360714665636465, 'k0_scale': 0.22242584355076356, 'gamma': 2.2858275700178405e-09, 'Cdl': 9.844551474481184e-07, 'alpha': 0.35822572276185904, 'Ru': 91.56868145656738, 'cpe_alpha_cdl': 0.5070420269200153, 'cpe_alpha_faradaic': 0.1970959976913189, 'phase': -1.9689465346727104}
#Both CPE
EIS_params_10={'E0_mean': 0.24015377746572225, 'E0_std': 0.0021373301993834804, 'k0_shape': 1.0429509932174068, 'k0_scale': 1.7509356211175566, 'gamma': 8.738837903502286e-10, 'Cdl': 7.947284126522816e-06, 'alpha': 0.499215202883645, 'Ru': 81.68491214990672, 'cpe_alpha_cdl': 0.7680050975512425, "cpe_alpha_faradaic":1}
#k0_distribution with hard lower limits 
{'E_0': 0.2241032166519744, 'k0_shape': 0.36042285494223125, 'k0_scale': 30.0, 'gamma': 3.81680321550312e-10, 'Cdl': 1.239479377634673e-06, 'alpha': 0.6499999999999999, 'Ru': 94.97839502706194, 'cpe_alpha_cdl': 0.7813622563224152, 'cpe_alpha_faradaic': 0.3162880458183584}

#######################HERE######################
param_dict=EIS_params_6
#################################################
cdl_vals=[9e-7, 2e-6,5e-6, 9e-06]
alpha_vals=[0.7, 0.75, 0.8]
fig, axis=plt.subplots(2, 2)
for i in range(0, len(cdl_vals)):
    param_dict=copy.deepcopy(EIS_params_6)
    ax=axis[i//2, i%2]
    twinx=ax.twinx()
    param_dict["Cdl"]=cdl_vals[i]
    EIS().bode(np.column_stack((real, imag)),frequencies,ax=ax, twinx=twinx, label="Data", compact_labels=True)
    ax.set_title("$Q_{C_{dl}}=%.1e$" %cdl_vals[i])
    for j in range(0, len(alpha_vals)):
        param_dict["cpe_alpha_cdl"]=alpha_vals[j]
        circ_params=[param_dict[x] for x in laviron.optim_list]

        sim_vals=laviron.simulate(circ_params, fitting_frequencies)
        
        if j==1:
            alpha=1
        else:
            alpha=0.5
        EIS().bode(sim_vals,frequencies,ax=ax, twinx=twinx, data_type="phase_mag", compact_labels=True, alpha=alpha, label="$\\alpha_{CPE}=%.2f$"%alpha_vals[j])
axis[0,0].legend()
plt.show()
fig.savefig("Alpha_scans.png", dpi=500)