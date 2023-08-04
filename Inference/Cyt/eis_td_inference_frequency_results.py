
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
from EIS_TD import EIS_TD
from heuristic_class import Laviron_EIS
import numpy as np
import time
import pints
from pints.plot import trace
data_loc="/home/henney/Documents/Oxford/Experimental_data/Henry/7_6_23/Text_files/DCV_EIS_text"
data_file="EIS_modified.txt"

data=np.loadtxt(data_loc+"/"+data_file, skiprows=10)    

fitting_data=np.column_stack((np.flip(data[:,0]), np.flip(data[:,1])))
DC_val=0
frequencies=np.flip(data[:,2])
param_list={
       "E_0":DC_val,
        'E_start':  DC_val-10e-3, #(starting dc voltage - V)
        'E_reverse':DC_val+10e-3,
        'omega':0,  #    (frequency Hz)
        "v":100e-3,
        'd_E': 10e-3,   #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl': 1e-5, #(capacitance parameters)
        'CdlE1': 0,
        'CdlE2': 0,
        "CdlE3":0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 100, #(reaction rate s-1)
        'alpha': 0.55,
        "sampling_freq":1/(2**8),
        "cpe_alpha_faradaic":1,
        "cpe_alpha_cdl":1,
        "phase":0,
        "E0_mean":DC_val,
        "E0_std":0.02,
        "cap_phase":0,
        "num_peaks":3
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]

simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[2],
    "test": False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "optim_list":[],
 
    "data_representation":"bode",
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(1,2)),
    "bounds_val":20000,
}
param_bounds={
    'E_0':[-0.1, 0.1],
    'E0_mean':[-0.4, -0.1],
    'E0_std':[1e-3, 0.1],
    'omega':[0.95*param_list['omega'],1.05*param_list['omega']],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],1e-8],
    'k_0': [1e-9, 1e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0,1],
    "phase":[0, 2*math.pi],
    "cap_phase":[0, 2*math.pi],
}

td=EIS_TD(param_list, simulation_options, other_values, param_bounds)
orig_params=["E_0","gamma","k_0",  "Cdl", "alpha", "Ru", "phase", "cap_phase"]
td.def_optim_list(orig_params)
cdl_and_cap=[0.10487903661126297, 0.40342620734920664, 0.002075749105321917, 0.3009251284017961, 0.060558151585271586, 0.9991871631368601, 0.13940419476545704, 0.6052332550768639, 0.12974339152505715, 0.9832979181892831, 0.12431734835336218, 0.23812962824229134, 0.11544882374999965, 0.6762446821399406, 0.10579337320066806, 0.8414774438800677, 0.10147454633082653, 0.9906443412212456, 0.09331830337043563, 0.5309916851528229, 0.08830871073399182, 0.9580881741026729, 0.086143264666799, 0.1948997264442598, 0.08507246109928035, 0.5133648283379861, 0.08395038933831511, 0.0014495605348126318, 0.08602844758983896, 0.82494111221458, 0.08521482119660415, 0.9957271042332291, 0.07736493979732485, 0.4850280265914546, 0.06628811694169465, 0.922746208222359, 0.05309384795610281, 0.9214525254386359, 0.0398161839754897, 0.7677127050134143, 0.028088332860913685, 0.5129289308204024, 0.02135545127877325, 0.2586238473230888, 0.0165562707766563, 0.7031301365654345, 0.014796773064802322, 0.19007062551169662, 0.014376539143019324, 0.9569628420277217, 0.014566737122198982, 0.9722786691520988, 0.015184227866228127, 0.3996837056391702, 0.014648213369974536, 0.824763964197522, 0.014468688720447815, 0.37339254264735566, 0.013510739520238381, 0.3612719506637174, 0.012182194235140872, 0.30793467685869613, 0.011353592701443743, 0.3039842291724558, 0.010199463157714962, 0.08789400212269982, 0.009197115225967056, 0.27875341275289744, 0.00842022795607674, 0.706259169550062, 0.007437997702330093, 0.48054418719370406, 0.006704768363041972, 0.8141564152734311, 0.006063930289110556, 0.6205260868215499, 0.005472373567335361, 0.6913876221828467, 0.005001565834974639, 0.9615827012126724, 0.004515601970155641, 0.36846148733856277, 0.00413451027625518, 0.478446284734083, 0.0037843865802153595, 0.2381414171936448, 0.0035140945002042076, 0.3241315944475005, 0.3630102050301122, 0.4418598788448017, 0.26611562476624084, 0.5507431084003114, 0.36822461667434747, 0.9300761371444194, 0.00298422322358273, 0.31892488106919414, 0.0033626487021251933, 0.9999864627042889, 0.6011881533062242, 0.12087686421737767, 0.04643302567204272, 0.03468957252911603, 0.002069094960605531, 0.4039027307472205, 0.4537298987590881, 0.6186689231942744, 0.9999996187843311, 0.308494802686553, 0.6095259545125571, 0.3370509630566414, 0.697125228710645, 0.9279430060490236, 0.8212980967818672, 0.431444994544704, 0.431678784287003, 0.06882830242976944, 0.4544301789568981, 0.49330042874081886, 0.46554176667304237, 0.22077069916003966, 0.4417711418161031, 0.9997050579595013, 0.9973292709750347, 0.9468210186404324, 0.2651361862913255, 0.49179632155749786, 0.018207800600027267, 0.5340196035607356, 7.819435723494565, 0.019994722148184763]
cap=[0.7386140888665734, 0.002257335183626449, 0.29703073721809703, 0.10484576320105299, 0.6036560754345592, 2.1262894092112462e-07, 0.00533126329978909, 0.15138479419834117, 0.8258159270325642, 0.6070122895720534, 0.8058614488783049, 0.40012927671585086, 0.13123612822774655, 0.7825776798143065, 0.8162004018442157, 0.7155843541859761, 0.5751174078999108, 0.7377995054792947, 0.9018045629281848, 0.24662458334112597, 0.3762382866594526, 0.726701100568321, 0.5549112089512134, 0.9629194334234028, 0.6884793847658459, 0.42122505653367026, 0.7252349567355048, 0.7967882445309358, 0.9992233326713166, 0.2082675601786672, 0.3124371077174304, 0.20467519991384897, 0.5076927109413516, 0.552947701032845, 0.3397862704710217, 0.9082075080322942, 0.5092989340169949, 0.4255163692639238, 0.47614849882248533, 0.7639855808609599, 0.9499213675464552, 0.831838125198987, 0.8281032457358841, 0.790406203841252, 0.19697949609313614, 0.9165667718890956, 0.38170499076287395, 0.2527053265381533, 0.6694020578623422, 0.8086742364122339, 0.3781342865903708, 0.9066709822665606, 0.0046323557938735774, 0.00014967364404704078, 0.500380758088071, 0.02487330233937352, 0.3048526958133424, 0.6254958684839216, 0.1825691973336257, 0.28418420833870117, 0.34296169909443136, 0.43740129387691745, 0.9782133911367208, 0.4140080677156669, 0.4487568018741057, 0.006325320316460821, 0.32382954527749697, 0.23820170858292358, 5.848733308628149, 0.030139219407793554]
plt.plot(cap[6:-2:2])
plt.show()
vals={'R0': 106.15130275124821, 'R1': 229.75209691347146, 'C1': 3.834939019943294e-06, 'Q2': 0.00024417332992628603, 'alpha2': 0.7912685395546909}

dispersion_params=["cap_phase"]
all_dispersion_params=["" for x in range(0, len(dispersion_params)*len(frequencies))]
def flatten(l):
    return [item for sublist in l for item in sublist]

for j in range(0, len(frequencies)):
    for i in range(0, len(dispersion_params)):
        all_dispersion_params[(len(dispersion_params)*j)+i]="{0}_{1}".format(dispersion_params[i], j)
        

non_disped_params=list(set(orig_params)-set(dispersion_params))
all_parameters=non_disped_params+all_dispersion_params

print(len(cdl_and_cap), len(all_parameters))
td.simulation_options["frequency_dispersion"]=dispersion_params
td.def_optim_list(all_parameters)
td.simulation_options["label"]="cmaes"
#test_vals=td.change_norm_group(cdl_and_cap[:-2], "un_norm")
test_vals=td.simulate(cap, frequencies)

fig, ax=plt.subplots()
twinx=ax.twinx()
EIS().bode(test_vals, frequencies, ax=ax, twinx=twinx, data_type="phase_mag")
EIS().bode(fitting_data, frequencies, ax=ax, twinx=twinx)

plt.show()


