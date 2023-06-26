import os
import sys
import copy
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
from EIS_optimiser import EIS_optimiser, EIS_genetics
from heuristic_class import Laviron_EIS
from circuit_drawer import circuit_artist
from MCMC_plotting import MCMC_plotting
from pandas import read_csv
rows=5
cols=4
mplot=MCMC_plotting()

colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
mode="error"
circuit_1={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}
norm_potential=0.001
#RTF=(self.R*self.T)/((self.F**2))
circuit_2={ "z2":"R0", "z3":{"p_1":["R3","C3" ], "p_2":"C2"},}
circuit_3={ "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
mark_circuit={"z1":{"p_1":"R1", "p_2":"C1"}, "z2":"R0", "z3":{"p_1":["R3",("Q1", "alpha1") ], "p_2":"C2"},}
F=96485.3321
R=8.3145
T=298
RT=R*T
FRT=F/(R*T)
k0=0.1
e0=0.001
Cdl=2e-5
alpha=0.55
gamma=1e-10
area=0.07
dc_pot=0
Ru=100
lav_params={"k_0":k0, "Ru":Ru, "Cdl":Cdl, "gamma":gamma, "E_0":e0, "alpha":alpha, "area":area, "DC_pot":dc_pot}
ratio=np.exp(FRT*(e0-dc_pot))
gamma_red=gamma/(ratio+1)
gamma_ox=gamma-gamma_red
exp=np.exp
norm_E=dc_pot-e0
fitted_params=["Ru", "alpha", "k_0", "Cdl"]

circs=[mark_circuit]
for plot in ["both"]:
    fig, axis=plt.subplots(rows, cols, constrained_layout=True)
    if "both" in plot:
        twinxis=[[axis[i, j].twinx() for j in range(0, cols) ] for i in range(0, rows)]
    for z in range(0, len(circs)):
            
            current_file=np.load("BV_param_scans_best_fits_circuit_{0}_a.npy".format(z), allow_pickle=True).item()
            freq=current_file["data"]["freq"]
            keys=list(current_file.keys())
            
            sub_keys=list(current_file["results"].keys())
            titles=mplot.get_titles(sub_keys, units=False)
            units=mplot.get_units(sub_keys)
            for i in range(0, len(sub_keys)):
                    copy_params=copy.deepcopy(lav_params)
                        #if plot=="both":
                        #        twinx[i,z].set_axis_off()
                    key=sub_keys[i]
                    param_vals=list(current_file["data"][key].keys())
                    for j in range(0, len(param_vals)):
                            ax=axis[i,j]
                            if "both" in plot:
                                twinx=twinxis[i][j]
                            else:
                                twinx=None
                            val=param_vals[j]
                            fitting_data=current_file["data"][key][val]
                            phase=fitting_data[:,0]
                            mag=np.log10(fitting_data[:,1])
                            fitted_sim_vals=current_file["results"][key][val+"_results"]
                            print(key, val)
                            copy_params[key]=float(val)
                            sim=Laviron_EIS().clean_simulate(params=copy_params, frequencies=freq, EIS_Cdl="C", EIS_Cf="C")
                            #print("="*30, circs[z], val)
                            #for keyz in fitted_sim_vals:
                            #    print(keyz, fitted_sim_vals[keyz])
                            if mode=="bode" or plot=="both":
                                    if z==0:
                                            EIS().bode(fitting_data, freq, data_type="phase_mag", ax=ax, twinx=twinx, compact_labels=True, data_is_log=False, lw=2, alpha=0.65,type=plot)
                                    EIS().bode(sim, freq, ax=ax, twinx=twinx, compact_labels=True, type=plot)
                                    ax.set_title("{0}={1} {2}".format(titles[i], val, units[i]))
                       
plt.show()