import numpy as np
letter_list=list("abcdefghi")
DC_val=0
psv_vals=[0.2591910307724134, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509, -0.0004999993064740369, 2.5653514370132974e-05, 6.037508022415195e-11, 9.015057685643711, 5.58768403688611, 4.964330246307874, 0.5999998004431891]
psv_params=["E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"]
rftv1_vals=[0.2591910307724134, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, 8.82, 0, 0, 0.5999998004431891]
rftv1_params=["E0_mean", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"]
Laviron={'E_0': 0.23708843969139082-DC_val, 'k_0': 4.028523388682444, 'gamma': 7.779384163661676e-10, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825,  'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
CdlCPELaviron={"E_0":0.24,'k_0': 2.7264466013265967, 'gamma': 6.312795515810845e-10, 'Cdl': 1.4793284277931158e-05, 'alpha': 0.47125898955325585, 'Ru': 75.59649718628145, 'cpe_alpha_cdl': 0.690583155657475, 'cpe_alpha_faradaic': 0.9740515699839547}
CPECPELAviron={'E_0': 0.45787769024326985, 'k_0': 1.9399798351038797, 'gamma': 3.609270985983326e-08, 'Cdl': 8.793255414148275e-06, 'alpha': 0.5243005057452185, 'Ru': 80.76397850768745, 'cpe_alpha_cdl': 0.7557057511156808, 'cpe_alpha_faradaic': 0.8471965362771445}
CfaradLaviron={"E_0":0.24, 'k_0': 42.3589742547644, 'gamma': 5.839640664055033e-11, 'Cdl': 8.793255424507551e-06, 'alpha': 0.3954483294530293, 'Ru': 80.76397847166463, 'cpe_alpha_cdl': 0.7557057509996593, 'cpe_alpha_faradaic': 0.8471965357461846, 'Cfarad': 4.983591555543026e-05}
k0_dist={'E_0': 0.19066872485338204-DC_val, 'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}
e0_dist={'E0_mean': 0.3499999999999999-DC_val, 'E0_std': 0.045854752108924646, 'k_0': 0.9166388879743895, 'gamma': 5.405583319246063e-09, 'Cdl': 9.23395426422013e-06, 'alpha': 0.6499999999999999, 'Ru': 80.37330196288727, 'cpe_alpha_cdl': 0.7495664487939422, 'cpe_alpha_faradaic': 0.1477882191366903}
double_dist={'E0_mean': 0.24015377746572225, 'E0_std': 0.0021373301993834804, 'k0_shape': 1.0429509932174068, 'k0_scale': 1.7509356211175566, 'gamma': 8.738837903502286e-10, 'Cdl': 7.947284126522816e-06, 'alpha': 0.499215202883645, 'Ru': 81.68491214990672, 'cpe_alpha_cdl': 0.7680050975512425, "cpe_alpha_faradaic":1}
double_dist_c_only={'E0_mean': 0.2552237543929984-DC_val, 'E0_std': 0.0010014967867590788, 'k0_shape': 2.4360714665636465, 'k0_scale': 0.22242584355076356, 'gamma': 2.2858275700178405e-09, 'Cdl': 9.844551474481184e-07, 'alpha': 0.35822572276185904, 'Ru': 91.56868145656738, 'cpe_alpha_faradaic': 0.1970959976913189, 'phase': -1.9689465346727104}



mega_param_dict=[{"params":dict(zip(psv_params, psv_vals)),
                "figure":r"\ref{fig:td-best-fits}(A)",
                "model":"Differential equation",
                "data":"PSV",
                },
                {"params":dict(zip(rftv1_params, rftv1_vals)),
                "figure":r"\ref{fig:td-best-fits}(B)",
                "model":"Differential equation",
                "data":"rFTACV",
                },
                {"params":dict(zip(rftv1_params, rftv1_vals)),
                "figure":r"\ref{fig:td-best-fits}(C)",
                "model":"Differential equation",
                "data":"rFTACV",
                },
                {"params":Laviron,
                "figure":r"\ref{fig:PureLaviron}",
                "model":"Laviron circuit",
                "data":"EIS",
                },
                {"params":CdlCPELaviron,
                "figure":r"\ref{fig:CPE-CDL}",
                "model":"Modified Laviron circuit",
                "data":"EIS",
                },
                {"params":CPECPELAviron,
                "figure":r"\ref{fig:CPE-Both}",
                "model":"Modified Laviron circuit",
                "data":"EIS",
                },
                 {"params":CfaradLaviron,
                "figure":r"\ref{fig:Cfarad}",
                "model":"Modified Laviron circuit",
                "data":"EIS",
                },
                  {"params":e0_dist,
                "figure":r"\ref{fig:EIS_E0_disp}",
                "model":"Ladder network",
                "data":"EIS",
                },
                  {"params":k0_dist,
                "figure":r"\ref{fig:EIS_k0_disp}",
                "model":"Ladder network",
                "data":"EIS",
                },
                  {"params":double_dist,
                "figure":r"\ref{fig:EIS_k0_e0_disp}",
                "model":"Ladder network",
                "data":"EIS",
                },
                  {"params":double_dist_c_only,
                "figure":r"\ref{fig:EIS_k0_e0_disp_noCPE}",
                "model":"Ladder network",
                "data":"EIS",
                },
]


from math import log10, floor
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def format_vals(value):
    absval=abs(value)
    if absval>10:
        return "%.1f"%value
    elif absval==0:
        return "0"
    elif absval>0.01:
        return "%.2f"%value
    else:
        return "%.2e"%value
        

counter=-1
with open("table.tex", "r") as read_file:
    with open("edited_table.tex", "w") as write_file:
        for line in read_file:
            if line[0]=="\\" or line[0]==" ":
                write_file.write(line)
                continue
            else:
                split_line=line.split("&")
                if len(split_line)>1:
                    counter+=1
                    
                    print(counter, mega_param_dict[counter]["figure"])
                    for i in range(0,len(letter_list)):
                        
                        letter=letter_list[i]
                       
                        loc=[x for x in range(0, len(split_line[i])) if split_line[i][x]==letter]
                        #print(split_line[i], loc[0])
                        if len(loc)>0:
                            loc=loc[0]
                            edit_line=list(split_line[i])
                            exp=mega_param_dict[counter]
                            params=exp["params"]
                            if letter=="a":
                                edit_line[loc]=exp["data"]
                            elif letter=="b":
                                edit_line[loc]=exp["figure"]
                            elif letter=="c":
                                edit_line[loc]=exp["model"]
                            elif letter=="d":
                                if "E_0" in params.keys():
                                    edit_line[loc]=format_vals(exp["params"]["E_0"]*1e3)
                                else:
                                    edit_line[loc]="$\\mu$:%s, $\\sigma$:%s"%(format_vals(exp["params"]["E0_mean"]*1e3), format_vals(exp["params"]["E0_std"]*1e3))
                            elif letter=="e":
                                if "k_0" in params.keys():
                                     edit_line[loc]=format_vals(exp["params"]["k_0"])+" s$^{-1}$"
                                else:
                                    edit_line[loc]="$\\log(\\mu)$:%s, $\\log(\\sigma)$:%s"%(format_vals(exp["params"]["k0_scale"]), format_vals(exp["params"]["k0_shape"]))
                            elif letter=="f":
                                edit_line[loc]=format_vals(exp["params"]["gamma"])
                            elif letter=="g":
                                #if "CdlE1" in params.keys():
                                #    z=[format_vals(exp["params"][x]) for x in ["Cdl", "CdlE1","CdlE2","CdlE3"]]
                                #    edit_line[loc]="C_{dl}:%s F cm$^{-2}$, $E_1$:%s, $E_2$:%s, $E_3$:%s"%(z[0], z[1], z[2], z[3])
                                if "cpe_alpha_cdl" in params.keys():
                                    edit_line[loc]="$Q_{C_{dl}}$:%s, $\\alpha_{CPE}$:%s"%(format_vals(exp["params"]["Cdl"]/0.07), format_vals(exp["params"]["cpe_alpha_cdl"]))
                                else:
                                    edit_line[loc]=format_vals(exp["params"]["Cdl"]/0.07) + " F cm$^{-2}$"

                            elif letter=="h":
                                edit_line[loc]=format_vals(exp["params"]["alpha"])
                            elif letter=="i":
                                edit_line[loc]=format_vals(exp["params"]["Ru"])
                            if letter!="i":
                                edit_line[loc+1]+="&"
                            #print(edit_line)
                        #    split_line=
                        write_file.write(("").join(edit_line))

                    