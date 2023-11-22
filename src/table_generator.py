import os
from decimal import Decimal
import pickle
import numpy as np
unit_dict={
    "E_0": "V",
    'E_start': "V", #(starting dc voltage - V)
    'E_reverse': "V",
    'omega':"Hz",#8.88480830076,  #    (frequency Hz)
    'd_E': "V",   #(ac voltage amplitude - V) freq_range[j],#
    'v': '$V s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm^{-2}$',
    'k_0': 's^{-1}$', #(reaction rate s-1)
    'alpha': "",
    'E0_skew':"",
    "E0_mean":"V",
    "E0_std": "V",
    "k0_shape":"",
    "sampling_freq":"$s^{-1}$",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"rads",
    'phase' : "rads",
    "alpha_mean": "",
    "alpha_std": "",
    "":"",
    "noise":"",
    "error":"$\\mu A$",
}
fancy_names={
    "E_0": '$E^0$',
    'E_start': '$E_{start}$', #(starting dc voltage - V)
    'E_reverse': '$E_{reverse}$',
    'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
    'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "v",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Ru",  #     (uncompensated resistance ohms)
    'Cdl': "$C_{dl}$", #(capacitance parameters)
    'CdlE1': "$C_{dlE1}$",#0.000653657774506,
    'CdlE2': "$C_{dlE2}$",#0.000245772700637,
    'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
    'gamma': '$\\Gamma',
    'E0_skew':"$E^0$ skew",
    'k_0': '$k_0', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "cap_phase":"C$_{dl}$ phase",
    "k0_shape":"$\\log(k^0) \\sigma$",
    "k0_scale":"$\\log(k^0) \\mu$",
    "alpha_mean": "$\\alpha\\mu$",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "Phase",
    "sampling_freq":"Sampling rate",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"RMSE",
}

"(-6.490E-2 ) (0.0239) (3.858e-11)"
optim_list=["","E_0","k_0","Ru","Cdl","gamma", "alpha"]
ramped_vals1=[0.2591910307724134, 0.0674086382052161, 177.04633092062943, 88.31972285297374, 0.000342081409583126, 0.02292512550909509*0, -0.0004999993064740369*0, 2.5653514370132974e-05*0, 6.037508022415195e-11, 8.794196510802587, 0, 0, 0.5999998004431891]
ramped_vals2=[0.2515085054963522, 0.0637810584632682, 62.915075289229755, 109.99988420501067, 6.334048572909808e-11, 8.799273223827607,0.7961993346010553, 0.6,0.000342081409583126]
list1=["E_0", "E0_std", "k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"]
list2=["E_0", "E0_std","k_0","Ru","gamma","omega","phase", "alpha", "Cdl"]
EIS_params1={'E_0': 0.23708843969139082, 'k_0': 4.028523388682444, 'gamma': 7.779384163661676e-10, 'Cdl': 1.4936235822043384e-06, 'alpha': 0.4643410476326257, 'Ru': 97.73001050950825, 'cpe_alpha_cdl': 0.8931193741640449, 'cpe_alpha_faradaic': 0.8522148375036664, "omega":8.794196510802587}
#CPE
EIS_params2={'E_0': 0.3047451457126534, 'k_0': 39.40663787158313, 'gamma': 1.0829517784499947e-10, 'Cdl': 8.7932554621096e-06, 'alpha': 0.5394294479538084, 'Ru': 80.76397847517714,"omega":8.794196510802587}
#Cfarad
EIS_params3={'E_0': 0.2161051668499098, 'k_0': 106.6602436491309, 'gamma': 2.5360979030661595e-11, 'Cdl': 8.751614540745486e-06, 'alpha': 0.47965820103670564, 'Ru': 80.92159716231082,"omega":8.794196510802587}

EIS_params_4={'E_0': 0.2014214483444881, 'k_0': 1.0950956335756536, 'k0_shape': 1.043401547065882, 'gamma': 1.4645920920242938e-09, 'Cdl': 7.945475589121264e-06, 'alpha': 0.359816590101354, 'Ru': 81.69086207153816, 'cpe_alpha_cdl': 0.768033972041215, 'cpe_alpha_faradaic': 0.9119951540999298, 'phase': -0.20113351781378697,"omega":8.794196510802587} 

massive_dict_list=[dict(zip(list1, ramped_vals1)),dict(zip(list2, ramped_vals2)), EIS_params1, EIS_params2, EIS_params3, EIS_params_4]


name_list=[fancy_names[x] for x in optim_list]
parameter_orientation="row"
param_num=len(name_list)+1

values=[[element[x] for x in optim_list[1:]] for element in massive_dict_list]
names=["PSV", "Wiggled", "Laviron C", "Laviron CPE", "Laviron Rct only", "Ladder network"]
title_num=len(names)+1
table_file=open("image_tex_edited.tex", "w")
parameter_orientation="row"

#names=my_list[0]
if parameter_orientation=="row":
    f =open("image_tex_test.tex", "r")
    table_title="\\multicolumn{"+str(param_num-1)+"}{|c|}{Parameter values}\\\\ \n"
    table_control1="\\begin{tabular}{|"+(("c|"*param_num))+"}\n"
    table_control2="\\begin{tabular}{|"+(("p{3cm}|"*param_num))+"}\n"
    value_rows=[]
    row_1=""
    for i in range(0, len(name_list)):
        if unit_dict[optim_list[i]]!="":
            unit_dict[optim_list[i]]=" ("+unit_dict[optim_list[i]]+")"
        if i ==len(values[0]):
            row_1=row_1+name_list[i]+unit_dict[optim_list[i]] +"\\\\\n"
        else:
            row_1=row_1+(name_list[i]+unit_dict[optim_list[i]]+" & ")
    value_rows.append(row_1)
    for i in range(0, len(names)):
        row_n=""
        row_n=row_n+(names[i]+ " & ")
        for j in range(0, len(values[0])):
            if j ==len(values[0])-1:
                end_str="\\\\\n"
            else:
                end_str=" & "
            print(names[i])
            if abs(values[i][j])>1e-2 or values[i][j]==0:

                row_n=row_n+(str(round(values[i][j],3))+ end_str)
            else:
                row_n=row_n+("{:.3E}".format(Decimal(str(values[i][j])))+ end_str)
        value_rows.append(row_n)

    control_num=0

    for line in f:
        if line[0]=="\\":
            if line.strip()=="\hline":
                table_file.write(line)
            try:
                line.index("{")
                command=line[line.index("{")+1:line.index("}")]
                if (command=="tabular")and (control_num==0):
                    line=table_control1
                    control_num+=1
                elif (command=="tabular") and (control_num==2):
                    line=table_control2
                    control_num+=1
                elif command=="4":
                    line=table_title
                table_file.write(line)
            except:
                continue
        elif line[0]=="e":
            for q in range(0, len(names)+1):
                line=value_rows[q]
                table_file.write(line)
                table_file.write("\hline\n")
elif parameter_orientation =="column":
    f =open("image_tex_test_param_col.tex", "r")
    table_control_1="\\begin{tabular}{|"+(("c|"*title_num))+"}\n"
    titles=["& {0}".format(x) for x in names]
    titles="Parameter "+(" ").join(titles)+"\\\\\n"

    row_headings=[name_list[i]+" ("+unit_dict[optim_list[i]]+") " if unit_dict[optim_list[i]]!="" else name_list[i] for i in range(1, len(optim_list))]
    numerical_rows=[]
    for j in range(0, len(values[0])):
        int_row=""
        for q in range(0, len(names)):
            if values[q][j]>1e-2 or values[q][j]==0:
                int_row=int_row+"& "+(str(round(values[q][j],3)))+" "
            else:
                int_row=int_row+"& "+"{:.3E}".format(Decimal(str(values[q][j])))+" "

        numerical_rows.append(int_row+"\\\\\n\hline\n")
    for i in range(0, len(numerical_rows)):
        numerical_rows[i]=row_headings[i]+numerical_rows[i]
    for line in f:
        if line[0]=="\\":
            if "begin{tabular}" in line:
                table_file.write(table_control_1)
            else:
                table_file.write(line)
        elif "Parameter_line" in line:
                table_file.write(titles)
        elif "Data_line" in line:
            for q in range(0, len(numerical_rows)):
                table_file.write(numerical_rows[q])



f.close()

table_file.close()
filename=""
filename="alice_PSV_versiont_"
filename=filename+"table.png"
os.system("pdflatex image_tex_edited.tex")
os.system("convert -density 300 -trim image_tex_edited.pdf -quality 100 " + filename)
os.system("mv " +filename+" ~/Documents/Oxford/Cytochrome_SV/Results/Param_tables")
