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
    'v': '$s^{-1}$',   #       (scan rate s^-1)
    'area': '$cm^{2}$', #(electrode surface area cm^2)
    'Ru': "$\\Omega$",  #     (uncompensated resistance ohms)
    'Cdl': "F", #(capacitance parameters)
    'CdlE1': "",#0.000653657774506,
    'CdlE2': "",#0.000245772700637,
    'CdlE3': "",#1.10053945995e-06,
    'gamma': 'mol cm$^{-2}$',
    'k_0': 's^{-1}$', #(reaction rate s-1)
    'alpha': "",
    "E0_mean":"V",
    "E0_std": "V",
    "E0_skew":"",
    "k0_shape":"",
    "k0_loc":"",
    "k0_scale":"",
    "cap_phase":"rads",
    'phase' : "rads",
    "alpha_mean": "",
    "alpha_std": "",
    "sampling_freq":"$s^{-1}$",
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
    'Ru': "R$_u$",  #     (uncompensated resistance ohms)
    'Cdl': "$C_{dl}$", #(capacitance parameters)
    'CdlE1': "$C_{dlE1}$",#0.000653657774506,
    'CdlE2': "$C_{dlE2}$",#0.000245772700637,
    'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
    'gamma': '$\\Gamma$',
    'k_0': '$k_0', #(reaction rate s-1)
    'alpha': "$\\alpha$",
    "E0_mean":"$E^0 \\mu$",
    "E0_std": "$E^0 \\sigma$",
    "E0_skew":"$E^0 \\alpha$",
    "cap_phase":"C$_{dl}$ phase",
    "k0_shape":"$log(k^0(s^{-1}))\\sigma$",
    "k0_scale":"$log(k^0(s^{-1}))\\mu$",
    "alpha_mean": "$\\alpha\\mu$",
    "sampling_freq":"",
    "alpha_std": "$\\alpha\\sigma$",
    'phase' : "Phase",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"RMSE",
}
total_names={
    "E_0": 'Midpoint potential',
    'E_start': 'Start potential', #(starting dc voltage - V)
    'E_reverse': 'Reverse potential',
    'omega':'Potential frequency',#8.88480830076,  #    (frequency Hz)
    'd_E': "Potential amplitude",   #(ac voltage amplitude - V) freq_range[j],#
    'v': "Scan rate",   #       (scan rate s^-1)
    'area': "Area", #(electrode surface area cm^2)
    'Ru': "Uncompensated resistance",  #     (uncompensated resistance ohms)
    'Cdl': "Linear double-layer capacitance", #(capacitance parameters)
    'CdlE1': "$1^{st}$ order $C_{dl}$",#0.000653657774506,
    'CdlE2': "$2^{nd}$ order $C_{dl}$",#0.000245772700637,
    'CdlE3': "$3^{rd}$ order $C_{dl}$",#1.10053945995e-06,
    'gamma': 'Surface coverage',
    'k_0': 'Rate constant', #(reaction rate s-1)
    'alpha': "Symmetry factor",
    "E0_mean":"Midpoint potential mean",
    "E0_std": "Midpoint potential standard deviation",
    "E0_skew": "Midpoint potential skew",
    "cap_phase":"C$_{dl}$ phase",
    "k0_shape":"$k^0$ shape",
    "k0_scale":"$k^0$ scale",
    "alpha_mean": "Symmetry factor mean",
    "alpha_std": "Symmetry factor standard deviation",
    'phase' : "Phase",
    "sampling_freq":"Sampling rate",
    "":"Experiment",
    "noise":"$\sigma$",
    "error":"Error",
}

optim_list=["","E0_mean", "E0_std","k_0","Ru","Cdl","CdlE1", "CdlE2", "CdlE3","gamma","omega","cap_phase","phase", "alpha"]

name_list=[fancy_names[x] for x in optim_list]
title_list=[total_names[x] for x in optim_list[1:]]

values=[
[-0.07292658517785523, 0.05193560925470919, 2998.8556810906766, 35.99526556665516, 9.946796575207644e-06, -0.012921496361667895, 0.08494909531617927, -0.0005465500095515746, 1.3629147068830184e-11, 9.015059345540491, 4.692336610750279, 4.443801455736968, 0.539994692454598],
[-0.07294769155362478, 0.051962575637966885, 2999.990375330362, 35.492159242552916, 9.039749504563757e-06, -0.014251817867932712, 0.09415578558894251, -0.0006014959087911192, 1.3636807735014027e-11, 9.015059373786189, 4.69167813464518, 4.4434618285655, 0.5299999922213879],
[-0.07297173166763068, 0.05198577276744508, 2999.9897871725298, 35.025277179410125, 9.35490381772471e-06, -0.01406125283152293, 0.09161911380045286, -0.0005813820075654536, 1.3642753532042338e-11, 9.01505941029573, 4.691516752953211, 4.443476987643263, 0.5199999947193429],
[-0.07264580933125617, 0.05278148518728566, 2999.9984922820136, 37.47243441594181, 9.97727010240396e-06, -0.006135928593905743, 0.08338474293305194, -0.0005465707738678618, 1.392841293571772e-11, 9.015071679100052, 4.690263928312251, 4.443395484214806, 0.5099999611492858],
[-0.07300690682142727, 0.05203286428992002, 2999.997178061295, 34.2022056959325, 9.318800212464752e-06, -0.014419788640598702, 0.09311746339010758, -0.0005839301790235077, 1.3656273171754645e-11, 9.015059348337555, 4.69094376744224, 4.443349047042268, 0.4999999787831698]]
names=[str(x) for x in [ 0.54, 0.53, 0.52, 0.51, 0.5]]

parameter_orientation="column"
param_num=len(name_list)+1
title_num=len(names)+2
table_file=open("image_tex_edited.tex", "w")


#names=my_list[0]
if parameter_orientation=="row":
    f =open("image_tex_test.tex", "r")
    table_title="\\multicolumn{"+str(param_num-1)+"}{|c|}{Parameter values}\\\\ \n"
    table_control1="\\begin{tabular}{|"+(("c|"*param_num))+"}\n"
    table_control2="\\begin{tabular}{|"+(("p{3cm}|"*param_num))+"}\n"
    value_rows=[]
    row0=(" & ").join(title_list)
    row0+="\\\\\n"
    value_rows.append(row0)
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

                row_n=row_n+(str(round(values[i][j],2))+ end_str)
            else:
                row_n=row_n+("{:.2E}".format(Decimal(str(values[i][j])))+ end_str)
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
            for q in range(0, len(names)+2):
                line=value_rows[q]
                table_file.write(line)
                table_file.write("\hline\n")
elif parameter_orientation =="column":
    f =open("image_tex_test_param_col.tex", "r")
    table_control_1="\\begin{tabular}{|"+(("c|"*title_num))+"}\n"
    titles=["& {0}".format(x) for x in names]
    titles="Parameter & "+"Symbol"+(" ").join(titles)+"\\\\\n"
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
        numerical_rows[i]=title_list[i]+" & "+row_headings[i]+numerical_rows[i]
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
filename="All_PSV_"
filename=filename+"table.png"
os.system("pdflatex image_tex_edited.tex")
os.system("convert -density 300 -trim image_tex_edited.pdf -quality 100 " + filename)
os.system("mv " +filename+" ~/Documents/Oxford/FTacV_2/FTacV_results/Param_tables")
