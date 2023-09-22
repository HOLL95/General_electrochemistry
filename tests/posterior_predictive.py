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
from heuristic_class import Laviron_EIS, DCVTrumpet, PSV_harmonic_minimum
from multiplotter import multiplot
from harmonics_plotter import harmonics
from EIS_class import EIS
num_harmonics=5
from matplotlib.gridspec import GridSpec
import numpy as np
import copy 
fig=plt.figure()
min_height=2
plot_height=num_harmonics*min_height
plot_width=6
gap=1
lower_x=2*plot_height+(gap)
total_height=4*plot_height+(gap)
total_width=3*plot_width+(2*gap)
gs=GridSpec(total_height,total_width) # 2 rows, 3 columns
ax1=[fig.add_subplot(gs[x*min_height:(x*min_height)+min_height, plot_width+gap:gap+plot_width*2]) for x in range(0, num_harmonics)] # First row, first column

ax4=[fig.add_subplot(gs[(total_height-plot_height)+(x*min_height):(total_height-plot_height)+(x*min_height)+min_height,plot_width+gap:gap+plot_width*2])for x in range(0, num_harmonics)]



ax7=fig.add_subplot(gs[plot_height+gap:lower_x+plot_height-gap,plot_width:gap+plot_width*2], projection="3d")

ax2=fig.add_subplot(gs[plot_height:plot_height*2,0:plot_width]) # First row, second column

ax3=fig.add_subplot(gs[lower_x:lower_x+plot_height,0:plot_width])

ax5=fig.add_subplot(gs[plot_height:plot_height*2,total_width-plot_width:total_width]) # First row, second column
ax6=fig.add_subplot(gs[lower_x:lower_x+plot_height,total_width-plot_width:total_width])

#ax3=fig.add_subplot(gs[0,2]) # First row, third column
#ax4=fig.add_subplot(gs[1,:]) # Second row, span all columns
plt.subplots_adjust(top=0.965,
bottom=0.075,
left=0.05,
right=0.965,
hspace=0,
wspace=0)

param_list={
        "E_0":-0.09,  #(ac voltage amplitude - V) freq_range[j],#
        'area': 0.07, #(electrode surface area cm^2)
        'Ru': 100,  #     (uncompensated resistance ohms)
        'Cdl':5e-5, #(capacitance parameters)
        'CdlE1': 0.0653657774506,
        'CdlE2': 0.00245772700637,
        "CdlE3":-1e-5*0,
        'gamma': 1e-10,
        "original_gamma":1e-10,        # (surface coverage per unit area)
        'k_0': 50, #(reaction rate s-1)
        'alpha': 0.51,
        'sampling_freq' : (1.0/2**8),
        'phase' :3*math.pi/2,
        "cap_phase":3*math.pi/2+0.1,
        "time_end": None,
        'num_peaks': 4,
    }
solver_list=["Bisect", "Brent minimisation", "Newton-Raphson", "inverted"]
likelihood_options=["timeseries", "fourier"]
#time_start=1/(param_list["omega"])
simulation_options={
    "no_transient":False,
    "numerical_debugging": False,
    "experimental_fitting":False,
    "dispersion":False,
    "dispersion_bins":[16],
    "test":False,
    "method": "sinusoidal",
    "phase_only":False,
    "likelihood":likelihood_options[0],
    "numerical_method": solver_list[1],
    "label": "MCMC",
    "top_hat_return":"composite",
    "optim_list":[]
}
other_values={
    "filter_val": 0.5,
    "harmonic_range":list(range(3, 3+num_harmonics)),
    "bounds_val":2000,
    
}
param_bounds={
    'E_0':[-0.13, -0.05],
    'omega':[0,100],#8.88480830076,  #    (frequency Hz)
    'Ru': [0, 2e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-3], #(capacitance parameters)
    'CdlE1': [-0.05,0.15],#0.000653657774506,
    'CdlE2': [-0.01,0.01],#0.000245772700637,
    'CdlE3': [-0.01,0.01],#1.10053945995e-06,
    'gamma': [0.1*param_list["original_gamma"],5*param_list["original_gamma"]],
    'k_0': [30, 70], #(reaction rate s-1)
    'alpha': [0.4, 0.6],
}
big_ol_dict={"ramped":{"params":{"E_start":-500e-3, "E_reverse":200e-3, "v":22.5e-3, "d_E":150e-3, "omega":10}, "options":{"method":"ramped","no_transient":2/10}, "ax":ax1},
            "sinusoidal":{"params":{"E_start":-400e-3, "E_reverse":300e-3, "d_E":300e-3, "omega":10, "original_omega":10}, "options":{"method":"sinusoidal", "no_transient":2/10}, "ax":ax4},
            "dcv":{"params":{"E_start":-500e-3, "E_reverse":200e-3, "v":10000e-3, "omega":0,"d_E":0,}, "options":{"method":"dcv"},"ax":ax2},
            "eis":{"params":{"Cdl":param_list["Cdl"]*param_list["area"]}, "options":{"EIS_Cf":"C", "EIS_Cdl":"C", "DC_pot":param_list["E_0"]-0.005, "data_representation":"bode"}, "ax":ax6, "twinx":ax6.twinx()},
            "trumpet_plot":{"params":{"E_start":-500e-3, "E_reverse":200e-3, "v":50e-3, "omega":0,"d_E":0,"Ru":0, "dcv_sep":0.5, "sampling_freq":1/50}, "options":{"method":"dcv"}, "ax":ax5},
            "harmonic_minimum":{"params":{"E_start":param_list["E_0"]-300e-3, "E_reverse":200e-3, "d_E":300e-3, "omega":10, "original_omega":10}, "options":{"E_step_start":param_list["E_0"]-300e-3-12.5e-3, "num_steps":200,"E_step_range":25e-3,"return_magnitudes":True}, "ax":ax3},

}
 
from scipy.stats import multivariate_normal       
length=200
mid=int(length//2)
mean = np.array([-0.09, 50, 0.5])
covariance_matrix = np.matrix(np.random.rand(3,3))*0.05
vals=[0.05, 30, 0.1]
for i in range(0, 3):
    covariance_matrix[i,i]=vals[i]
cov=covariance_matrix*covariance_matrix.H
x1=np.linspace(-0.13, -0.05, length)
x2=np.linspace(30, 70, length)
x3=np.linspace(0.4, 0.6, length)
x, y, z = np.meshgrid(x1,x2,x3)
pos = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
pdf_values = multivariate_normal.pdf(pos, mean=mean, cov=cov)
print(pdf_values)
pdf_values = pdf_values.reshape(length, length, length)

x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
pdf_flat = pdf_values.flatten()
threshold=0.9*np.max(pdf_flat)
inside_sphere = pdf_flat > threshold
ax7.set_xlabel("$E^0$", rotation=0)

ax7.set_ylabel("$k_0$", rotation=0)
ax7.set_zlabel("$\\alpha$", rotation=0)
from matplotlib import cm
step=1
ax7.scatter(x_flat[inside_sphere][::step], y_flat[inside_sphere][::step], z_flat[inside_sphere][::step], c=pdf_flat[inside_sphere][::step], marker='.',s=0.1, alpha=0.01, cmap=cm.plasma)

z_plots=[pdf_values[:, :,mid],pdf_values[mid, :,:].T,pdf_values[:, mid,:].T]
from itertools import combinations
axes=list(combinations([x1, x2, x3],2))
for i in range(0, 3):
    current_z=z_plots[i].flatten()
    inside_sphere = current_z > threshold
    x_axis,y_axis=np.meshgrid(axes[i][0], axes[i][1])
    flatx=x_axis.flatten()
    flaty=y_axis.flatten()
    if i==0:#xE0y#k0
        x_arg=flatx[inside_sphere]
        
        y_arg=flaty[inside_sphere]
        z_arg=(np.ones(length**2)*x3[0])[inside_sphere]
    elif i==1:#xE0yalpha
        x_arg=flatx[inside_sphere]
        
        y_arg=(np.ones(length**2)*x2[-1])[inside_sphere]
        z_arg=flaty[inside_sphere]
    elif i==2:#xk0yalpha
        x_arg=(np.ones(length**2)*x1[0])[inside_sphere]
        y_arg=flatx[inside_sphere]
        z_arg=flaty[inside_sphere]

    
    

    ax7.scatter(x_arg, y_arg, z_arg,c=current_z[inside_sphere], alpha=1, cmap=cm.PuBu, s=3)
for axis in [ax7.xaxis, ax7.yaxis, ax7.zaxis]:
    axis.set_ticklabels([])
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = (0, 0, 0)
    axis._axinfo['grid']['linewidth'] = 0.1
    axis._axinfo['grid']['linestyle'] = "-"
    axis._axinfo['grid']['color'] = (0, 0, 0)
    axis._axinfo['tick']['inward_factor'] = 0.0
    axis._axinfo['tick']['outward_factor'] = 0.0
    axis.set_pane_color((1, 1, 1))

plot_keys=list(big_ol_dict.keys())

weird_ones=["ramped", "sinusoidal", "dcv"]
class_dict={key:single_electron for key in weird_ones}
class_dict["eis"]=Laviron_EIS
class_dict["trumpet_plot"]=DCVTrumpet
class_dict["harmonic_minimum"]=PSV_harmonic_minimum
harmonic_exps=["ramped", "sinusoidal"]
num_draws=15
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
#i_f= n F A -[Ox] e^{\big(\frac{-\alpha nF}{RT} (E-E^0)\big)}\bigg),
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
params=["E_0", "k_0", "alpha"]
ax5.text(-0.1, 1.7, s=r"$k_{ox}(E)=k_0e^{\frac{nF}{RT}(1-\alpha) (E-E^0)}$", transform=ax5.transAxes, weight="bold", fontsize=16)
ax5.text(-0.1, 1.4, s=r"$k_{red}(E)=k_0e^{(\frac{-\alpha nF}{RT} (E-E^0))}$", transform=ax5.transAxes, weight="bold", fontsize=16)
for j in range(0, num_draws):
    x0=np.random.rand(len(params))
    for i in range(0, len(plot_keys)):
        key=plot_keys[i]
    
        current_params=copy.deepcopy(param_list)
        current_options=copy.deepcopy(simulation_options)
        current_other=copy.deepcopy(other_values)
        current_bounds=copy.deepcopy(param_bounds)
        for pkey in big_ol_dict[key]["params"].keys():
            current_params[pkey]=big_ol_dict[key]["params"][pkey]
        for okey in big_ol_dict[key]["options"].keys():
            current_options[okey]=big_ol_dict[key]["options"][okey]
        
        if key in weird_ones:
            current_class=class_dict[key]("", current_params, current_options, current_other, current_bounds)
        else:
            current_class=class_dict[key](current_params, current_options, current_other, current_bounds)
            
        current_class.def_optim_list(params)
        if j==0:
            sim_params=[current_params[x] for x in current_class.optim_list]
            plot_args={"color":colours[0], "lw":3, "alpha":1}
        else:
            sim_params=current_class.change_norm_group(x0, "un_norm")
            plot_args={"color":colours[1], "lw":0.5, "alpha":0.5}
            print(sim_params)

        if key=="dcv":
            axis=big_ol_dict[key]["ax"]
            data=current_class.i_nondim(current_class.test_vals(sim_params, "timeseries"))
            xvals=current_class.e_nondim(current_class.define_voltages())
            axis.plot(xvals, data, **plot_args)
            axis.set_axis_off()
        elif key in harmonic_exps:
            h_class=harmonics(current_other["harmonic_range"], current_params["omega"], 0.5)
            axis=big_ol_dict[key]["ax"]
            t=current_class.t_nondim(current_class.time_vec)[current_class.time_idx]

            data=current_class.i_nondim(current_class.test_vals(sim_params, "timeseries"))
            if key=="sinusoidal":
                xvals=current_class.e_nondim(current_class.define_voltages())[current_class.time_idx]
                hanning=False
                plot_func=np.real
            else:
                xvals=t
                hanning=True
                plot_func=np.abs
            
            h_class.plot_harmonics(t, reference_time_series=data, xaxis=xvals, hanning=hanning, plot_func=plot_func, axes_list=axis, legend=None, h_num=False,colour=plot_args["color"], lw=plot_args["lw"], alpha=plot_args["alpha"])
            for element in axis:
                element.set_axis_off()
        elif key=="eis":
            frequency_powers=np.linspace(-1, 6, 7*10)
            frequencies=[10**x for x in frequency_powers]
            vals=current_class.simulate(sim_params, frequencies)
            axis=big_ol_dict[key]["ax"]
            twinx=big_ol_dict[key]["twinx"]
            EIS().bode(vals,frequencies, ax=axis, twinx=twinx, data_type="phase_mag",colour=plot_args["color"], lw=plot_args["lw"], alpha=plot_args["alpha"])
            axis.set_axis_off()
            twinx.set_axis_off()
        elif key=="trumpet_plot":
            axis=big_ol_dict[key]["ax"]
            scan_rate_powers=np.linspace(1, 5, 6*5)
            scan_rates=[1e-3*10**(x) for x in scan_rate_powers]
            vals=current_class.e_nondim(current_class.simulate(sim_params, scan_rates))
            if j==0:
                current_class.trumpet_plot(scan_rates, vals, ax=axis)
            else:
                axis.plot(np.log10(scan_rates), vals[:,0], **plot_args)
                axis.plot(np.log10(scan_rates), vals[:,1], **plot_args)
            axis.set_axis_off()
        elif key=="harmonic_minimum":
            axis=big_ol_dict[key]["ax"]
            vals=current_class.test_vals(sim_params, "timeseries")
            xaxis=current_class.define_potentials()
            axis.semilogy(xaxis, vals[2,:], **plot_args)
            axis.set_axis_off()
fig.set_size_inches(9, 7)
fig.savefig("bigfig.png", dpi=500)
plt.show()