import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
from PIL import Image
from pints import rhat
class MCMC_plotting:
    def __init__(self,**kwargs):
        self.options=kwargs
        if "burn" not in self.options:
            self.options["burn"]=0
        self.unit_dict={
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
                        'gamma': '$mol cm^{-2}$',
                        'k_0': '$s^{-1}$', #(reaction rate s-1)
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
                        "sep":"V"
                        }
        self.fancy_names={
                        "E_0": '$E^0$',
                        'E_start': '$E_{start}$', #(starting dc voltage - V)
                        'E_reverse': '$E_{reverse}$',
                        'omega':'$\\omega$',#8.88480830076,  #    (frequency Hz)
                        'd_E': "$\\Delta E$",   #(ac voltage amplitude - V) freq_range[j],#
                        'v': "v",   #       (scan rate s^-1)
                        'area': "Area", #(electrode surface area cm^2)
                        'Ru': "$R_u$",  #     (uncompensated resistance ohms)
                        'Cdl': "$C_{dl}$", #(capacitance parameters)
                        'CdlE1': "$C_{dlE1}$",#0.000653657774506,
                        'CdlE2': "$C_{dlE2}$",#0.000245772700637,
                        'CdlE3': "$C_{dlE3}$",#1.10053945995e-06,
                        'gamma': '$\\Gamma$',
                        'E0_skew':"$E^0$ skew",
                        'k_0': '$k_0$', #(reaction rate s-1)
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
                        "sep":"Seperation"
                        }
    def det_subplots(self, value):
        if np.floor(np.sqrt(value))**2==value:
            return int(np.sqrt(value)), int(np.sqrt(value))
        elif value<4:
            return 1, value
        if value<=10:
            start_val=2
        else:
            start_val=3

        rows=range(start_val, int(np.ceil(value/start_val)))
        for i in range(0, 10):
            modulos=np.array([value%x for x in rows])
            idx_0=(np.where(modulos==0))
            if len(idx_0[0])!=0:
                return int(rows[idx_0[0][-1]]), int(value/rows[idx_0[0][-1]])
            value+=1
    def get_titles(self, titles, **kwargs):
        if "units" not in kwargs:
            kwargs["units"]=True
        if "positions" not in kwargs:
            kwargs["positions"]=range(0, len(titles))
        if kwargs["units"]==True:
            params=[self.fancy_names[titles[x]]+" ("+self.unit_dict[titles[x]]+")" if self.unit_dict[titles[x]]!="" else self.fancy_names[titles[x]] for x in kwargs["positions"]]
        else:
            params=[self.fancy_names[titles[x]] for x in kwargs["positions"]]
        return params
    def change_param(self, params, optim_list, parameter, value):
        param_list=copy.deepcopy(params)
        param_list[optim_list.index(parameter)]=value
        return param_list
    def chain_appender(self, chains, param):
        new_chain=chains[0, self.options["burn"]:, param]
        for i in range(1, len(chains)):
            new_chain=np.append(new_chain, chains[i, self.options["burn"]:, param])
        return new_chain
    def plot_params(self, titles, set_chain, **kwargs):
        if "positions" not in kwargs:
            kwargs["positions"]=range(0, len(titles))
        if "row" not in kwargs:
            row, col=self.det_subplots(len(titles))
        else:
            row=kwargs["row"]
            col=kwargs["col"]
        if "label" not in kwargs:
            kwargs["label"]=None
        if "axes" not in kwargs:
            fig, ax=plt.subplots(row, col)
        else:
            ax=kwargs["axes"]
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        if "pool" not in kwargs:
            kwargs["pool"]=True
        if "Rhat_title" not in kwargs:
            kwargs["Rhat_title"]=False

        if "true_values" not in kwargs:
            kwargs["true_values"]=False        
        titles=self.get_titles(titles, units=True, positions=kwargs["positions"])
        for i in range(0, len(titles)):
            axes=ax[i//col, i%col]
            plot_chain=self.chain_appender(set_chain, kwargs["positions"][i])#axes.set_title()
            if abs(np.mean(plot_chain))<1e-5:
                axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
            elif abs(np.mean(plot_chain))<0.001:
                axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            else:
                order=np.log10(np.std(plot_chain))
                if order>1:
                    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                else:
                    order_val=abs(int(np.ceil(order)))+1
                    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.{0}f'.format(order_val)))
            if kwargs["pool"]==True:
                axes.hist(plot_chain,bins=20, stacked=True, label=kwargs["label"], alpha=kwargs["alpha"])
            else:
                for j in range(0, len(set_chain[:, 0, 0])):
                    axes.hist(set_chain[j, self.options["burn"]:, i],bins=20, stacked=True, label="Chain {0}".format(i+1), alpha=kwargs["alpha"])
            if kwargs["Rhat_title"]==True:
                rhat_val=rhat(set_chain[:, self.options["burn"]:, i])
                axes.set_title(round(rhat_val, 3))
            if kwargs["true_values"]!=False:
                axes.axvline(kwargs["true_values"][kwargs["positions"][i]], linestyle="--", color="black")
            #axes.legend()
            lb, ub = axes.get_xlim( )
            axes.set_xticks(np.linspace(lb, ub, 3))
            axes.set_xlabel(titles[i])
            axes.set_ylabel('frequency')
            
            #axes.set_title(graph_titles[i])
        return  ax
    def axes_legend(self, label_list,  ax,**kwargs):
        
            
        for i in range(0, len(label_list)):
            ax.plot(0, 0, label=label_list[i])
        if "bbox" not in kwargs:
            ax.legend()
        else:
            ax.legend(bbox_to_anchor=kwargs["bbox"])
        ax.set_axis_off()
    def trace_plots(self, titles, chains, names, rhat=False, burn_in_thresh=0):
        row, col=det_subplots(len(titles))
        if rhat==True:
            rhat_vals=pints.rhat_all_params(chains[:, burn_in_thresh:, :])
        for i in range(0, len(titles)):
            axes=plt.subplot(row,col, i+1)
            for j in range(0, len(chains)):
                axes.plot(chains[j, :, i], label="Chain "+str(j), alpha=0.7)
            if abs(np.mean(chains[j, :, i]))<0.01:
                axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            if "omega" in titles[i]:
                axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
            #else:
            #    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            axes.set_xticks([0, 10000])

            if i>(len(titles))-(col+1):
                axes.set_xlabel('Iteration')
            #if i==len(titles)-1:
            #    axes.legend(loc="center", bbox_to_anchor=(2.0, 0.5))
            #lb, ub = axes.get_xlim( )
            #axes.set_xticks(np.linspace(lb, ub, 5))
            if rhat == True:
                axes.set_title(names[i]+ " Rhat="+str(round(rhat_vals[i],3))+"")
            else:
                axes.set_title(names[i])
            axes.set_ylabel(titles[i])
    