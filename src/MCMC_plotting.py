import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
from PIL import Image
import re

import pints
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
                        'gamma': 'mol cm$^{-2}$',
                        'k_0': '$s^{-1}$', #(reaction rate s-1)
                        'alpha': "",
                        'E0_skew':"",
                        "E0_mean":"V",
                        "E0_std": "V",
                        "k0_shape":"",
                        "sampling_freq":"$s^{-1}$",
                        "k0_loc":"",
                        "k0_scale":"",
                        "cap_phase":"",
                        'phase' : "",
                        "alpha_mean": "",
                        "alpha_std": "",
                        "":"",
                        "noise":"",
                        "error":"$\\mu A$",
                        "sep":"V",
                        "cpe_alpha_faradaic" :"",
                        "sigma":""
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
                        "cap_phase":"C$_{dl}$ $\\eta$",
                        "k0_shape":"$\\log(k^0) \\sigma$",
                        "k0_scale":"$\\log(k^0) \\mu$",
                        "alpha_mean": "$\\alpha\\mu$",
                        "alpha_std": "$\\alpha\\sigma$",
                        'phase' : "$\\eta$",
                        "sampling_freq":"Sampling rate",
                        "":"Experiment",
                        "noise":"$\sigma$",
                        "error":"RMSE",
                        "sep":"Seperation",
                        "cpe_alpha_faradaic" :"$\\psi$",
                        "sigma":"$\\sigma$"
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
        params=["" for x in kwargs["positions"]]
       
        for i in range(0, len(kwargs["positions"])):
            z=kwargs["positions"][i]
            if titles[z] in self.fancy_names:
                if kwargs["units"]==True and self.unit_dict[titles[z]]!="":
                    
                    params[i]=self.fancy_names[titles[z]]+" ("+self.unit_dict[titles[z]]+")" 
                else:
                    params[i]=self.fancy_names[titles[z]]
            else:
                for key in self.fancy_names.keys():
                    if re.search(".*_[1-9]+", titles[z]) is not None:
                        underscore_idx=[i for i in range(0, len(titles[z])) if titles[z][i]=="_"]
                        true_name=titles[z][:underscore_idx[-1]]
                        value=titles[z][underscore_idx[-1]+1:]
                        if kwargs["units"]==True and self.unit_dict[true_name]!="":
                            params[i]=self.fancy_names[true_name]+"$_{"+value +"}$"+" ("+self.unit_dict[true_name]+")" 
                        else:
                            params[i]=self.fancy_names[true_name]+"$_{"+value +"}$"
                        break

        return params
    def format_values(self, value, dp=2):
        abs_val=abs(value)
        if abs_val<1000 and abs_val>0.01:
            return str(round(value, dp))
        else:
            return "{:.{}e}".format(value, dp)
    def get_units(self, titles, **kwargs):
        if "positions" not in kwargs:
            kwargs["positions"]=range(0, len(titles))
        units=[self.unit_dict[titles[x]] for x in kwargs["positions"]]
        return units
    def change_param(self, params, optim_list, parameter, value):
        param_list=copy.deepcopy(params)
        param_list[optim_list.index(parameter)]=value
        return param_list
    def chain_appender(self, chains, param):
        new_chain=chains[0, self.options["burn"]:, param]
        for i in range(1, len(chains)):
            new_chain=np.append(new_chain, chains[i, self.options["burn"]:, param])
        return new_chain
    def concatenate_all_chains(self, chains):
        return [self.chain_appender(chains, x) for x in range(0, len(chains[0, 0, :]))]
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
    def trace_plots(self, params, chains,**kwargs):
        if "rhat" not in kwargs:
            rhat =False
        else:
            rhat=kwargs["rhat"]
        if "burn" not in kwargs:
            burn=0
        else:
            burn=kwargs["burn"]
        if "order" not in kwargs:
            order=list(range(0, len(params)))
        else:
            
            order=kwargs["order"]
            print(order)
        if "true_values" not in kwargs:
            kwargs["true_values"]=None
        row, col=self.det_subplots(len(params))
        
        if rhat==True:
            rhat_vals=pints.rhat(chains[:, burn:, :], warm_up=0.5)
        names=self.get_titles(params, units=False)
        y_labels=self.get_titles(params, units=True)
        for i in range(0, len(params)):
            axes=plt.subplot(row,col, i+1)
            for j in range(0, len(chains)):
                axes.plot(chains[j, :, order[i]], label="Chain "+str(j), alpha=0.7)
            if abs(np.mean(chains[j, :, order[i]]))<0.01:
                axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
            if "omega" in params[i]:
                axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.5f'))
            #else:
            #    axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            #axes.set_xticks([0, 10000])
            if kwargs["true_values"] is not None:
                if params[i] in kwargs["true_values"]:
                    print(params[i], params[i])
                    axes.axhline(kwargs["true_values"][params[i]], color="black", linestyle="--")
            if i>(len(params))-(col+1):
                axes.set_xlabel('Iteration')
            #if i==len(titles)-1:
            #    axes.legend(loc="center", bbox_to_anchor=(2.0, 0.5))
            #lb, ub = axes.get_xlim( )
            #axes.set_xticks(np.linspace(lb, ub, 5))
            if rhat == True:
                axes.set_title("$\\hat{R}$="+str(round(rhat_vals[i],2))+"")
            else:
                axes.set_title(names[i])
            
            axes.set_ylabel(y_labels[i])
    def plot_2d(self, params, chains, **kwargs):
        n_param=len(params)
        if "pooling" not in kwargs:
            kwargs["pooling"]=False
        if "burn" not in kwargs:
            burn=0
        else:
            burn=kwargs["burn"]
        if "order" not in kwargs:
            kwargs["order"]=list(range(0, n_param))
        if "true_values" not in kwargs:
            kwargs["true_values"]=None
        if "density" not in kwargs:
            kwargs["density"]=False
        fig, ax=plt.subplots(n_param, n_param)
        chain_results=chains
        labels=self.get_titles(params, units=True)
        for i in range(0,n_param):
           
            z=kwargs["order"][i]
                
            pooled_chain_i=[chain_results[x, burn:, z] for x in range(0, 3)]
            if kwargs["pooling"]==True: 
                
                chain_i=[np.concatenate(pooled_chain_i)]
            else:
                chain_i=pooled_chain_i
            if "rotation" not in kwargs:
                kwargs["rotation"]=False
            #for m in range(0, len(labels)):
            #    box_params[chain_order[i]][labels[m]][exp_counter]=func_dict[labels[m]]["function"](chain_i, *func_dict[labels[m]]["args"])
            #chain_i=np.multiply(chain_i, values[i])
            for j in range(0, n_param):
                
                m=kwargs["order"][j]
                if i==j:
                    axes=ax[i,j]
                    ax1=axes.twinx()
                    for z in range(0, len(chain_i)):
                        axes.hist(chain_i[z], density=kwargs["density"])
                    ticks=axes.get_yticks()
                    axes.set_yticks([])
                    ax1.set_yticks(ticks)
                    if kwargs["density"] is False:
                        ax1.set_ylabel("Frequency")
                    else:
                        ax1.set_ylabel("Density")
                elif i<j:
                    ax[i,j].axis('off')
                else:
                    axes=ax[i,j]
                    chain_j=[chain_results[x, burn:, m] for x in range(0, 3)]
                    if kwargs["pooling"]==True: 
                        chain_j=[np.concatenate(chain_j)]
                    for z in range(0, len(chain_i)):
                        axes.scatter(chain_j[z], chain_i[z], s=0.5)
                if kwargs["true_values"] is not None:
                    
                    if params[j] in kwargs["true_values"] and params[i] in kwargs["true_values"]:
                        if i>j:
                            axes.scatter(kwargs["true_values"][params[j]],kwargs["true_values"][params[i]], color="black", s=20, marker="x")
                        elif i==j:
                            axes.axvline(kwargs["true_values"][params[i]], color="black", linestyle="--")

                """ if i!=0:
                    if chain_order[i]=="CdlE3":
                        ax[i, 0].set_ylabel(titles[i], labelpad=20)
                    elif chain_order[i]=="gamma":
                        ax[i, 0].set_ylabel(titles[i], labelpad=30)
                    else:
                        """
                if i<n_param-1:
                    ax[i,j].set_xticklabels([])#
                if j>0 and i!=j:
                    ax[i,j].set_yticklabels([])
                if j!=n_param:
                    ax[-1, i].set_xlabel(labels[i])
                    if np.mean(np.abs(chain_i))<1e-4:
                        ax[-1, i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
                    if kwargs["rotation"] is not False:
                        plt.setp( ax[-1, i].xaxis.get_majorticklabels(), rotation=kwargs["rotation"] )
                if i!=0:
                    ax[i, 0].set_ylabel(labels[i])
                    if np.mean(np.abs(chain_i))<1e-4:
                        ax[i, 0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
                    
                    
        return ax

    