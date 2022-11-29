import numpy as np
import matplotlib.pyplot as plt
import copy
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
from EIS_class import EIS
from circuit_drawer import circuit_artist
from matplotlib.widgets import Slider, Button
class explore_fit:
    def __init__(self, circuit, start_parameters, **kwargs):
        if not isinstance(start_parameters, dict):
            raise ValueError("initial parameters need to be a dictionary of parameter:value pairs")
        translator=EIS(circuit=circuit)
        if "frequencies" not in kwargs and "frequency_powers" not in kwargs:
            frequency_powers=np.arange(2, 10, 0.1)
            frequencies=[10**x for x in frequency_powers]
        elif "frequency_powers" in kwargs:
            frequency_powers=kwargs["frequency_powers"]
            frequencies=[10**x for x in frequency_powers]
        elif "frequencies" in kwargs:
            frequencies=kwargs["frequencies"]
        if "autoscale_factor" not in kwargs:
            autoscale_factor=0.1
        else:
            autoscale_factor=kwargs["autoscale_factor"]
        fig, ax=plt.subplots()
        if "bounds" in kwargs:
            param_bounds=kwargs["bounds"]
            if "bounds_factor" in kwargs:
                raise ValueError("Either bounds or bounds_factor, not both")
        elif "bounds_factor" in kwargs:
            if "bounds" in kwargs:
                raise ValueError("Either bounds or bounds_factor, not both")
            param_bounds={}
            min_bound=1/kwargs["bounds_factor"]
            for param in start_parameters.keys():
                param_bounds[param]=[min_bound*start_parameters[param], kwargs["bounds_factor"]*start_parameters[param]]
        else:
            param_bounds={}
            for param in start_parameters.keys():
                param_bounds[param]=[0.1*start_parameters[param], 10*start_parameters[param]]
        self.true_params_value_store=copy.deepcopy(start_parameters)
        spectra=translator.test_vals(start_parameters, frequencies )
        ncols=1
        if "data" in kwargs:
            if not isinstance(kwargs["data"], dict):
                translator.nyquist(kwargs["data"], ax=ax, colour="red", scatter=1, label="Data")#
                ncols=2
            else:
                data_keys=list(kwargs["data"].keys())
                for i in range(0, len(data_keys)):
                    translator.nyquist(kwargs["data"][data_keys[i]], ax=ax, colour=colours[i+1], scatter=1, label=data_keys[i])#
                ncols=len(data_keys)+1
        circuit_ax= fig.add_axes([0.77, 0.35, 0.2, 0.5])
        artist=circuit_artist(circuit, ax=circuit_ax)
        circuit_ax.set_axis_off()

        # Create the figure and the line that we will manipulate

        line, = ax.plot(spectra[:, 0], -spectra[:,1], lw=2, label="Simulation")

        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.1, bottom=0.35, right=0.75)
        # Make a horizontal slider to control the frequency.
        params=list(start_parameters.keys())
        interval=0.25/(len(params))
        slider_array={}
        class_array={}
        for i in range(0, len(params)):
            axfreq = fig.add_axes([0.1, 0.25-(i*interval), 0.65, 0.03])
            class_array[params[i]]=generic_function(params[i], frequencies, start_parameters, fig,ax, artist, translator, line, autoscale_factor)
            slider_array[params[i]] = Slider(
                ax=axfreq,
                label=params[i],
                valmin=param_bounds[params[i]][0],
                valmax=param_bounds[params[i]][1],
                valinit=start_parameters[params[i]],
            )
            slider_array[params[i]].on_changed(class_array[params[i]].update)
        scaleax = fig.add_axes([0.1, 0.9, 0.1, 0.04])
        button = Button(scaleax, 'Autoscale', hovercolor='0.975')
        resetax = fig.add_axes([0.2, 0.9, 0.1, 0.04])
        resetbutton = Button(resetax, 'Reset', hovercolor='0.975')
        button.on_clicked(self.scaling)
        resetbutton.on_clicked(self.reset)
        self.artist=artist
        self.params=params
        self.class_array=class_array
        ax.legend(loc="upper right", bbox_to_anchor=(1, 1.2), ncol=ncols)
        plt.show()



    def reset(self, event):
        for i in range(0, len(self.params)):
            self.class_array[self.params[i]].update(self.true_params_value_store[self.params[i]])
        for names in self.artist.patch_library.keys():
            current_colour=self.artist.patch_library[names].get_facecolor()
            if current_colour[0]==0.8627450980392157:
                self.artist.patch_library[names].set_facecolor(colours[0])
    def scaling(self, event):
        #values=aritst.patch_library["CPE_1"]
        #rect=patches.Rectangle(values["position"], values["width"], values["height"], faceolor="crimson "

        for key in self.params:
            element=self.class_array[key]
            #print(element)
            element.autoscale=not element.autoscale

class generic_function:
    def __init__(self, param, frequencies, true_params, fig, ax, artist, sim_class, line, autoscale_factor):
        self.param=param
        self.frequencies=frequencies
        self.true_params=true_params
        self.fig=fig
        self.ax=ax
        self.autoscale=True
        self.artist=artist
        self.autoscale_factor=autoscale_factor
        if "Q" in param:
            self.name="CPE"+self.param[self.param.index("Q")+1]
        elif "alpha" in param:
            self.name="CPE"+self.param[self.param.index("h")+2]
        else:
            self.name=self.param
        self.sim_class=sim_class
        self.line=line
    def update(self, value):
        #print(value, self.param)
        for names in self.artist.patch_library.keys():
            current_colour=self.artist.patch_library[names].get_facecolor()
            if current_colour[0]==0.8627450980392157:
                self.artist.patch_library[names].set_facecolor(colours[0])
        self.artist.patch_library[self.name].set_facecolor("crimson")
        self.true_params[self.param]=value
        spectra=self.sim_class.test_vals(self.true_params,self.frequencies )
        neg_spectra=-spectra[:,1]
        self.line.set_ydata(neg_spectra)
        self.line.set_xdata(spectra[:,0])
        #self.ax.autoscale()
        minmax={"x":{}, "y":{}}
        minmax_funcs={"min":min, "max":max}
        spectras={"x":spectra[:,0], "y":neg_spectra}
        for axis in ["x", "y"]:
            for val in ["min", "max"]:
                minmax[axis][val]=minmax_funcs[val](spectras[axis])
        scale_factor=self.autoscale_factor
        if self.autoscale==True:
            self.ax.set_xlim([minmax["x"]["min"]-scale_factor*minmax["x"]["min"], minmax["x"]["max"]+0.1*minmax["x"]["max"]])
            self.ax.set_ylim([max(0,minmax["y"]["min"]-scale_factor*minmax["y"]["min"]) , minmax["y"]["max"]+scale_factor*minmax["y"]["max"]])

    #spectra=translator.test_vals(true_params,frequencies )
    #line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    #fig.canvas.draw_idle()#
