import numpy as np
import matplotlib.pyplot as plt
import os
import sys
colours=plt.rcParams['axes.prop_cycle'].by_key()['color']
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
from circuit_drawer import circuit_artist
randles={"z1":"R0", "z2":{"p1":"R1", "p2":"C1"},"z3":{"p1":["R2", ("Q2", "alpha2")], "p2":("Q1", "alpha1")}, }
translator=EIS(circuit=randles)
frequency_powers=np.arange(2, 10, 0.05)
frequencies=[10**x for x in frequency_powers]
fig, ax=plt.subplots()
bounds={
"R1":[0, 25],
"Q1":[0, 0.5],
"Q2":[0, 0.5],
"alpha1":[0.1, 0.9],
"alpha2":[0.1, 0.9],
"R2":[0, 25],
"R0":[0, 25],
"C1":[0, 1e-4]
}
from matplotlib.widgets import Slider, Button
true_params={   "R0":10, "R1":2, "C1":1.5e-5,"R2":5, "Q1":0.0125, "alpha1":0.5,"Q2":0.075, "alpha2":0.25,}
spectra=translator.test_vals(true_params,frequencies )
circuit_ax= fig.add_axes([0.77, 0.35, 0.2, 0.5])
artist=circuit_artist(randles, ax=circuit_ax)
circuit_ax.set_axis_off()

# Create the figure and the line that we will manipulate

line, = ax.plot(spectra[:, 0], -spectra[:,1], lw=2)
import copy

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.1, bottom=0.35, right=0.75)

class generic_function:
    def __init__(self, param, frequencies, true_params, fig, ax):
        self.param=param
        self.frequencies=frequencies
        self.true_params=true_params
        self.fig=fig
        self.ax=ax
        self.autoscale=True
        if "Q" in param:
            self.name="CPE"+self.param[self.param.index("Q")+1]
        elif "alpha" in param:
            self.name="CPE"+self.param[self.param.index("h")+2]
        else:
            self.name=self.param
    def update(self, value):
        #print(value, self.param)
        for names in artist.patch_library.keys():
            current_colour=artist.patch_library[names].get_facecolor()
            if current_colour[0]==0.8627450980392157:
                artist.patch_library[names].set_facecolor(colours[0])
        artist.patch_library[self.name].set_facecolor("crimson")
        self.true_params[self.param]=value
        spectra=translator.test_vals(self.true_params,self.frequencies )
        neg_spectra=-spectra[:,1]
        line.set_ydata(neg_spectra)
        line.set_xdata(spectra[:,0])
        #self.ax.autoscale()
        minmax={"x":{}, "y":{}}
        minmax_funcs={"min":min, "max":max}
        spectras={"x":spectra[:,0], "y":neg_spectra}
        for axis in ["x", "y"]:
            for val in ["min", "max"]:
                minmax[axis][val]=minmax_funcs[val](spectras[axis])
        if self.autoscale==True:
            self.ax.set_xlim([minmax["x"]["min"]-0.05*minmax["x"]["min"], minmax["x"]["max"]+0.1*minmax["x"]["max"]])
            self.ax.set_ylim([0, minmax["y"]["max"]+0.05*minmax["y"]["max"]])

    #spectra=translator.test_vals(true_params,frequencies )
    #line.set_ydata(f(t, amp_slider.val, freq_slider.val))
    #fig.canvas.draw_idle()#

# Make a horizontal slider to control the frequency.
params=list(true_params.keys())
interval=0.25/(len(params))
slider_array={}
class_array={}
for i in range(0, len(params)):
    axfreq = fig.add_axes([0.1, 0.25-(i*interval), 0.65, 0.03])
    class_array[params[i]]=generic_function(params[i], frequencies, true_params, fig,ax)
    slider_array[params[i]] = Slider(
        ax=axfreq,
        label=params[i],
        valmin=bounds[params[i]][0],
        valmax=bounds[params[i]][1],
        valinit=true_params[params[i]],
    )
    slider_array[params[i]].on_changed(class_array[params[i]].update)
resetax = fig.add_axes([0.1, 0.9, 0.1, 0.04])
button = Button(resetax, 'Autoscale', hovercolor='0.975')


def reset(event):
    #values=aritst.patch_library["CPE_1"]
    #rect=patches.Rectangle(values["position"], values["width"], values["height"], faceolor="crimson "

    for key in params:
        element=class_array[key]
        #print(element)
        element.autoscale=not element.autoscale
button.on_clicked(reset)
plt.show()
# Make a vertically oriented slider to control the amplitude



# The function to be called anytime a slider's value changes


# register the update function with each slider
"""
freq_slider.on_changed(update)
amp_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')


def reset(event):
    freq_slider.reset()
    amp_slider.reset()
button.on_clicked(reset)

plt.show()
plt.legend()
plt.show()
"""
