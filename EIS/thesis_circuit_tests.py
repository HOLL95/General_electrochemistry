import os
import sys
import math
dir=os.getcwd()
dir_list=dir.split("/")
source_list=dir_list[:-1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
from EIS_class import EIS
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.image as mpimg
from scipy import special
thesis_circuits=[{"z1":"R1"}, {"z1":"C1"}, {"z1":("Q1", "alpha1")}]
faradaic_circuit={"z1":"R1", "z2":{"p_1":"R1", "p_2":"C1"}}
#debeye={"z1":{"p_1":"C1", "p_2":["R1", ("Q1", "alpha1")]}}
params={"R1":10, "C1":1e-5, "Q1":1, "alpha1":0.5}
variables={"R1":{"vals":[5, 10, 20], "sym":"R", "unit":"$\\Omega$"}, "C1":{"vals":np.flip([1e-1, 2.5e-1, 3e-1]), "sym":"C", "unit":"F"},  "alpha1":{"vals":[0.9, 0.8, 0.7], "sym":"$\\alpha$", "unit":""}}
variable_key=list(variables.keys())
#test.write_model(circuit=Randles)
frequency_powers=np.arange(-1.5, 5, 0.1)
frequencies=[10**x for x in frequency_powers]
spectra=np.zeros((2, len(frequencies)), dtype="complex")
image_files=["circuit_pics/{0}.png".format(x) for x in ["Rs", "Cdl", "CPE"]]
t=np.linspace(0, 2*math.pi)
amp=5e-3
potential = amp*np.sin(t)
fig, ax=plt.subplots(2, 3)
def cpe1(denom, i, charge_array, time_array, dt):
    total=0
    if i!=0:
        time_array=np.flip(time_array)
        total=np.sum(np.multiply(time_array, charge_array))
    return dt*total/denom


alphas=variables["alpha1"]["vals"]
cpes=[]
for z in range(0, len(alphas)):
        alpha1=alphas[z]
        q=0
        ramp=0.01
        step=0.0001
        sampling_rate=20000
        freq_percentage=0.05
        sampling_frequency=1/sampling_rate
        interval_points=sampling_rate
        period=t[-1]#(num_oscillations/frequencies[j])
        interval_times=np.linspace(0, period, interval_points)
        dt=interval_times[1]-interval_times[0]
        convolv=np.zeros(interval_points)
        current_2=np.zeros(interval_points)
        convolv[1:]=np.power(interval_times[1:], alpha1-1)
        denom=params["Q1"]*special.gamma(alpha1)
        interval_potential=amp*np.sin(interval_times)
        for i in range(0, interval_points):
            vtotal=interval_potential[i]
            vcpe=cpe1(denom, i, current_2[:i], convolv[:i], dt)
            current_2[i]=(vtotal-vcpe)
        print(i)
        cpes.append(current_2)
time_series={"R1":{"t":t, "e":potential, "i":[np.divide(potential, x) for x in variables["R1"]["vals"]]},
            "C1": {"t":t, "e":potential, "i":[amp*x*np.cos(t) for x in variables["C1"]["vals"]]},
            "alpha1": {"t":interval_times, "e":interval_potential, "i":cpes}}

for i in range(0, len(thesis_circuits)):
    sim_class=EIS(circuit=thesis_circuits[i])

    #circuit_pic=
    for j in range(0, len(variables[variable_key[i]]["vals"])):
        val=variables[variable_key[i]]["vals"][j]
        params[variable_key[i]]=val
        spectra=sim_class.test_vals(params, frequencies)
        img = mpimg.imread(image_files[i])
        sim_class.nyquist(spectra, ax=ax[0,i], scatter=2, label="{0}={1} {2}".format(variables[variable_key[i]]["sym"], val, variables[variable_key[i]]["unit"]))
        #print(["10^{0}".format(round(x,3)) for x in np.log10(frequencies[0::2])])
        key=variable_key[i]
        ax[1,i].plot(time_series[key]["t"], time_series[key]["i"][j]*1e3)
        ax[1,i].set_xlabel("Time (s)")
        ax[1,i].set_ylabel("Current (mA)")
        if j==0:
            ts_twinx=ax[1,i].twinx()
            ts_twinx.plot(time_series[key]["t"], time_series[key]["e"]*1e3, linestyle="--", color="black")
            ts_twinx.set_ylabel("Potential (mV)")
    ax[0,i].legend(loc = "lower right")

        #sim_class.bode(spectra, frequencies, ax=ax[1,i], type="phase")
    #print(spectra)
    #print(spectra[:,0])
    #plt.plot(spectra[:,0], spectra[:,1], scatter=8)
    #print(spectra[:,0], spectra[:,1])
    im_ax=ax[0, i].inset_axes([0.1, 0.6, 0.35, 0.35])
    im_ax.imshow(img)
    im_ax.axis("off")


plt.show()
fig.savefig("thesis_appendix_nyquist.png", dpi=500)
