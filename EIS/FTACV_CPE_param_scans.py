import matplotlib.pyplot as plt
import numpy as np
data=np.load("FTACV_CPE_param_scans.npy", allow_pickle=True).item()
print(data.keys())
keys=list(data.keys())
fig, ax=plt.subplots(1, 3)
params=["psi", "Ru", "Cdl"]
for i in range(0, len(params)):
    plot_list=[]
    key_list=[]

    for key in keys:
        if params[i] in key:
            plot_list.append(data[key]["current"])
            key_list.append(key)
    for j in range(0, len(plot_list)):
        ax[i].plot(data[key]["time"], plot_list[len(plot_list)-j-1], label=key_list[len(plot_list)-j-1])
    ax[i].legend()
    ax[i].set_xlabel("Time(s)")
    ax[i].set_ylabel("Current(A)")
plt.show()
fig, ax =plt.subplots(3,3)
for i in range(0, len(keys)):
    key=keys[i]
    axes=ax[i//3, i%3]
    axes.plot(data[key]["time"], data[key]["current"], label=key)
    axes.plot(data[key]["time"], data[key]["numerical"])

    axes.set_xlabel("time(s)")
    axes.set_ylabel("Current(A)")
    axes.legend()
plt.show()
