import numpy as np
import matplotlib.pyplot as plt
#results_dict={"minima":minima, "vals":{"k_0":k0_vals, "alpha":alpha_vals}}
results_dict=np.load("Minima_location.npy", allow_pickle=True).item()
minima=results_dict["minima"]
k0_vals=results_dict["vals"]["k_0"]
alpha_vals=results_dict["vals"]["alpha"]

for i in range(0, len(results_dict["minima"])):
    plt.scatter(alpha_vals, minima[i, :], label=("$k_0$= %.1f s$^{-1}$" % k0_vals[i]))
plt.ylabel("Potential offset (V)")
plt.xlabel("$\\alpha$")
plt.legend(ncol=4)
fig=plt.gcf()
plt.show() 


fig.savefig("minima_loc.png", dpi=500)