import numpy as np
from pints import plot, rhat

import matplotlib.pyplot as plt
#chains=np.load("Combined_initial_scaling_{0}".format(10))
chains=np.load("Combined_initial_long_13")
print(rhat(chains, warm_up=0.5))
chains[:,:,1]=np.subtract(chains[:, :, 1], 0.5)
plot.trace(chains)
save_file="Combined_initial_results"
f=open(save_file, "wb")
np.save(f, chains)
plt.show()



