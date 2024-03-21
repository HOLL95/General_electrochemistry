import matplotlib.pyplot as plt
from pints.plot import trace
import numpy as np
chains=np.load("/home/userfs/h/hll537/Documents/General_electrochemistry/Inference/Ella_LPMO/MCMC/Ella_set2_MCMC_50Ohms_Cj")
trace(chains)
plt.show()