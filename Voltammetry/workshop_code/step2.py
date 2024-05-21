##########STEP1#############
import numpy as np
data_location="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_3"
current_file=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_current")
potential_file=current=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_voltage")
current=current_file[:,1]
time=current_file[:,0]
potential=potential_file[:,1]
#########STEP2###############
import matplotlib.pyplot as plt
plt.plot(time, current)
plt.show()
plt.plot(time, potential)
plt.show()
plt.plot(potential, current)
plt.show()