import numpy as np
data_location="/home/henryll/Documents/Experimental_data/Nat/Dummypaper/Figure_3"
current_file=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_current")
potential_file=current=np.loadtxt(data_location+"/FTacV_Monash_+Fc_72_Hz_export_cv_potential")
current=current_file[:,1]
time=current_file[:,0]
potential=potential_file[:,1]