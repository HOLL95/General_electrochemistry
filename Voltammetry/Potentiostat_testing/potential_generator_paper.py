import numpy as np
###########################USER VALUES############################################################
min_interval=1/5000 #Delta T in seconds. The minimum value of this will be defined by the intrument
desired_Hz=10 #Input frequency in Hz
desired_scan_rate=22.5e-3 # Scan rate in V/s
E_reverse=0.5 # Switching potential in V
E_start=-0.2 # Start potential in V
phase=0 # Phase of the input sinusoid
strictly_periodic=True # For various reasons, you may want to enforce periodicity of your sine wave. 
Ac_amplitude=0.15 # Amplitude of the sine wave in volts. Check with the manufacturer as to how they define this value
#################################################################################################

phase_rad=phase*(np.pi/180)
distance=(E_reverse-E_start)
DC_range=distance*2
end_time=DC_range/desired_scan_rate
if strictly_periodic==True:
    end_time=end_time-(end_time%(1/desired_Hz))
    actual_scan_rate=distance/(end_time/2)
else:
    actual_scan_rate=desired_scan_rate
num_points=int(end_time//min_interval)+1

t=np.linspace(0, end_time, int(num_points))
switch_time=t[len(t)//2]
E=E_reverse-(actual_scan_rate*np.abs(t-switch_time))+Ac_amplitude*np.sin((desired_Hz*2*np.pi*t)+phase_rad)
with open("Potential_values.txt", "w") as f:
    np.savetxt( f,np.column_stack((t, E)))
    