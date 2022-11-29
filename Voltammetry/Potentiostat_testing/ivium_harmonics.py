import os
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
import numpy as np
import matplotlib.pyplot as plt
from harmonics_plotter import harmonics
data_loc="Experimental_data/14_11_22_testing"
import sqlite3 as sql
print(os.listdir(data_loc))
files=['200_cycles_FTV.sqlite', '2_hz_approx_FTV.sqlite']
connect=sql.connect(data_loc+"/"+files[0])
cur=connect.cursor()

sql_obj=cur.execute('SELECT t, y, z from point')
data=sql_obj.fetchall()
connect.close()
time=[elem[0] for elem in data]
potential=[elem[2] for elem in data]
current=[elem[1] for elem in data]
y=np.fft.fft(current)
fft_freq=np.fft.fftfreq(len(current), time[1]-time[0])
plt.plot(fft_freq, np.log10(np.abs(y)))
plt.plot(time, current)
plt.show()
harms=harmonics(list(range(0,3)), 20, 0.5)
harms.plot_harmonics(time, experimental_time_series=current, plot_func=abs, hanning=True)
print(time[1]-time[0]/len(current))
plt.show()