import matplotlib.pyplot as plt
import math
import os
import re
import sys
dir=os.getcwd()
dir_list=dir.split("/")
loc=[i for i in range(0, len(dir_list)) if dir_list[i]=="General_electrochemistry"]
source_list=dir_list[:loc[0]+1] + ["src"]
source_loc=("/").join(source_list)
sys.path.append(source_loc)
print(sys.path)
from single_e_class_unified import single_electron
from EIS_class import EIS
import pints
import numpy as np
import matplotlib.pyplot as plt
import pandas
class ComposedProblem(object):

    def __init__(self, model, times, values , measurements_per_time=1):

        # Check model
        self._model = model

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.matrix2d(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
       
        # Check values, copy so that they can no longer be changed
        self._values = pints.matrix2d(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times[:,0])
        for i in range(0, len(times)):
            shape=values.shape[0]

      

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.

        The returned data is a NumPy array with shape ``(n_times, n_outputs)``.
        """
        y = np.asarray(self._model.simulate(parameters, self._times))
        return y.reshape(self._n_times, self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs
    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters
    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times
    def times(self):
        """
        Returns this problem's times.

        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times
    def values(self):
        """
        Returns this problem's values.

        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values

data_loc="/home/henryll/Documents/Experimental_data/Alice/Galectin_5_4/"
directories=os.listdir(data_loc)
header=2
footer=1
boundaries={"R0":[0, 1e4,],
            "R1":[1e-6, 1e6], 
            "C2":[0,1],
            "Q2":[0,2], 
            "alpha2":[0,1],
            "Q1":[0,2], 
            "alpha1":[0,1],
            "W1":[0,1e6]}
class composedwrapper(EIS):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
for directory in directories:
    if "GFF" in directory:
        new_loc=data_loc+directory+"/"
        files=os.listdir(new_loc)
        conc_list=[]

        for i in range(0, len(files)):
            if ".xlsx" in files[i]:
                removed=files[i].split(".")
                curr_file=("").join(removed[:-1])
                id_list=curr_file.split("_")
                conc_idx=id_list.index("SPE")+2
                decimal_idx=conc_idx+1
                try:
                    decimal=int(id_list[decimal_idx])*0.1
                    conc_list.append((int(id_list[conc_idx])+decimal,i))
                except:
                    conc_list.append((int(id_list[conc_idx]),i))
        concs_only=[x[0] for x in conc_list]
        sort_index = np.argsort(concs_only)
        sorted_concs=[concs_only[x] for x in sort_index]
        file_list=[files[conc_list[x][1]] for x in sort_index]
        sim_class=composedwrapper(circuit={"z1":"R0", "z2":{"p_1":("Q1", "alpha1"),"p_2":["R1", "W1"]}}, fitting=True, parameter_bounds=boundaries, normalise=True)
        
        for i in range(0, len(file_list)):
          
            name=file_list[i]
            associated_conc=sorted_concs[i]
            pd_data=pandas.read_excel(new_loc+name, skiprows=header, skipfooter=footer)
            data=pd_data.to_numpy(copy=True, dtype='float')
            if i==0:
                composed_data=[]#np.zeros((len(data[:,0]), len(file_list)*2))
                frequencies=[]#np.zeros((len(data[:,0]), len(file_list)))
            spectra=np.column_stack((np.flip(data[:,3]), -np.flip(data[:,4])))
            bode=sim_class.convert_to_bode(spectra)
            composed_data.append(bode)##[:,i*2:(i+1)*2]=
            frequencies.append(np.flip(data[:,0])*2*np.pi)
           