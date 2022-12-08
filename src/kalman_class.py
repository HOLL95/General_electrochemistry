import os
import sys
import math
import copy
import pints
from single_e_class_unified import single_electron
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
class Kalman(single_electron):
    def __init__(self, dim_parameter_dictionary, simulation_options, other_values, param_bounds):
        
        super().__init__("", dim_parameter_dictionary, simulation_options, other_values, param_bounds)
     def Kalman_capacitance(self, q, error=0.005):
        #error=0.005*max(self.secret_data_time_series)
        my_filter = KalmanFilter(dim_x=2, dim_z=1)
        my_filter.x = np.array([[self.secret_data_time_series[0]],
                        [1.]])       # initial state (location and velocity)
        dt=self.time_vec[1]-self.time_vec[0]
        cdl=self.nd_param.nd_param_dict["Cdl"]
        ru=self.nd_param.nd_param_dict["Ru"]
        dt_rc=dt/(ru*cdl)
        my_filter.F=np.array([[1/(1+dt_rc), (dt/ru)/(1+dt_rc)],[0, 0]])  # state transition matrix
        tr=self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]
        u=np.ones(len(self.time_vec))
        u[np.where(self.time_vec>tr)]*=-1
        my_filter.H = np.array([[1.,0.]])    # Measurement function
        my_filter.P *= error      # covariance matrix
        my_filter.R = error                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(2, dt, q) # process uncertainty
        my_filter.B=np.array([[0], [1]])
        means, _, _, _=my_filter.batch_filter(self.secret_data_time_series, Fs=None, Qs=None, Hs=None, Bs=None, us=u)
        return means[:, 0, 0]
    def kalman_pure_capacitance(self, current, q):
        my_filter = KalmanFilter(dim_x=2, dim_z=1)
        my_filter.x = np.array([[self.nd_param.nd_param_dict["Cdl"]],
                        [0]])       # initial state (location and velocity)
        dt=self.time_vec[1]-self.time_vec[0]

        my_filter.F=np.array([[1, dt],[0, 1]])  # state transition matrix
        tr=self.nd_param.nd_param_dict["E_reverse"]-self.nd_param.nd_param_dict["E_start"]
        u=np.ones(len(self.time_vec))
        ru=self.nd_param.nd_param_dict["Ru"]
        if self.simulation_options["method"]=="dcv":
            u[np.where(self.time_vec>tr)]*=-1
            self.kalman_u=u
            pred_cap=self.kalman_u*current
        elif self.simulation_options["method"]=="sinusoidal":
            orignal_phase=self.nd_param.nd_param_dict["phase"]
            self.nd_param.nd_param_dict["phase"]=self.nd_param.nd_param_dict["cap_phase"]
            for i in range(0, len(self.time_vec)):
                u[i]=(self.voltage_query(self.time_vec[i])[1])
            self.kalman_u=u
            pred_cap=np.divide((current), u)
        norm_denom=np.zeros(len(current))
        norm_denom[0]=u[0]

        """
        for i in range(1, len(self.secret_data_time_series)):
            gradient=(self.secret_data_time_series[i]-self.secret_data_time_series[i-1])/dt
            norm_denom[i]=u[i]-ru*gradient
            norm=np.divide(1, norm_denom)
        """
        #H_array=[np.array([[1, 0]]) for x in range(0, len(norm_denom))]
        my_filter.H = np.array([[1.,0.]])
        my_filter.P *= 0.1      # covariance matrix
        my_filter.R = 1                      # state uncertainty
        my_filter.Q = Q_discrete_white_noise(2, dt, q) # process uncertainty
        means=np.zeros(len(self.secret_data_time_series))
        for i in range(0, len(self.time_vec)):
            my_filter.predict()
            my_filter.update(z=pred_cap[i], R=0.1)
            means[i]=my_filter.x[0][0]

        means=np.divide(means, self.kalman_u)
        return pred_cap, means
    def kalman_dcv_simulate(self, Faradaic_current, q):

        candidate_current=np.subtract(self.secret_data_time_series, Faradaic_current)
        predicted_capcitance, predicted_current=self.kalman_pure_capacitance(candidate_current, q)
        if self.simulation_options["Kalman_capacitance"]==True:
            print([(x, self.dim_dict[x]) for x in self.optim_list])
            self.pred_cap=predicted_current
            self.farad_current=Faradaic_current
        return np.add(predicted_current, Faradaic_current)