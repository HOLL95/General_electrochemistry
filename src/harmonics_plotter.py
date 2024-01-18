import numpy as np
import copy
import matplotlib.pyplot as plt
class harmonics:
    def __init__(self, harmonics, input_frequency, filter_val):
        self.harmonics=harmonics
        self.num_harmonics=len(harmonics)
        self.input_frequency=input_frequency
        self.filter_val=filter_val
    def reorder(list, order):
        return [list[i] for i in order]
    def plot_ffts(self, time, current, **kwargs):
        if "ax" not in kwargs:
            _, ax=plt.subplots(1,1)
        else:
            ax=kwargs["ax"]
        if "harmonics" not in kwargs:
            harmonics=list(range(0, self.harmonics[-1]))
        else:
            harmonics=kwargs["harmonics"]
        if "colour" not in kwargs:
            colour=None
        else:
            colour=kwargs["colour"]
        if "label" not in kwargs:
            kwargs["label"]=None
        if "log" not in kwargs:
            kwargs["log"]=True
        if "plot_func" not in kwargs:
            kwargs["plot_func"]=abs
        fft=np.fft.fft(current)
        fft_freq=np.fft.fftfreq(len(current), time[1]-time[0])
        freq_idx=np.where((fft_freq>(self.input_frequency*harmonics[0])) & (fft_freq<(self.input_frequency*harmonics[-1])))
        if kwargs["log"]==True:
            ax.semilogy(fft_freq[freq_idx], kwargs["plot_func"](fft[freq_idx]), label=kwargs["label"], color=colour)
        else:
            ax.plot(fft_freq[freq_idx], kwargs["plot_func"](fft[freq_idx]), label=kwargs["label"],color=colour)
        if kwargs["label"] is not None:
            ax.legend()
    def generate_harmonics(self, times, data, **kwargs):
        if "func" not in kwargs or kwargs["func"]==None:
            func=self.empty
        else:
            func=kwargs["func"]
        if "zero_shift" not in kwargs:
            kwargs["zero_shift"]=False
        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "return_amps" not in kwargs:
            kwargs["return_amps"]=False
        if kwargs["return_amps"]==True:
            amps=np.zeros(self.num_harmonics)
        if "return_fourier" not in kwargs:
            kwargs["return_fourier"]=False
        L=len(data)
        if kwargs["hanning"]==True:
            window=np.hanning(L)
            time_series=np.multiply(data, window)
        else:
            time_series=data
        f=np.fft.fftfreq(len(time_series), times[1]-times[0])
        Y=np.fft.fft(time_series)
        last_harm=(self.harmonics[-1]*self.input_frequency)
        frequencies=f[np.where((f>0) & (f<(last_harm+(0.5*self.input_frequency))))]
        top_hat=(copy.deepcopy(Y[0:len(frequencies)]))
        
        harmonics=np.zeros((self.num_harmonics, len(time_series)), dtype="complex")
        if kwargs["return_fourier"]==True:
            ft_peak_return= harmonics=np.zeros((self.num_harmonics, len(frequencies)), dtype="complex")
        for i in range(0, self.num_harmonics):
            true_harm=self.harmonics[i]*self.input_frequency
            #plt.axvline(true_harm, color="black")
            freq_idx=np.where((frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (frequencies>true_harm-(self.input_frequency*self.filter_val)))
            filter_bit=(top_hat[freq_idx])
            if kwargs["return_amps"]==True:

                abs_bit=abs(filter_bit)
                #print(np.real(filter_bit[np.where(abs_bit==max(abs_bit))]))
                amps[i]=np.real(filter_bit[np.where(abs_bit==max(abs_bit))])
            if kwargs["zero_shift"]==True:
                harmonics[i,0:len(filter_bit)]=func(filter_bit)
            else:
                harmonics[i,np.where((frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (frequencies>true_harm-(self.input_frequency*self.filter_val)))]=func(filter_bit)
            #harmonics[i,0:len(filter_bit)]=func(filter_bit)
            if kwargs["return_fourier"]==False:
                harmonics[i,:]=((np.fft.ifft(harmonics[i,:])))
            else:
                ft_peak_return[i,:]=harmonics[i,0:len(frequencies)]
        self.f=f
        self.Y=Y
        if kwargs["return_amps"]==True:
            return harmonics, amps
        if kwargs["return_fourier"]==False:
            return harmonics
        else:
            return ft_peak_return, frequencies
    def empty(self, arg):
        return arg
    def single_oscillation_plot(self, times, data, **kwargs):
        if "colour" not in kwargs:
            kwargs["colour"]=None
        if "label" not in kwargs:
            kwargs["label"]=""
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        if "ax" not in kwargs:
            kwargs["ax"]=plt.subplots()
        if "start_time" not in kwargs:
            start_time=3
        else:
            start_time=kwargs["start_time"]
        if "end_time" not in kwargs:
            end_time=int(times[-1]//1)-1
        else:
            end_time=kwargs["end_time"]
        if isinstance(end_time, int) and isinstance(start_time, int):
            step=1
        else:
            if "oscillation_frequency" not in kwargs:
                raise ValueError("Need to define an oscillation_frequency")
            else:
                step=kwargs["oscillation_frequency"]
        full_range=np.arange(start_time, end_time, step)

        for i in range(0, len(full_range)-1):
            data_plot=data[np.where((times>=full_range[i]) & (times<(i+full_range[i+1])))]
            time_plot=np.linspace(0, 1, len(data_plot))
            if i==start_time:
                line=kwargs["ax"].plot(time_plot, data_plot, color=kwargs["colour"], label=kwargs["label"], alpha=kwargs["alpha"])
            else:
                line=kwargs["ax"].plot(time_plot, data_plot, color=kwargs["colour"], alpha=kwargs["alpha"])
        return line
    def inv_objective_fun(self, time_series,**kwargs):
        if "func" not in kwargs:
            func=None
        else:
            func=kwargs["func"]
        if "dt" not in kwargs:
            dt=None
        else:
            dt=kwargs["dt"]
        func_not_right_length=False
        if func!=None:
            likelihood=func(time_series)
            if len(likelihood)==(len(time_series)//2)-1:
                likelihood=np.append(likelihood, [0, np.flip(likelihood)])
            if len(likelihood)!=len(time_series):
                func_not_right_length=True
        if func==None or func_not_right_length==True:
            if dt==None:
                raise ValueError("To create the likelihood you need to give a dt")
            L=len(time_series)
            window=np.hanning(L)
            #time_series=np.multiply(time_series, window)
            f=np.fft.fftfreq(len(time_series), dt)
            Y=np.fft.fft(time_series)
            top_hat=copy.deepcopy(Y)
            first_harm=(self.harmonics[0]*self.input_frequency)-(self.input_frequency*0.5)
            last_harm=(self.harmonics[-1]*self.input_frequency)+(self.input_frequency*0.5)
            print(first_harm, last_harm)
            abs_f=np.abs(f)
            Y[np.where((abs_f<(first_harm)) | (abs_f>last_harm))]=0
            likelihood=Y
        time_domain=np.fft.ifft(likelihood)
        return time_domain
    def harmonic_selecter(self, ax, time_series, times,**kwargs): 
        if "box" not in kwargs:
            kwargs["box"]=True
        if "arg" not in kwargs:
            arg=np.real
        else:
            arg=kwargs["arg"]
        if "label" not in kwargs:
            kwargs["label"]=None
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        if "log" not in kwargs:
            kwargs["log"]=True
        if "extend" not in kwargs:
            kwargs["extend"]=0
        f=np.fft.fftfreq(len(time_series), times[1]-times[0])
        hann=np.hanning(len(time_series))
        time_series=np.multiply(time_series, hann)
        Y=np.fft.fft(time_series)
        last_harm=(kwargs["extend"]+self.harmonics[-1])*self.input_frequency
        first_harm=self.harmonics[0]*self.input_frequency

        frequencies=f[np.where((f>=0 )& (f<(last_harm+(0.5*self.input_frequency))))]
        fft_plot=arg(Y[np.where((f>=0 )& (f<(last_harm+(0.5*self.input_frequency))))])
        if kwargs["log"]==True:
            ax.semilogy(frequencies, fft_plot, label=kwargs["label"], alpha=kwargs["alpha"])
        else:
            ax.plot(frequencies, fft_plot, label=kwargs["label"], alpha=kwargs["alpha"])
        if kwargs["box"]==True:
            len_freq=np.linspace(0, 100, len(frequencies))
            longer_len_freq=np.linspace(0, 100, 10000)
            extended_frequencies=np.interp(longer_len_freq, len_freq, frequencies)
            box_area=np.zeros(len(extended_frequencies))
            for i in range(0, self.num_harmonics):
                true_harm=self.harmonics[i]*self.input_frequency
                peak_idx=np.where((frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (frequencies>true_harm-(self.input_frequency*self.filter_val)))
                extended_peak_idx=np.where((extended_frequencies<(true_harm+(self.input_frequency*self.filter_val))) & (extended_frequencies>true_harm-(self.input_frequency*self.filter_val)))
                box_area[extended_peak_idx]=max(fft_plot[peak_idx])
                
            ax.plot(extended_frequencies, box_area, color="r", linestyle="--")
            if self.harmonics[0]==0:
                print(box_area[0])
                ax.plot([0,0], [0, box_area[0]],color="r", linestyle="--")
    def plot_harmonics(self, times, **kwargs):
        label_list=[]
        time_series_dict={}
        harm_dict={}
        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "xaxis" not in kwargs:
            kwargs["xaxis"]=times
        if "plot_func" not in kwargs:
            kwargs["plot_func"]=np.real
        if "fft_func" not in kwargs:
            kwargs["fft_func"]=None
        if "xlabel" not in kwargs:
            kwargs["xlabel"]=""
        if "ylabel" not in kwargs:
            kwargs["ylabel"]=""
        if "DC_component" not in kwargs:
            kwargs["DC_component"]=False
        if "legend" not in kwargs:
            kwargs["legend"]={"loc":"center"}
        if "axes_list" not in kwargs:
            define_axes=True
        else:
            define_axes=False
        if "h_num" not in kwargs:
            kwargs["h_num"]=True
        if "colour" not in kwargs:
            kwargs["colour"]=None
        if "lw" not in kwargs:
            kwargs["lw"]=1
        if "alpha" not in kwargs:
            kwargs["alpha"]=1
        else:
            if len(kwargs["axes_list"])!=self.num_harmonics:
                raise ValueError("Wrong number of axes for harmonics")
            else:
                define_axes=False
        label_counter=0


        for key in kwargs:
            if "time_series" in key:
                index=key.find("time_series")
                if key[index-1]=="_" or key[index-1]=="-":
                    index-=1
                label_list.append(key[:index])

                time_series_dict[key[:index]]=kwargs[key]

                label_counter+=1

        if label_counter==0:
            return
        for label in label_list:
            harm_dict[label]=self.generate_harmonics(times, time_series_dict[label], hanning=kwargs["hanning"], func=kwargs["fft_func"])
        num_harms=self.num_harmonics
        if kwargs["DC_component"]==True:
            pot=kwargs["xaxis"]
            fft_pot=np.fft.fft(pot)
            fft_freq=np.fft.fftfreq(len(pot), times[1]-times[0])
            max_freq=self.input_frequency
            zero_harm_idx=np.where((fft_freq>-(0.5*max_freq)) & (fft_freq<(0.5*max_freq)))
            dc_pot=np.zeros(len(fft_pot), dtype="complex")
            dc_pot[zero_harm_idx]=fft_pot[zero_harm_idx]
            kwargs["xaxis"]=np.real(np.fft.ifft(dc_pot))
            self.dc_pot=kwargs["xaxis"]


        for i in range(0, num_harms):
            if define_axes==True:
                plt.subplot(num_harms, 1,i+1)
                ax=plt.gca()
            else:
                ax=kwargs["axes_list"][i]
            if kwargs["h_num"]!=False:
                ax2=ax.twinx()
                ax2.set_yticks([])
                ax2.set_ylabel(self.harmonics[i], rotation=0)
            plot_counter=0
            for plot_name in label_list:
                if i==0:
                    print(plot_name)
                    ax.plot(kwargs["xaxis"], kwargs["plot_func"](harm_dict[plot_name][i,:]), label=plot_name, alpha=kwargs["alpha"], color=kwargs["colour"], lw=kwargs["lw"])
                else:
                    ax.plot(kwargs["xaxis"], kwargs["plot_func"](harm_dict[plot_name][i,:]), alpha=kwargs["alpha"], color=kwargs["colour"], lw=kwargs["lw"])
                plot_counter+=1
            if i==((num_harms)//2):
                ax.set_ylabel(kwargs["ylabel"])
            if i==num_harms-1:
                ax.set_xlabel(kwargs["xlabel"])
            if i==0:
                if kwargs["legend"] is not None:
                    ax.legend(**kwargs["legend"])
