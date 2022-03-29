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
            harmonics[i,:]=((np.fft.ifft(harmonics[i,:])))
        self.f=f
        self.Y=Y
        if kwargs["return_amps"]==True:
            return harmonics, amps
        else:
            return harmonics
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
        end_time=int(times[-1]//1)-1
        start_time=3

        for i in range(start_time, end_time):
            data_plot=data[np.where((times>=i) & (times<(i+1)))]
            time_plot=np.linspace(0, 1, len(data_plot))
            if i==start_time:
                line=kwargs["ax"].plot(time_plot, data_plot, color=kwargs["colour"], label=kwargs["label"], alpha=kwargs["alpha"])
            else:
                line=kwargs["ax"].plot(time_plot, data_plot, color=kwargs["colour"], alpha=kwargs["alpha"])
        return line
    def inv_objective_fun(self, time_series, dt=None,func=None):
        func_not_right_length=False
        if func!=None:
            likelihood=func(time_series)
            if len(likelihood)==(len(time_series)//2)-1:
                likelihood=np.append(likelihood, [0, np.flip(likelihood)])
            if len(likelihood)!=len(time_series):
                func_not_right_length=True
        if func==None or func_not_right_length==True:
            L=len(time_series)
            window=np.hanning(L)
            #time_series=np.multiply(time_series, window)
            f=np.fft.fftfreq(len(time_series), dt)
            Y=np.fft.fft(time_series)
            top_hat=copy.deepcopy(Y)
            first_harm=(self.harmonics[0]*self.input_frequency)-(self.input_frequency*0.5)
            last_harm=(self.harmonics[-1]*self.input_frequency)+(self.input_frequency*0.5)
            Y[np.where((f<(first_harm-(self.input_frequency*self.filter_val))) & (f>last_harm+(self.input_frequency*self.filter_val)))]=0
            likelihood=Y
        time_domain=np.fft.ifft(likelihood)
        return time_domain
    def harmonic_selecter(self, ax, time_series, times, box=True, arg=np.real, line_label=None, alpha=1.0, extend=False):
        f=np.fft.fftfreq(len(time_series), times[1]-times[0])
        hann=np.hanning(len(time_series))
        time_series=np.multiply(time_series, hann)
        Y=np.fft.fft(time_series)
        last_harm=(5+self.harmonics[-1])*self.input_frequency
        first_harm=self.harmonics[0]*self.input_frequency
        frequencies=f[np.where((f>=0 )& (f<(last_harm+(0.5*self.input_frequency))))]
        fft_plot=Y[np.where((f>=0 )& (f<(last_harm+(0.5*self.input_frequency))))]
        ax.semilogy(frequencies, abs(arg(fft_plot)), label=line_label, alpha=alpha)
        if box==True:
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


    def harmonics_plus(self, title, method, times, **kwargs):
        plt.rcParams.update({'font.size': 12})
        large_plot_xaxis=times
        fig=plt.figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')
        if method=="abs":
            a=abs
        else:
            a=self.empty
        time_list=[]
        titles=[]
        for key, value in list(kwargs.items()):
            if key=="voltage":
                large_plot_xaxis=voltages
                continue
            time_list.append(value)
            titles.append(key)

        title_lower=[x.lower() for x in titles]
        exp_idx=title_lower.index("experimental")
        new_order=list(range(0, len(titles)))
        new_order[0]=exp_idx
        new_order[exp_idx]=0
        titles=[titles[x] for x in new_order]
        time_list=[time_list[x] for x in new_order]
        harmonics_list=[]
        print("~"*50)
        for i in range(0, len(time_list)):
            harms=self.generate_harmonics(times, time_list[i])
            harmonics_list.append(harms)
        harm_axes=[]
        harm_len=2
        fig.text(0.03, 0.5, 'Current($\mu$A)', ha='center', va='center', rotation='vertical')

        for i in range(0,self.num_harmonics):
            harm_axes.append(plt.subplot2grid((self.num_harmonics,harm_len*2), (i,0), colspan=harm_len))
            for q in range(0, len(titles)):
                harm_axes[i].plot(times, np.multiply(a(harmonics_list[q][i,:]), 1e6), label=titles[q])
            harm_axes[i].yaxis.set_label_position("right")
            harm_axes[i].set_ylabel(str(self.harmonics[i]), rotation=0)
        harm_axes[i].legend()
        harm_axes[i].set_xlabel("Time(s)")

        time_ax=plt.subplot2grid((self.num_harmonics,harm_len*2), (0,harm_len), rowspan=self.num_harmonics, colspan=harm_len)
        for p in range(0, len(titles)):
            if titles[p].lower()=="experimental":
                time_ax.plot(large_plot_xaxis, np.multiply(time_list[p], 1e3), label=titles[p], alpha=1.0)
            else:
                time_ax.plot(large_plot_xaxis, np.multiply(time_list[p], 1e3), label=titles[p], alpha=0.5)
        time_ax.set_ylabel("Current(mA)")
        time_ax.set_xlabel("Time(s)")
        plt.legend()
        plt.suptitle(title)
        plt.subplots_adjust(left=0.08, bottom=0.09, right=0.95, top=0.92, wspace=0.23)
        plt.show()
    def plot_harmonics(self, times, **kwargs):
        label_list=[]
        time_series_dict={}
        harm_dict={}
        if "hanning" not in kwargs:
            kwargs["hanning"]=False
        if "xaxis" not in kwargs:
            kwargs["xaxis"]=times
        if "alpha_increment" not in kwargs:
            kwargs["alpha_increment"]=0
        if "plot_func" not in kwargs:
            kwargs["plot_func"]=np.real
        if "fft_func" not in kwargs:
            kwargs["fft_func"]=None
        if "xlabel" not in kwargs:
            kwargs["xlabel"]=""
        if "ylabel" not in kwargs:
            kwargs["ylabel"]=""

        if "legend" not in kwargs:
            kwargs["legend"]={"loc":"center"}
        if "axes_list" not in kwargs:
            define_axes=True
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
        for i in range(0, num_harms):
            if define_axes==True:
                plt.subplot(num_harms, 1,i+1)
                ax=plt.gca()
            else:
                ax=kwargs["axes_list"][i]

            ax2=ax.twinx()
            ax2.set_yticks([])
            ax2.set_ylabel(self.harmonics[i], rotation=0)
            plot_counter=0
            for plot_name in label_list:
                if i==0:
                    print(plot_name)
                    ax.plot(kwargs["xaxis"], kwargs["plot_func"](harm_dict[plot_name][i,:]), label=plot_name, alpha=1-(plot_counter*kwargs["alpha_increment"]))
                else:
                    ax.plot(kwargs["xaxis"], kwargs["plot_func"](harm_dict[plot_name][i,:]),  alpha=1-(plot_counter*kwargs["alpha_increment"]))
                plot_counter+=1
            if i==((num_harms)//2):
                ax.set_ylabel(kwargs["ylabel"])
            if i==num_harms-1:
                ax.set_xlabel(kwargs["xlabel"])
            if i==0:
                if kwargs["legend"] is not None:
                    ax.legend(**kwargs["legend"])
