fig, ax=plt.subplots(1,2)
ax[0].plot(log_freq, np.log10(abs_i[:len(ft_freq)//2+1]))
ax[0].axvline(np.log10(ft_freq[i_freq][0]), color="black", linestyle="--")

ax[1].plot(log_freq, np.log10(abs_v[:len(ft_freq)//2+1]))
ax[1].axvline(np.log10(ft_freq[v_freq][0]), color="black", linestyle="--")
plt.show()


i_fft=np.fft.fft(current)
v_fft=np.fft.fft(potential)


abs_i=abs(i_fft)
abs_v=abs(v_fft)
ft_freq=np.fft.fftfreq(len(current), dt)
i_freq=np.where(abs_i==max(abs_i))
v_freq=np.where(abs_v==max(abs_v))
i_peak=i_fft[i_freq]
v_peak=v_fft[v_freq]
z_freq=v_peak[0]/i_peak[0]
