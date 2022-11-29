import numpy as np
import matplotlib.pyplot as plt
max_iv_points=2048
min_bio_sample_freq=1/0.0002
desired_points=200
def end_time(e_start, e_reverse, v):
    return (2*(e_reverse-e_start))/v
def num_oscillations(end_time, freq):
    return end_time*freq
def num_points(num_freqs, sf):
    return num_freqs*sf
scan_rates=[1e-3*10**x for x in np.linspace(0, 2.5, 1000)]
e_start=-0.5
e_reverse=0.5
times=[end_time(e_start, e_reverse, x) for x in scan_rates]
#print(times)
#print(min_bio_sample_freq)
max_freq=np.divide(max_iv_points, np.multiply(times, desired_points))
plot_rates=np.multiply(scan_rates, 1000)

plt.plot(plot_rates, max_freq, label="Ivium")
plt.title("FTACV")
plt.ylabel("Maximum frequency @ {0} points/cycle (Hz)".format(desired_points))
plt.xlabel("Scan rate (mV)")
plt.plot(plot_rates, [min_bio_sample_freq/desired_points]*len(plot_rates), label="Biologic")
plt.legend()
plt.show()
plt.subplot(1,2,2)
ivium_sample_freq=1/10e-6
plt.title("PSV")
plt.ylabel("Maximum frequency @ {0} points/cycle (Hz)".format(desired_points))
plt.bar(["Ivium", "Biologic"], [ivium_sample_freq/desired_points, min_bio_sample_freq/desired_points])
plt.show()
frequencies=np.linspace(0.1, 100, 1000)
sampling_rate=0.0002
plt.subplot(1,2,1)
e_start=-0.5
e_reverse=0.5
time=end_time(e_start, e_reverse, 22.5e-3)

signal_len_FTV=int(time*min_bio_sample_freq)

freqs=np.arange(10, 1000, 10)
accessible_harms=np.zeros(len(freqs))
dt=0.0002

for j in range(0, len(freqs)):
   
    signal_len=signal_len_FTV
    fft_freq=np.fft.fftfreq(signal_len, dt)
    accessible_harms[j]=abs(fft_freq[signal_len//2])//(freqs[j])
plt.subplot(1,2,1)
plt.plot(freqs, accessible_harms)
plt.axhline(6, linestyle="--", color="black")
plt.xlabel("Frequency")
plt.ylabel("Maximum accessible harmonic")
plt.subplot(1,2,2)
plt.plot(freqs, np.divide(min_bio_sample_freq, freqs))
plt.xlabel("Frequency")
plt.ylabel("Points per cycle")
plt.tight_layout()

plt.show()
