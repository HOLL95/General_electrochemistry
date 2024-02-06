from multiplotter import multiplot
import matplotlib.pyplot as plt
figure=multiplot(3, 4, **{"harmonic_position":2, "num_harmonics":7, "orientation":"landscape", "fourier_position":1, "plot_width":5, "row_spacing":2, "plot_height":1})
plt.show()
