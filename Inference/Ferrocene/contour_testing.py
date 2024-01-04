import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1234)  # fix seed for reproducibility
x = np.random.rand(100)
y = np.random.rand(100)
z = np.sin(x)+np.cos(y)
f, ax = plt.subplots(1,2, sharex=True, sharey=True, clear=True)
for axes, shading in zip(ax, ['flat', 'gouraud']):
    axes.tripcolor(x,y,z, shading=shading)
    axes.plot(x,y, 'k.')
    axes.set_title(shading)
plt.show()