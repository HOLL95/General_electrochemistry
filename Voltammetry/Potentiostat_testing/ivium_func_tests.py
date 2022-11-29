import numpy as np
import matplotlib.pyplot as plt
import math
x=np.linspace(0, 1, 62000)
y=(0.5-np.abs(x-0.5))+0.1*np.sin(200*2*math.pi*x)-0.5
plt.plot(x, y)
plt.show()