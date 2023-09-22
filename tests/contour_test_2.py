from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D
import pylab as p


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X = np.arange(0, 2.5, 0.1)
Y = np.arange(0, 2.5, 0.1)
X, Y = np.meshgrid(X, Y)

Z = ((X+2))*(Y**2)
surf = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, alpha=0.3, cmap=cm.jet)
cset=plt.contour(X,Y,Z,zdir='x',offset=0)


ax.clabel(cset, fontsize=9, inline=1)
ax.set_zlim3d(0, 30)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()