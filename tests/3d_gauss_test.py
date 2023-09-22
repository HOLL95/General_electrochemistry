import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import multivariate_normal

length=50
mid=int(length//2)
mean = np.array([-0.1, 50, 0.5])
covariance_matrix = np.matrix(np.random.rand(3,3))*0.05
vals=[0.05, 30, 0.1]
for i in range(0, 3):
    covariance_matrix[i,i]=vals[i]
cov=covariance_matrix*covariance_matrix.H
x1=np.linspace(-0.13, -0.05, length)
x2=np.linspace(30, 70, length)
x3=np.linspace(0.4, 0.6, length)
x, y, z = np.meshgrid(x1,x2,x3)
pos = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
pdf_values = multivariate_normal.pdf(pos, mean=mean, cov=cov)
print(pdf_values)
pdf_values = pdf_values.reshape(length, length, length)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_flat, y_flat, z_flat = x.flatten(), y.flatten(), z.flatten()
pdf_flat = pdf_values.flatten()
threshold=0.9*np.max(pdf_flat)
inside_sphere = pdf_flat > threshold
ax.set_xlabel("X")
#ax.set_xlim([-0.15, -0.05])
#ax.set_ylim([10, 90])
#ax.set_zlim([0.4, 0.6])
ax.set_ylabel("Y")
ax.set_zlabel("Z")
from matplotlib import cm
step=1
ax.scatter(x_flat[inside_sphere][::step], y_flat[inside_sphere][::step], z_flat[inside_sphere][::step], c=pdf_flat[inside_sphere][::step], marker='.',s=0.1, alpha=0.025, cmap=cm.viridis)
z_plot=pdf_values[mid, :,:]
z_plots=[pdf_values[:, :,mid],pdf_values[mid, :,:].T,pdf_values[:, mid,:].T]
from itertools import combinations
axes=list(combinations([x1, x2, x3],2))
for i in range(0, 3):
    current_z=z_plots[i].flatten()
    inside_sphere = current_z > threshold
    x_axis,y_axis=np.meshgrid(axes[i][0], axes[i][1])
    flatx=x_axis.flatten()
    flaty=y_axis.flatten()
    if i==0:#xE0y#k0
        x_arg=flatx[inside_sphere]
        
        y_arg=flaty[inside_sphere]
        z_arg=(np.ones(length**2)*x3[0])[inside_sphere]
    elif i==1:#xE0yalpha
        x_arg=flatx[inside_sphere]
        
        y_arg=(np.ones(length**2)*x2[-1])[inside_sphere]
        z_arg=flaty[inside_sphere]
    elif i==2:#xk0yalpha
        x_arg=(np.ones(length**2)*x1[0])[inside_sphere]
        y_arg=flatx[inside_sphere]
        z_arg=flaty[inside_sphere]

    
    

    ax.scatter(x_arg, y_arg, z_arg,c=current_z[inside_sphere], alpha=1, cmap=cm.coolwarm, s=3)

plt.show()

