import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
titles=["E_0","k0_shape", "k0_scale", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl"]+["score", "best"]
log_list=[False, False, True, True, True, False, True, False]
params=titles[:-2]

param_bounds={
    'E_0':[0.239, 0.5],
    "E0_mean":[0.239, 0.241],
    "E0_std":[1e-5, 0.05],
    'Ru': [0, 1e3],  #     (uncompensated resistance ohms)
    'Cdl': [0,1e-2], #(capacitance parameters)
    'Cfarad': [0,1], #(capacitance parameters)
    'CdlE1': [-1e-2,1e-2],#0.000653657774506,
    'CdlE2': [-5e-4,5e-4],#0.000245772700637,
    'CdlE3': [-1e-4,1e-4],#1.10053945995e-06,
    'gamma': [1e-8,1e-7],
    'k_0': [1e-9, 2e3], #(reaction rate s-1)
    'alpha': [0.35, 0.65],
    "dcv_sep":[0, 0.5],
    "cpe_alpha_faradaic":[0,1],
    "cpe_alpha_cdl":[0.65,1],
    "k0_shape":[0,10],
    "k0_scale":[0,200],
    "phase":[-180, 180],
}
comb=list(combinations(titles[:-2], 2))
idxs=dict(zip(params, range(0, len(params))))
log_dict=dict(zip(params, log_list))
npoints=50
import matplotlib.tri as tri
from matplotlib import cm
from scipy.interpolate import griddata
step=20
warmcool=cm.get_cmap("coolwarm_r")
for j in range(6, len(comb)):
    param1, param2=comb[j]
    idx1=idxs[param1]
    idx2=idxs[param2]
    min1=param_bounds[param1][0]*1.1
    min2=param_bounds[param2][0]*1.1
    max1=param_bounds[param1][1]*0.9
    max2=param_bounds[param2][1]*0.9
    xi = np.linspace(param_bounds[param1][0], param_bounds[param1][1], npoints)
    yi = np.linspace(param_bounds[param2][0], param_bounds[param2][1], npoints)
    fig,ax=plt.subplots()
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    for i in range(0,10):
        file_name="traces/Profile_trace_%d_k0_disp.csv" % (i+1)
        data=np.loadtxt(file_name, skiprows=1, delimiter=",")
        
        if i==0:
            x=data[:,idx1]
            y=data[:,idx2]
            z=np.log10(data[:,-2])
        else:
            x=np.append(x, data[:,idx1])
            y=np.append(y, data[:,idx2])
            z=np.append(z, np.log10(data[:,-2]))
        
        if i<3:
            best_name="traces/Profile_trace_%d_k0_disp_best.csv" % (i+1)
            best_data=np.loadtxt(file_name, skiprows=1, delimiter=",")
            best_x=best_data[:,idx1]
            best_y=best_data[:,idx2]
            best_z=np.log10(best_data[:,-1])
            threshold=np.where(best_z<1.7*min(best_z))
            ax.scatter(best_x[threshold][0::step], best_y[threshold][0::step], marker="x")
    #ax.tripcolor(np.log10(x), np.log10(y), z, shading='gouraud')
    ax.scatter(x,y, c=z, cmap=warmcool, s=0.5)
    if log_dict[param1]==True:
        ax.set_xscale('log')
    if log_dict[param2]==True:
        ax.set_yscale('log')
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.show()
    #isBad = np.where((x<min1) | (x>max1) | (y<min2) | (y>max2), True, False)
    #triang = tri.Triangulation(y, x)
    #mask = np.any(isBad[triang.triangles],axis=1)
    #triang.set_mask(mask)
    #ax.triplot(triang, c="#D3D3D3", marker='.', markerfacecolor="#DC143C",
    #            markeredgecolor="black")
    #ax.scatter(x, y, s=1)
    #plt.show()
    #triang = tri.Triangulation(x, y)
    #xy = np.dstack((triang.x[triang.triangles], triang.y[triang.triangles])) #
    #twice_area = np.cross(xy[:,1,:] - xy[:,0,:], xy[:,2,:] - xy[:,0,:]) #
    #mask = twice_area < 1e-10 # shape (ntri)
    #if np.any(mask):
    #    triang.set_mask(mask)
    #interpolator = tri.LinearTriInterpolator(triang, z)
    #Xi, Yi = np.meshgrid(xi, yi)
    #zi = interpolator(Xi, Yi)
    #zi = griddata((x, y), z, (Xi[None, :], Yi[:, None]), method='linear')
    #print(zi)
    #surf = ax.plot_surface(Xi, Yi, zi, cmap=cm.coolwarm,
    #                   linewidth=0, antialiased=True)
   
    #ax.scatter(x, y, z, s=0.1)                 
    #plt.show()