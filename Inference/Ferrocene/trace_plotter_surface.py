import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
from scipy import interpolate
titles=["E_0","k0_shape", "k0_scale", "gamma", "Cdl", "alpha", "Ru", "cpe_alpha_cdl"]+["score", "best"]
log_list=[False, False, False, False, False, False, False, False]
params=titles[:-2]

param_bounds={
    'E_0':[0.239, 0.5],
    "E0_mean":[0.239, 0.241],
    "E0_std":[1e-5, 0.05],
    'Ru': [0, 200],  #     (uncompensated resistance ohms)
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
    "k0_shape":[0,6],
    "k0_scale":[0,120],
    "phase":[-180, 180],
}
comb=list(combinations(titles[:-2], 2))
idxs=dict(zip(params, range(0, len(params))))
log_dict=dict(zip(params, log_list))
npoints=100
step=20
import matplotlib.tri as tri


from matplotlib import cm
from scipy.interpolate import griddata
def empty(x):
    return np.array(x)
for j in range(6, len(comb)):
    param1, param2=comb[j]
    idx1=idxs[param1]
    idx2=idxs[param2]
    if log_dict[param1]==True:
        x_func=np.log10
    else:
        x_func=empty
    if log_dict[param2]==True:
        y_func=np.log10
    else:
        y_func=empty
    min1=param_bounds[param1][0]*1.05
    min2=param_bounds[param2][0]*1.05
    max1=param_bounds[param1][1]*0.95
    max2=param_bounds[param2][1]*0.95
    xi = np.linspace(x_func(param_bounds[param1][0]), x_func(param_bounds[param1][1]), npoints)
    yi = np.linspace(y_func(param_bounds[param2][0]), y_func(param_bounds[param2][1]), npoints)
    #fig,ax=plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(0,1):
        file_name="traces/Profile_trace_%d_k0_disp.csv" % (i+1)
        data=np.loadtxt(file_name, skiprows=1, delimiter=",")
        
        if i==0:
            x=x_func(data[:,idx1])
            y=y_func(data[:,idx2])
            z=np.log10(data[:,-2])
            best_name="traces/Profile_trace_%d_k0_disp_best.csv" % (i+1)
            best_data=np.loadtxt(file_name, skiprows=1, delimiter=",")
            best_x=best_data[:,idx1]
            best_y=best_data[:,idx2]
            best_z=np.log10(best_data[:,-1])
            threshold=np.where(best_z<2*min(best_z))
            ax.plot(best_x[threshold][0::step], best_y[threshold][0::step], best_z[threshold][0::step],marker="x")
        isBad = np.where((x<min1) | (x>max1) | (y<min2) | (y>max2), True, False)
        min_circle_ratio = .01
        subdiv=3
        triang = tri.Triangulation(x, y)
        #mask = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio)
        mask = np.any(isBad[triang.triangles],axis=1)
        #triang.set_mask(mask)

        # refining the data
        #refiner = tri.UniformTriRefiner(triang)
        #tri_refi, z_test_refi = refiner.refine_field(z_test, subdiv=subdiv)
        #
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
        try:
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)
            #zi = griddata((x, y), z, (Xi[None, :], Yi[:, None]), method='linear')
            #print(zi)
            surf = ax.plot_surface(Xi, Yi, zi, cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)        
        except:
            print(param1, param2)     
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    plt.show()