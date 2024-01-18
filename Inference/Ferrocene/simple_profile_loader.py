import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
import matplotlib.tri as mtri
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
with open(r"simple_profile.pickle", "rb") as input_file:
    monster_dict = cPickle.load(input_file)
keys=list(monster_dict.keys())
num_vars=int(np.sqrt(max(monster_dict[keys[0]].keys())+1))
import matplotlib.ticker as mticker

# My axis should display but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    return f"$10^{{%.2f}}$" % val  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation

EIS_params_5={'E_0': 0.19066872485338204,'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}

for key in monster_dict.keys():
    params=key.split("-")
    x=np.zeros(num_vars)
    y=np.zeros(num_vars)
    z=np.zeros((num_vars,num_vars))
    for i in range(0, num_vars):
        for j in range(0, num_vars):
            idx=(num_vars*i)+j
            param_dict=monster_dict[key][idx]["params"]
            x[i]=param_dict[params[0]]
            y[j]=param_dict[params[1]]
            z[j,i]=monster_dict[key][idx]["score"]
            
            #print(monster_dict[key][idx]["params"],monster_dict[key][idx]["score"])
    xaxis=np.log10(x)
    yaxis=np.log10(y)
    xi, yi=np.meshgrid(xaxis,yaxis)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    zi=np.log10(z)
    surf = ax.plot_surface(xi, yi, zi, cmap=cm.coolwarm,
                            linewidth=0, antialiased=True)        
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_zlabel("Score")
    for currax in [ax.xaxis, ax.yaxis, ax.zaxis]:
        currax.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        currax.set_major_locator(mticker.MaxNLocator(integer=True))
    get_x=np.log10(EIS_params_5[params[0]])
    get_y=np.log10(EIS_params_5[params[1]])
    ax.plot([get_x]*2, [get_y]*2, [np.min(zi), np.max(zi)], linestyle="--", color="black")
    plt.show()
        
    
