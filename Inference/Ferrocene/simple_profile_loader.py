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
plt.rcParams.update({'font.size': 14})
# My axis should display but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    return f"$10^{{%.1f}}$" % val  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation

EIS_params_5={'E_0': 0.19066872485338204,'k0_shape': 1.042945880414477, 'k0_scale': 0.9795762782576537, 'gamma': 1.95684990431219e-09, 'Cdl': 7.947339398582637e-06, 'alpha': 0.40751831983141673, 'Ru': 81.68485916975126, 'cpe_alpha_cdl': 0.7680042639799866, 'cpe_alpha_faradaic': 0.5461987862331081}
fig = plt.figure()
fig.set_size_inches(19, 8)
key_list=["E_0",  "Cdl", "Ru", "cpe_alpha_cdl", "k0_shape","k0_scale"]
log=[False, True, False, False, False, False]
var_params=["gamma", "k0_scale"]
symbol_list={"gamma":"$\\Gamma$","Cdl":"$Q_{C_{dl}}$", "Ru":"$R_u$", "cpe_alpha_cdl":"$\\alpha_{CPE}$", "k0_scale":"$\\log(k_0\\mu)$", "k0_shape": "$\\log(k_0\\sigma)$", "E_0":"$E^0$"}
plt.subplots_adjust(top=0.95,
                        bottom=0.05,
                        left=0.0,
                        right=0.86,

                        hspace=0.2,
                        wspace=0.5)
num_keys=len(key_list)
ax_list=[]
for m in range(0, len(var_params)):
    for q in range(0, len(key_list)):
        params=[var_params[m], key_list[q]]
        if m==1 and key_list[q]=="k0_scale":
            continue
        ax = fig.add_subplot(2, len(key_list), 1+q+(m*num_keys), projection='3d')
        ax.set_box_aspect([6, 6, 9])  # [width, height, depth]
        #ax.set_box_aspect((20, 16, 9))
        #ax.set_box_aspect(aspect = (1,1,2))
        key=("-").join(params)
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
        if m==0:
            xaxis=np.log10(x)
        else:
            xaxis=x
        if log[q]==True:
            yaxis=np.log10(y)
        else:
            yaxis=y
        xi, yi=np.meshgrid(xaxis,yaxis)
        
        zi=np.log10(z)
        surf = ax.plot_surface(xi, yi, zi, cmap=cm.coolwarm,
                                linewidth=0, antialiased=True)        
        
        #ax.set_zlabel("Score")
        if m==0:
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.tick_params(axis="x", pad=10)
        ax.tick_params(axis="z", pad=5)
        ax.set_xlabel(symbol_list[params[0]], labelpad=15)
        ax.set_ylabel(symbol_list[params[1]], labelpad=10)
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        #    currax.set_major_locator(mticker.MaxNLocator(integer=True))
        if log[q]==True:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        if m==1:
            ax.view_init(elev=30, azim=120)
            #axx=ax.get_axes()
            #axx.azim=60
        ax_list.append(ax)
        ax.set_position([-0.08+q*0.17, 0.58-(0.45*m), 0.3, 0.35]) 
        #get_x=np.log10(EIS_params_5[params[0]])
        #get_y=np.log10(EIS_params_5[params[1]])
        #ax.plot([get_x]*2, [get_y]*2, [np.min(zi), np.max(zi)], linestyle="--", color="black")
        #fig=plt.gcf()
        #filename=key+"_surfance.png"
    

plt.show()
fig.savefig("test.png", dpi=500)
    
        
    
