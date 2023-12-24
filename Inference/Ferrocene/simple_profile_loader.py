import matplotlib.pyplot as plt
import numpy as np
import _pickle as cPickle
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
with open(r"someobject.pickle", "rb") as input_file:
    monster_dict = cPickle.load(input_file)
keys=list(monster_dict.keys())
num_vars=int(np.sqrt(max(monster_dict[keys[0]].keys())+1))

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
            z[i,j]=monster_dict[key][idx]["score"]
    plt.scatter(x, y)
    plt.show()
        
    
