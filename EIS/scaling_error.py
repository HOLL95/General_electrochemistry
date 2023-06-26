import numpy as np
import matplotlib.pyplot as plt
results=np.load("scaling_errors_expanded.npy", allow_pickle=True).item()
keys=list(results.keys())
fig, ax=plt.subplots(2, 3)
for i in range(0, len(keys)):
    ax[i//3, i%3].hist(results[keys[i]]["scale"], bins=20)
    ax[i//3, i%3].set_title(keys[i])
plt.show()
fig, ax=plt.subplots(2, 3)
for i in range(0, len(keys)):
    ax[i//3, i%3].plot(results[keys[i]]["error"])
    ax[i//3, i%3].set_title(keys[i])
means=[np.mean(results[key]["error"]) for key in keys]
stds=[np.std(results[key]["error"]) for key in keys]
ax[-1, -1].bar(keys, means, yerr=stds)
plt.show()
fig, ax=plt.subplots(5, 5)
for i in range(0, 5):
    for j in range(0, 5):
        if j>i:
            ax[i,j].set_axis_off()
        elif i==j:
            ax[i,j].hist(results[keys[i]]["scale"])
            ax[i,j].set_title(keys[i])
        else:
            ax[i,j].scatter(results[keys[i]]["scale"], results[keys[j]]["scale"], s=2)
plt.show()
"""print(dict(zip(keys, means)))
for i in range(0, len(keys)):
    
    ax[i//3, i%3].hist(results[keys[i]])
    ax[i//3, i%3].set_title(keys[i])
ax[-1, -1].set_axis_off()
plt.show()
fig, ax=plt.subplots(5, 5)
for i in range(0, 5):
    for j in range(0, 5):
        if j>i:
            ax[i,j].set_axis_off()
        elif i==j:
            ax[i,j].hist(results[keys[i]])
            ax[i,j].set_title(keys[i])
        else:
            ax[i,j].scatter(results[keys[i]], results[keys[j]], s=2)
plt.show()"""