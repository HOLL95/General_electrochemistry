import numpy as np
import time 
start=time.time()
sz=50000
z=np.zeros((sz, sz))
val=0.05
for i in range(0, sz):
    z[i, i]=val
print(time.time()-start)
start=time.time()
np.dot(z, z)
print(time.time()-start)
start=time.time()
np.matmul(z,z)
print(time.time()-start)