import time
import numpy as np
z=np.linspace(0, 18.4, int(1e6))
start=time.time()
np.sum(np.multiply(z, 2))

print(time.time()-start)
print(z)
