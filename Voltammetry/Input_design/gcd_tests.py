import numpy as np
sz=4
matrix = np.random.random((sz,sz))
for x in range(sz):
    for y in range(sz):
        matrix[x][y] = int(matrix[x][y]*10)
print(matrix)
z=np.array(np.ones(sz)*0.5)
z=[[0.5]*sz]
print(np.append(matrix,z, axis=1))