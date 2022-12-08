import numpy as np
sz=4
matrix = np.random.random((sz,sz))
for x in range(sz):
    for y in range(sz):
        matrix[x][y] = int(matrix[x][y]*10)
print(matrix)
print(np.max(np.abs(matrix), axis=1))